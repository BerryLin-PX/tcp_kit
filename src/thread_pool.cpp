// 参考自 Java JDK-11 中 java.util.concurrent.ThreadPoolExecutor 实现
#include <thread/thread_pool.h>
#include <stdexcept>
#include <memory>
#include <exception>
#include <logger/logger.h>

namespace tcp_kit {

    thread_local interrupt_flag this_thread_interrupt_flag;

    // worker
    thread_pool::worker::worker(thread_pool* tp, runnable first_task):
            _tp(tp),
            _state(-1),
            _exclusive_owner_thread(nullptr),
            _first_task(first_task),
            thread(make_shared<interruptible_thread>()) {

    }

    bool thread_pool::worker::try_lock() {
        bool locked = _mutex.try_lock();
        if(locked) {
            set_exclusive_owner_thread(this_thread::get_id());
            _state = 1;
        }
        return locked;
    }

    void thread_pool::worker::lock() {
        _mutex.lock();
        set_exclusive_owner_thread(this_thread::get_id());
        _state = 1;
    }

    void thread_pool::worker::unlock() {
        erase_exclusive_owner_thread();
        _state = 0;
        _mutex.unlock();
    }

    bool thread_pool::worker::held_exclusive() const {
        return _state != 0;
    }

    bool thread_pool::worker::locked() const {
        return held_exclusive();
    }

    void thread_pool::worker::erase_exclusive_owner_thread() {
        delete _exclusive_owner_thread;
        _exclusive_owner_thread = nullptr;
    }

    void thread_pool::worker::set_exclusive_owner_thread(thread::id thread_id) {
        delete _exclusive_owner_thread;
        _exclusive_owner_thread = new thread::id(thread_id);
    }

    thread_pool::worker::~worker() {
        delete _exclusive_owner_thread;
    }


    // thread_pool
    thread_pool::thread_pool(uint32_t core_pool_size,
                             uint32_t max_pool_size,
                             uint64_t keepalive_time,
                             unique_ptr<blocking_fifo<runnable>> work_fifo)
                             : _core_pool_size((core_pool_size > 0 && core_pool_size <= CAPACITY) ? core_pool_size : 1),
                               _max_pool_size((max_pool_size > 0 && max_pool_size >= core_pool_size && max_pool_size <= CAPACITY) ? max_pool_size : _core_pool_size),
                               _keepalive_time(keepalive_time), _work_fifo(move(work_fifo)), _ctl(RUNNING | 0) {

    }

    inline int32_t thread_pool::run_state_of(const int32_t c) {
        return c & ~CAPACITY;
    }

    inline bool thread_pool::run_state_at_least(int32_t c, int32_t s) {
        return c >= s;
    }

    inline bool thread_pool::run_state_less_than(int32_t c, int32_t s) {
        return c < s;
    }

    inline int32_t thread_pool::worker_count_of(int32_t ctl) {
        return ctl & CAPACITY;
    }

    inline int32_t thread_pool::ctl_of(int32_t rs, int32_t wc) {
        return rs | wc;
    }

    inline bool thread_pool::is_running(const int32_t c) {
        return run_state_of(c) == RUNNING;
    }

    void thread_pool::execute(runnable first_task) {
        if(!first_task) throw invalid_argument("Null Pointer Exception");
        int32_t c = _ctl.load();
        if(worker_count_of(c) < _core_pool_size) {
            if(add_worker(first_task, true))
                return;
            c = _ctl.load();
        }
        if(is_running(_ctl) && _work_fifo->offer(first_task)) {
            int32_t recheck = _ctl.load();
            if(!is_running(recheck) && remove(first_task))
                reject(first_task);
            else if(!worker_count_of(recheck))
                add_worker(nullptr, false);
        } else if(!add_worker(first_task, false))
            reject(first_task);
    }

    void thread_pool::await_termination() {
        unique_lock<recursive_mutex> main_lock(_mutex);
        if(run_state_less_than(_ctl.load(), TERMINATED))
            interruptible_wait(_termination, main_lock);
    }

    void thread_pool::shutdown() {
        {
            lock_guard<recursive_mutex> main_lock(_mutex);
            check_shutdown_access();
            advance_run_state(SHUTDOWN);
            interrupt_idle_workers();
            on_shutdown();
        }
        try_terminate();
    }

    bool thread_pool::is_shutdown() {
        return run_state_at_least(_ctl.load(), SHUTDOWN);
    }

    bool thread_pool::is_terminating() {
        int32_t c = _ctl.load();
        return run_state_at_least(c, SHUTDOWN) && run_state_less_than(c, TERMINATED);
    }

    bool thread_pool::is_terminated() {
        return run_state_at_least(_ctl.load(), TERMINATED);
    }

    bool thread_pool::add_worker(runnable first_task, bool core) {
        retry: // 当前状态是否允许添加新的线程
        for(;;) {
            int32_t c = _ctl.load();
            int32_t rs = run_state_of(c);
            if(rs >= SHUTDOWN && !(rs == SHUTDOWN && first_task == nullptr && !_work_fifo->empty()))
                return false;
            for(;;) {
                int wc = worker_count_of(c);
                if(wc >= CAPACITY || wc >= (core ? _core_pool_size : _max_pool_size))
                    return false;
                if(_ctl.exchange(c + 1) == c)
                    goto retry_end;
                c = _ctl.load();
                if(run_state_of(c) != rs)
                    goto retry;
            }
        }
        retry_end:
        bool work_started = false;
        bool work_added   = false;
        shared_ptr<worker> w;
        try {
            w = make_shared<worker>(this, first_task);
            //log_debug("New worker thread added");
            shared_ptr<interruptible_thread> t = w->thread;
            if(t) {
                t->set_runnable([this, w]{ run_worker(w); });
                {
                    lock_guard<recursive_mutex> main_lock(_mutex);
                    int32_t rs = run_state_of(_ctl.load());
                    if(rs < SHUTDOWN || (rs == SHUTDOWN && !first_task)) {
                        if(t->get_state() != interruptible_thread::state::NEW)
                            throw runtime_error("Illegal Thread State Exception");
                        _workers.insert(w);
                        auto s = _workers.size();
                        if(s > _largest_pool_size)
                            _largest_pool_size = s;
                        work_added = true;
                    }
                }
                if(work_added) {
                    t->start(); // TODO ?
                    work_started = true;
                }
            }
        } catch (...) {
            add_worker_failed(w);
            throw;
        }
        return work_started;
    }

    void thread_pool::run_worker(const shared_ptr<worker>& w) {
        shared_ptr<interruptible_thread> wt = w->thread;
        runnable task = w->_first_task;
        w->_first_task = nullptr;
        w->unlock();
        bool completed_abruptly = true;
        try {
            while(task || (task = get_task())) {
                //log_debug("Thread running");
                w->lock();
                if ((run_state_at_least(_ctl.load(), STOP) ||
                     (this_thread_interrupt_flag.is_set() &&
                      run_state_at_least(_ctl.load(), STOP))) &&
                    !this_thread_interrupt_flag.is_set())
                    this_thread_interrupt_flag.set();
                try {
                    before_execute(wt, task);
                    try {
                        task();
                        after_execute(wt, nullptr);
                    } catch (...) {
                        after_execute(wt, current_exception());
                    }
                } catch (...) {
                    task = nullptr;
                    w->completed_tasks++;
                    w->unlock();
                    throw;
                }
                task = nullptr;
                w->completed_tasks++;
                w->unlock();
            }
            completed_abruptly = false;
        } catch (...) {
            process_worker_exit(w, completed_abruptly);
            throw ;
        }
        process_worker_exit(w, completed_abruptly);
    }

    void thread_pool::before_execute(shared_ptr<interruptible_thread> &t, runnable r) { }
    void thread_pool::after_execute(shared_ptr<interruptible_thread> &t, const exception_ptr &exp) { }
    void thread_pool::terminated() { }
    void thread_pool::on_shutdown() { }

    runnable thread_pool::get_task() {
        bool timeout = false;
        for(;;) {
            int32_t c = _ctl.load();
            if(run_state_at_least(c, SHUTDOWN)
                && (run_state_at_least(c, STOP) || _work_fifo->empty())) {
                decrement_worker_count();
                return nullptr;
            }
            int32_t wc = worker_count_of(c);
            bool timed = _allow_core_thread_timeout || wc > _core_pool_size;
            if ((wc > _max_pool_size || (timed && timeout))
                && (wc > 1 || _work_fifo->empty())) {
                if(compare_and_decrement_worker_count(c))
                    return nullptr;
                continue;
            }
            try {
                runnable r;
                if(timed) {
                    _work_fifo->poll(r, chrono::nanoseconds(_keepalive_time));
                } else {
                    r = _work_fifo->pop();
                }
                if(r) return r;
                timeout = true;
            } catch (thread_interrupted& retry) {
                timeout = false;
            }
        }
    }

    void thread_pool::interrupt_idle_workers() {
        interrupt_idle_workers(false);
    }

    void thread_pool::interrupt_idle_workers(bool only_one) {
        lock_guard<recursive_mutex> main_lock(_mutex);
        for(const shared_ptr<worker>& w : _workers) {
            shared_ptr<interruptible_thread> thread = w->thread;
            if(!thread->flag->is_set() && w->try_lock()) {
                try {
                    thread->flag->set();
                } catch (...) {}
                w->unlock();
            }
            if(only_one)
                break;
        }
    }

    void thread_pool::add_worker_failed(const shared_ptr<worker>& w) {
        lock_guard<recursive_mutex> main_lock(_mutex);
        if(w) {
            _workers.erase(w);
            decrement_worker_count();
            try_terminate();
        }
    }

    void thread_pool::process_worker_exit(const shared_ptr<worker>& w, bool completed_abruptly) {
        //log_debug("On worker thread exit");
        if(completed_abruptly)
            decrement_worker_count();
        lock_guard<recursive_mutex> main_lock(_mutex);
        _completed_task_count += w->completed_tasks;
        _workers.erase(w);
        try_terminate();
        int32_t c = _ctl.load();
        if(run_state_less_than(c, STOP)) {
            uint32_t min = _allow_core_thread_timeout ? 0 : _core_pool_size;
            if (min == 0 && ! _work_fifo->empty())
                min = 1;
            if (worker_count_of(c) >= min)
                return;
        }
        add_worker(nullptr, false);
    }

    inline void thread_pool::decrement_worker_count() {
        _ctl.fetch_add(-1);
    }

    bool thread_pool::compare_and_decrement_worker_count(int32_t expect) {
        return _ctl.compare_exchange_weak(expect, expect - 1);
    }

    bool thread_pool::remove(runnable task) {
        bool removed = _work_fifo->remove(task);
        try_terminate();
        return removed;
    }

    void thread_pool::reject(runnable task) {
        // TODO
    }

    void thread_pool::advance_run_state(uint32_t target_state) {
        int32_t c = _ctl.load();
        while (!run_state_at_least(c, target_state)) {
            if(_ctl.compare_exchange_weak(c, ctl_of(target_state,worker_count_of(c))))
                break;
        }
    }

    void thread_pool::try_terminate() {
        for(;;) {
            int32_t c = _ctl.load();
            if (is_running(c) ||
                run_state_at_least(c, TIDYING) ||
                (run_state_less_than(c, STOP) && !_work_fifo->empty()))
                return;
            if (worker_count_of(c) != 0) {
                interrupt_idle_workers(ONLY_ONE);
                return;
            }
            lock_guard<recursive_mutex> main_lock(_mutex);
            if(_ctl.compare_exchange_weak(c, ctl_of(TIDYING, 0))) {
                try {
                    terminated();
                } catch (...) {
                    _ctl = ctl_of(TERMINATED, 0);
                    _termination.notify_all();
                    // TODO
                    // _container.close();
                    throw;
                }
                _ctl = ctl_of(TERMINATED, 0);
                _termination.notify_all();
            }
        }
    }

    void thread_pool::check_shutdown_access() {

    }

}
