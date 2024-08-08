#include <thread/thread_pool.h>
#include <logger/logger.h>
#include <stdexcept>
#include <memory>
#include <exception>

#pragma clang diagnostic push
namespace tcp_kit {

    // worker
    thread_pool::worker::worker(thread_pool *tp, runnable first_task):
            _tp(tp),
            _state(-1),
            _exclusive_owner_thread(nullptr),
            _first_task(first_task),
            thread(make_shared<interruptible_thread>([this] { _tp->run_worker(this); })) {

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
                             blocking_queue<runnable> *const work_queue)
                             : _core_pool_size((core_pool_size > 0 && core_pool_size <= CAPACITY) ? core_pool_size : 1),
                               _max_pool_size((max_pool_size > 0 && max_pool_size >= core_pool_size && max_pool_size <= CAPACITY) ? max_pool_size : _core_pool_size),
                               _keepalive_time(keepalive_time), _work_queue(work_queue), _ctl(RUNNING | 0) {

    }

    inline int32_t thread_pool::run_state_of(const int32_t c) {
        return c & ~CAPACITY;
    }

    inline bool thread_pool::run_state_at_least(int32_t c, int32_t s) {
        return c >= s;
    }

    inline int32_t thread_pool::work_count(const int32_t c) {
        return c & CAPACITY;
    }

    inline bool thread_pool::is_running(const int32_t c) {
        return run_state_of(c) == RUNNING;
    }

    void thread_pool::execute(runnable first_task) {
        if(!first_task) throw invalid_argument("Null Pointer Exception");
        int32_t c = _ctl.load();
        if(work_count(c) < _core_pool_size) {
            if(add_worker(first_task, true))
                return;
            c = _ctl.load();
        }
        if(is_running(_ctl) && _work_queue->offer(first_task)) {
            int32_t recheck = _ctl.load();
            if(!is_running(recheck) && remove(first_task))
                reject(first_task);
            else if(!work_count(recheck))
                add_worker(nullptr, false);
        } else if(!add_worker(first_task, false))
            reject(first_task);
    }

    bool thread_pool::add_worker(runnable first_task, bool core) {
        retry: // 当前状态是否允许添加新的线程
        for(;;) {
            int32_t c = _ctl.load();
            int32_t rs = run_state_of(c);
            if(rs >= SHUTDOWN && !(rs == SHUTDOWN && first_task == nullptr && !_work_queue->empty()))
                return false;
            for(;;) {
                int wc = work_count(c);
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
            shared_ptr<interruptible_thread> t = w->thread;
            if(t) {
                {
                    lock_guard<recursive_mutex> lock(_mutex);
                    int32_t rs = run_state_of(_ctl.load());
                    if(rs < SHUTDOWN || (rs == SHUTDOWN && !first_task)) {
                        if(t->get_state() == interruptible_thread::state::NEW)
                            throw runtime_error("Illegal Thread State Exception");
                        _workers.insert(w);
                        auto s = _workers.size();
                        if(s > _largest_pool_size)
                            _largest_pool_size = s;
                        work_added = true;
                    }
                }
                if(work_added) {
                    t->start();
                    work_started = true;
                }
            }
        } catch (...) {
            add_worker_failed(w);
            throw;
        }
        return work_started;
    }

    void thread_pool::run_worker(worker *w) {
        shared_ptr<interruptible_thread> wt = w->thread;
        runnable task = w->_first_task;
        w->_first_task = nullptr;
        w->unlock();
        bool completed_abruptly = true;
        try {
            while(task || (task = get_task())) {
                w->lock();
                if((run_state_at_least(_ctl.load(), STOP)
                    || (this_thread_interrupt_flag.is_set()
                       && run_state_at_least(_ctl.load(), STOP)))
                   && !wt->interrupt_flag->is_set())
                    wt->interrupt_flag->set();
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

    void thread_pool::before_execute(shared_ptr<interruptible_thread>& t, runnable r) {

    }

    void thread_pool::after_execute(shared_ptr<interruptible_thread> &t, const exception_ptr &exp) {

    }

    runnable thread_pool::get_task() {

    }

    void thread_pool::process_worker_exit(worker *w, bool completed_abruptly) {

    }

    bool thread_pool::remove(runnable task) {
        return false;
    }

    void thread_pool::reject(runnable task) {

    }

    void thread_pool::add_worker_failed(shared_ptr<worker> worker) {

    }

}
#pragma clang diagnostic pop