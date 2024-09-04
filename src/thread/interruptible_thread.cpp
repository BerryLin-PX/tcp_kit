#include <thread/interruptible_thread.h>

#include <utility>
#include <future>

namespace tcp_kit {

    using namespace std;

    // interrupt_flag
    interrupt_flag::interrupt_flag() noexcept: _thread_cond(0), _thread_cond_any(0) { }

    void interrupt_flag::set() {
        _flag.store(true, std::memory_order_relaxed);
        lock_guard<mutex> lk(_set_clear_mutex);
        if(_thread_cond)
            _thread_cond->notify_all();
        else if(_thread_cond_any)
            _thread_cond_any->notify_all();
    }

    bool interrupt_flag::is_set() {
        return _flag.load(std::memory_order_relaxed);
    }

    void interrupt_flag::set_condition_variable(condition_variable& cv) {
        lock_guard<mutex> lk(_set_clear_mutex);
        _thread_cond = &cv;
    }

    void interrupt_flag::clear_condition_variable() {
        lock_guard<mutex> lk(_set_clear_mutex);
        _thread_cond = 0;
    }

    interrupt_flag::clear_cv_on_destruct::~clear_cv_on_destruct() {
        this_thread_interrupt_flag.clear_condition_variable();
    }

    inline void interruption_point() {
        if(this_thread_interrupt_flag.is_set())
            throw thread_interrupted();
    }

    // interruptible_thread
    interruptible_thread::interruptible_thread(function<void()> runnable): _runnable((move(runnable))), _state(NEW) {

    }

    void interruptible_thread::set_runnable(function<void()> runnable) {
        this->_runnable = move(runnable);
    }

    void interruptible_thread::start() {
        promise<interrupt_flag*> p;
        _internal_thread = unique_ptr<thread>(new thread(([this, &p] {
            p.set_value(&this_thread_interrupt_flag);
            _state = ALIVE;
            try {
                _runnable();
            } catch (...) {
                _state = TERMINATED;
                throw;
            }
            _state = TERMINATED;
        })));
        interrupt_flag = p.get_future().get();
    }

    interruptible_thread::state interruptible_thread::get_state() {
        return _state;
    }

}