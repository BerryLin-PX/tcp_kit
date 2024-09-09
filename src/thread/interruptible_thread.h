#ifndef TCP_KIT_INTERRUPTIBLE_THREAD_H
#define TCP_KIT_INTERRUPTIBLE_THREAD_H

#include <memory>
#include <thread>
#include <condition_variable>
#include <mutex>

namespace tcp_kit {

    using namespace std;

    class thread_interrupted: std::exception {};

    class interrupt_flag {
    public:
        interrupt_flag() noexcept;
        void set();
        bool is_set();
        void set_condition_variable(condition_variable& cv);
        void clear_condition_variable();
        template<typename Lockable> void wait(condition_variable_any& cv, Lockable& lk);
        struct clear_cv_on_destruct { ~clear_cv_on_destruct(); };

    private:
        atomic<bool>             _flag;
        condition_variable*      _thread_cond;
        condition_variable_any*  _thread_cond_any;
        mutex                    _set_clear_mutex;

    };

    class interruptible_thread {
    public:
        interrupt_flag*     flag;

        enum state { NEW, ALIVE, TERMINATED };
        explicit interruptible_thread(function<void()> task = nullptr);
        void set_runnable(function<void()> task);
        void start();
        void join();
        state get_state();

        interruptible_thread(const interruptible_thread&) = delete;
        interruptible_thread& operator=(const interruptible_thread&) = delete;

    private:
        function<void()>    _runnable;
        unique_ptr<thread>  _internal_thread;
        state               _state;

    };

    extern thread_local interrupt_flag this_thread_interrupt_flag;

    void interruption_point();

    template <typename Lockable>
    void interruptible_wait(condition_variable& cv, Lockable& lk) {
        this_thread_interrupt_flag.wait(cv, lk);
    }

    template<typename Lockable>
    void interrupt_flag::wait(condition_variable_any& cv, Lockable& lk) {
        struct custom_lock {
            interrupt_flag*  self;
            Lockable&        lk;

            custom_lock(interrupt_flag* self_,
                        condition_variable_any& cond,
                        Lockable& lk_): self(self), lk(lk_) {
                self->_set_clear_mutex.lock();
                self->_thread_cond_any = &cond;
            }

            void unlock() {
                lk.unlock();
                self->_set_clear_mutex.unlock();
            }

            void lock() {
                std::lock(self->_set_clear_mutex, lk);
            }

            ~custom_lock() {
                self->_thread_cond_any = 0;
                self->_set_clear_mutex.unlock();
            }
        };

        custom_lock cl(this, cv, lk);
        interruption_point();
        cv.wait(lk);
        interruption_point();
    }

}
#endif