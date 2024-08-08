#ifndef TCP_KIT_THREAD_POOL_H
#define TCP_KIT_THREAD_POOL_H

#include <stdint.h>
#include <chrono>
#include <vector>
#include <utility>
#include <thread>
#include <atomic>
#include <mutex>
#include <unordered_set>
#include <concurrent/blocking_queue.hpp>
#include <thread/interruptible_thread.h>
//#include <thread/worker.h>

#define COUNT_BITS 29

namespace tcp_kit {

    using namespace std;

    typedef void (*runnable)();

    class thread_pool {

    public:
        explicit thread_pool(uint32_t core_pool_size,
                             uint32_t max_pool_size,
                             uint64_t keepalive_time,
                             blocking_queue<runnable> *work_queue);
        void execute(runnable);

        thread_pool(const thread_pool &) = delete;
        thread_pool &operator=(const thread_pool &) = delete;

    private:
        class worker {
        public:
            volatile uint64_t        completed_tasks;
            shared_ptr<interruptible_thread> thread;

            explicit worker(thread_pool *tp, runnable first_task);

            bool try_lock();
            void lock();
            void unlock();
            bool locked() const;
            bool held_exclusive() const;
            void erase_exclusive_owner_thread();
            void set_exclusive_owner_thread(thread::id thread_id);

            ~worker();

            worker(const worker &) = delete;
            worker &operator=(const worker &) = delete;

        private:
            thread_pool             *_tp;
            mutex                    _mutex;
            volatile int8_t          _state;
            thread::id              *_exclusive_owner_thread;
            runnable                 _first_task;
            friend thread_pool;

        };

        static const int32_t CAPACITY   = (1 << COUNT_BITS) - 1;
        static const int32_t RUNNING    = -1 << COUNT_BITS;
        static const int32_t SHUTDOWN   =  0 << COUNT_BITS;
        static const int32_t STOP       =  1 << COUNT_BITS;
        static const int32_t TIDYING    =  2 << COUNT_BITS;
        static const int32_t TERMINATED =  3 << COUNT_BITS;
        std::atomic<int32_t> _ctl;

        recursive_mutex _mutex;
        const uint32_t  _core_pool_size;
        const uint32_t  _max_pool_size;
        const uint64_t  _keepalive_time;
        uint32_t        _largest_pool_size;
        blocking_queue<runnable> *const   _work_queue;
        unordered_set<shared_ptr<worker>> _workers;

        static int32_t run_state_of(int32_t c);
        static bool run_state_at_least(int32_t c, int32_t s);
        static int32_t work_count(int32_t ctl);
        static bool is_running(int32_t c);
        bool add_worker(runnable first_task, bool core);
        void run_worker(worker *w);
        void before_execute(shared_ptr<interruptible_thread> &t, runnable r);
        void after_execute(shared_ptr<interruptible_thread> &t, const exception_ptr& exp);
        runnable get_task();
        void process_worker_exit(worker *w, bool completed_abruptly);
        void add_worker_failed(shared_ptr<worker> worker);
        bool remove(runnable task);
        void reject(runnable task);
    };


}

#endif
