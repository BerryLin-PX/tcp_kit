#ifndef TCP_KIT_THREAD_POOL_TEST_H
#define TCP_KIT_THREAD_POOL_TEST_H

#include <iostream>
#include <gtest/gtest.h>
#include <thread/thread_pool.h>
#include <logger/logger.h>

namespace tcp_kit {

    namespace thread_pool_test {

        using namespace std;

        void t1() {
            thread_pool tp(4, 10, 1000, make_unique<blocking_fifo<runnable>>(1));
            atomic<bool> condition;
            atomic<unsigned> count(0);
            for(int i = 0; i < 10; ++i) {
                tp.execute([&condition, &count] {
                    while(!condition.load());
                    log_info("count: %d", ++count);
                });
            }
            condition = true;
            while(count.load() != 10);
            this_thread::sleep_for(chrono::seconds(1));
            log_debug("TO SHUTDOWN THE THREAD POOL");
            tp.shutdown();
            log_debug("THREAD POOL SHUTDOWN INVOKED");
            tp.await_termination();
        }

        void t2() {
            thread_pool tp(4, 4, 1000, make_unique<blocking_fifo<runnable>>(10));
            tp.execute([] {
//        GTEST_LOG_(INFO) << "TASK RUNNING";
            });
        }

        void execute_task_test() {
            unique_ptr<blocking_fifo<runnable>> work_queue(new blocking_fifo<runnable>(3));
            thread_pool pool(2, 4, 1000, std::move(work_queue));
            std::atomic<int> counter{0};
            const int tasks = 10;
            for (int i = 0; i < tasks; ++i) {
                pool.execute([&counter] { ++counter; });
            }
            pool.shutdown();
            pool.await_termination();
            log_info("Count of task: %d", counter.load());
        }

        std::atomic<uint32_t> thread_counter(0);
        class count_thread { public: count_thread() { ++thread_counter; } };
        thread_local count_thread count;

        void core_size_limit_test() {
            unique_ptr<blocking_fifo<runnable>> work_queue(new blocking_fifo<runnable>(3));
            thread_pool pool(2, 2, 1000, std::move(work_queue));
            for (int i = 0; i < 1000; ++i) {
                pool.execute([]{ count; });
            }
            pool.shutdown();
            pool.await_termination();
            log_info("Count of thread: %d", thread_counter.load());
        }

        void shutdown_test() {
            unique_ptr<blocking_fifo<runnable>> work_queue(new blocking_fifo<runnable>(3));
            thread_pool pool(2, 4, 1000, std::move(work_queue));
            for (int i = 0; i < 10; ++i) {
                pool.execute([]{});
            }
            pool.shutdown();
            pool.await_termination();
            pool.execute([]{});
        }

    }

}


#endif