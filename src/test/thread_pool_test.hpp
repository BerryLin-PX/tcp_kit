#ifndef TCP_KIT_THREAD_POOL_TEST_H
#define TCP_KIT_THREAD_POOL_TEST_H

#include <iostream>
#include <gtest/gtest.h>
#include <thread/thread_pool.h>
#include <logger/logger.h>

TEST(thread_pool_tests, execute) {
    thread_pool tp(4, 4, 1000, make_unique<blocking_fifo<runnable>>(10));
    tp.execute([] {
//        GTEST_LOG_(INFO) << "TASK RUNNING";
    });
}

TEST(thread_pool_tests, exectue_fill_task) {
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

    }

}


#endif