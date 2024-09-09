#ifndef TCP_KIT_THREAD_POOL_WORKER_TEST_H
#define TCP_KIT_THREAD_POOL_WORKER_TEST_H

#include <gtest/gtest.h>
#include <thread/thread_pool.h>
#include <concurrent/blocking_queue.hpp>

using namespace std;
using namespace tcp_kit;

//TEST(worker_tests, lock_and_unlock) {
//    blocking_queue<runnable> q(10);
//    thread_pool* tp = new thread_pool(4, 4, 1000, &q);
//    runnable first_task = []() { };
//    thread_pool::worker worker_instance(tp, first_task);
//
//    worker_instance.lock();
//    ASSERT_TRUE(worker_instance.locked());
//
//    worker_instance.unlock();
//    ASSERT_FALSE(worker_instance.locked());
//
//    delete tp;
//}

#endif