#include <gtest/gtest.h>
#include <test/blocking_queue_test.hpp>
#include <test/interruptible_thread_test.hpp>
#include <test/thread_pool_test.hpp>
#include <test/thread_pool_worker_test.hpp>

#define init_google_test InitGoogleTest

//int main() {
//    testing::init_google_test();
//    return RUN_ALL_TESTS();
//}

int main() {
    tcp_kit::thread_pool_test::t1();
    return 0;
}