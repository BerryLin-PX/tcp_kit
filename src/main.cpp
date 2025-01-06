#include <gtest/gtest.h>
#include <test/blocking_queue_test.hpp>
#include <test/interruptible_thread_test.hpp>
#include <test/thread_pool_test.hpp>
#include <test/thread_pool_worker_test.hpp>
#include <test/system_util_test.hpp>
#include <test/server_test.hpp>
#include <test/tcp_util_test.hpp>
#include <test/lock_free_queue_test.hpp>
#include <util/func_traits.h>
#include <network/filter_chain.h>

#define init_google_test InitGoogleTest

//int main() {
//    testing::init_google_test();
//    return RUN_ALL_TESTS();
//}


int main() {
    tcp_kit::lock_free_queue_test::t1();
    return 0;
}
