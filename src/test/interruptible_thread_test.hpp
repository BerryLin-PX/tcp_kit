#ifndef TCP_KIT_INTERRUPTIBLE_THREAD_TEST_H
#define TCP_KIT_INTERRUPTIBLE_THREAD_TEST_H

#include <gtest/gtest.h>
#include <thread/interruptible_thread.h>

using namespace std;
using namespace tcp_kit;

TEST(interruptible_thread_tests, interrupt_the_thread) {
    bool interrupted = false;
    interruptible_thread thread([&interrupted] {
        try{
            for(;;)
                interruption_point();
        } catch (thread_interrupted) {
            interrupted = true;
        }
    });

    thread.start();
    thread.flag->set();
    thread.join();

    EXPECT_TRUE(interrupted);
}

#endif