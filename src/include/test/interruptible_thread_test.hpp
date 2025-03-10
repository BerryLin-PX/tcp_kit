#ifndef TCP_KIT_INTERRUPTIBLE_THREAD_TEST_H
#define TCP_KIT_INTERRUPTIBLE_THREAD_TEST_H

#include <gtest/gtest.h>
#include <logger/logger.h>
#include <thread/interruptible_thread.h>
#include <ctime>

using namespace std;
using namespace tcp_kit;

namespace tcp_kit {

    namespace interruptible_thread_test {

        void interruption_test() {
            interruptible_thread thread([] {
                try{
                    for(;;)
                        interruption_point(); // 检查中断信号
                } catch (thread_interrupted) {
                    log_error("The thread is interrupted");
                }
            });
            thread.start();     // 启动线程
            thread.flag->set(); // 中断线程
            thread.join();
        }

        void wait_test() {
            std::mutex m;
            std::unique_lock<std::mutex> lock(m);
            std::condition_variable_any condition;
            interruptible_thread t1([&]{
                try {
                    log_info("Will block");
                    interruptible_wait(condition, lock);
                } catch (thread_interrupted) {
                    log_error("The thread is interrupted");
                }
            });
            t1.start();
            t1.flag->set();
            t1.join();
        }

        void print_timestamp(const char* message) {
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts); // 获取当前时间（秒 + 纳秒）
            log_info("%s: %ld", message, ts.tv_sec);
        }

        void wait_for_test() {
            std::mutex m;
            std::unique_lock<std::mutex> lock(m);
            std::condition_variable_any condition;
            interruptible_thread t1([&]{
                print_timestamp("Will block");
                interruptible_wait_for(condition, lock, std::chrono::seconds(1));
                print_timestamp("Thread recovery run");
            });
            t1.start();
            t1.join();
        }

    }

}

#endif