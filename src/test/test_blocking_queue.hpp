#include <concurrent/blocking_queue.hpp>
#include <logger/logger.h>
#include <thread>

using namespace tcp_kit;
using namespace std;

void t1() {
    blocking_queue<int> b_queue(10);
    std::thread producer([&b_queue]() {
        for (int i = 0; i < 100; ++i) {
            int temp = i;
            b_queue.push(temp);
        }
    });
    // 创建消费者线程
    std::thread consumer([&b_queue]() {
        for (int i = 0; i < 100; ++i) {
            int value = b_queue.pop();
            log_debug("pop from queue: %d", value);
        }
    });
    producer.join();
    consumer.join();

}

void t2() {
    blocking_queue<string> queue(10);
    string str = "123";
    queue.push(str);
    queue.push(move(str));
}