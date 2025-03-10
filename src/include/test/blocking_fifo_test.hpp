#ifndef TCP_KIT_BLOCKING_QUEUE_TEST_H
#define TCP_KIT_BLOCKING_QUEUE_TEST_H

#include <gtest/gtest.h>
#include <thread>
#include <chrono>
#include "concurrent/blocking_fifo.h"
#include <logger/logger.h>

using namespace tcp_kit;

// 测试1：单个线程的基本入队和出队操作
TEST(blocking_queue_tests, single_thread_basic_operations) {
    blocking_fifo<int> queue(5);  // 队列大小为5

    // 测试入队操作
    queue.push(1);
    EXPECT_FALSE(queue.empty());
    EXPECT_EQ(queue.pop(), 1);  // 测试出队操作

    EXPECT_TRUE(queue.empty());
}

// 测试2：多线程生产者-消费者测试
TEST(blocking_queue_tests, multi_thread_producer_consumer) {
    blocking_fifo<int> queue(10);

    auto producer = [&queue]() {
        for (int i = 0; i < 10; ++i) {
            queue.push(i);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));  // 模拟生产过程
        }
    };

    auto consumer = [&queue]() {
        for (int i = 0; i < 10; ++i) {
            int val = queue.pop();
            EXPECT_GE(val, 0);  // 检查值是否正确
        }
    };

    std::thread t1(producer);
    std::thread t2(consumer);

    t1.join();
    t2.join();
}

// 测试3：阻塞和通知机制测试
TEST(blocking_queue_tests, blocking_and_notification) {
    blocking_fifo<int> queue(1);  // 设置队列大小为1，确保在满时会阻塞

    auto producer = [&queue]() {
        queue.push(1);  // 应该成功
        queue.push(2);  // 队列已满，此时应阻塞
    };

    auto consumer = [&queue]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));  // 等待一段时间，让生产者线程被阻塞
        EXPECT_EQ(queue.pop(), 1);  // 消费第一个元素
        EXPECT_EQ(queue.pop(), 2);  // 消费第二个元素，解锁生产者线程
    };

    std::thread t1(producer);
    std::thread t2(consumer);

    t1.join();
    t2.join();
}

// 测试4：测试超时机制
TEST(blocking_queue_tests, poll_with_timeout) {
    blocking_fifo<int> queue(5);

    auto consumer = [&queue]() {
        int val;
        bool success = queue.poll(val, std::chrono::milliseconds(100));  // 在队列为空的情况下尝试poll
        EXPECT_FALSE(success);  // 应该超时返回false
    };

    std::thread t1(consumer);
    t1.join();
}

// 测试5：测试删除功能
TEST(blocking_queue_tests, remove_element) {
    blocking_fifo<int> queue(5);
    queue.push(1);
    queue.push(2);
    queue.push(3);

    EXPECT_TRUE(queue.remove(2));  // 删除中间的元素
    EXPECT_FALSE(queue.remove(4));  // 尝试删除不存在的元素
    EXPECT_EQ(queue.pop(), 1);  // 验证队列状态
    EXPECT_EQ(queue.pop(), 3);
}

namespace tcp_kit {

    namespace blocking_fifo_test {

        void push_pop_test() {
            blocking_fifo<int> queue(10);
            std::thread t1([&queue]() {
                for (int i = 0; i < 5; ++i) {
                    queue.push(i);
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));  // 模拟生产过程
                }
            });
            std::thread t2([&queue]() {
                for (int i = 0; i < 5; ++i) {
                    int val = queue.pop();
                    log_info("%d", val);
                }
            });
            t1.join();
            t2.join();
        }

        void try_test() {
            blocking_fifo<int> queue(3);
            int pop_out;
            if(!queue.try_pop(pop_out)) {
                log_info("The queue is empty");
            }
            queue.push(1);
            queue.push(2);
            queue.push(3);
            if(!queue.try_push(4)) {
                log_info("The queue is full");
            }
        }

        void multi_thread_test() {
            blocking_fifo<uint32_t> queue(10);
            uint32_t n_producer = 10, n_consumer = 10, n_element = 1000;
            std::atomic<uint32_t> sum{0};
            std::condition_variable latch;
            std::mutex mutex;
            bool ready = false;

            auto producer = [&] {
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    latch.wait(lock, [&] { return ready; });
                }
                for (uint32_t i = 1; i <= n_element; ++i) {
                    queue.push(i);
                }
            };

            auto consumer = [&] {
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    latch.wait(lock, [&] { return ready; });
                }
                for (uint32_t i = 1; i <= n_element; ++i) {
                    sum += queue.pop();
                }
            };
            std::vector<std::thread> producers, consumers;
            for (uint32_t i = 0; i < n_producer; ++i) {
                producers.emplace_back(producer);
            }
            for (uint32_t i = 0; i < n_consumer; ++i) {
                consumers.emplace_back(consumer);
            }
            {
                std::lock_guard<std::mutex> lock(mutex);
                ready = true;
            }
            latch.notify_all();
            for (auto& p : producers) {
                p.join();
            }
            for (auto& c : consumers) {
                c.join();
            }
            log_info("Expected sum: %d, Actual sum: %d",  n_producer * n_element * (n_element + 1) / 2, sum.load());
        }


        void performance_test(uint32_t num_elements) {
            blocking_fifo<int> fifo(100);  // 设置一个合理的队列大小

            auto start_time = std::chrono::high_resolution_clock::now();

            std::thread producer([&]() {
                for (uint32_t i = 0; i < num_elements; ++i) {
                    fifo.push(i);
                }
            });

            std::thread consumer([&]() {
                for (uint32_t i = 0; i < num_elements; ++i) {
                    fifo.pop();
                }
            });

            producer.join();
            consumer.join();

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            std::cout << "元素数量: " << num_elements << ", 用时: " << duration << " ms" << std::endl;
        }

        void m_performance_test(uint32_t num_elements, uint32_t num_producers, uint32_t num_consumers) {
            blocking_fifo<int> fifo(100);  // 设置一个合理的队列大小

            auto start_time = std::chrono::high_resolution_clock::now();

            std::vector<std::thread> producers, consumers;

            // 计算每个生产者应处理的元素数
            uint32_t elements_per_producer = num_elements / num_producers;
            uint32_t remaining_elements = num_elements % num_producers; // 处理除不尽的情况

            uint32_t start = 0;
            for (uint32_t i = 0; i < num_producers; ++i) {
                uint32_t count = elements_per_producer + (i < remaining_elements ? 1 : 0); // 让前面的生产者多处理一个
                producers.emplace_back([&, start, count]() {
                    for (uint32_t j = 0; j < count; ++j) {
                        fifo.push(start + j);
                    }
                });
                start += count;
            }

            // 计算每个消费者应消费的元素数
            uint32_t elements_per_consumer = num_elements / num_consumers;
            remaining_elements = num_elements % num_consumers;

            for (uint32_t i = 0; i < num_consumers; ++i) {
                uint32_t count = elements_per_consumer + (i < remaining_elements ? 1 : 0);
                consumers.emplace_back([&, count]() {
                    for (uint32_t j = 0; j < count; ++j) {
                        fifo.pop();
                    }
                });
            }

            for (auto &producer : producers) producer.join();
            for (auto &consumer : consumers) consumer.join();

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            std::cout << "元素数量: " << num_elements << ", 用时: " << duration << " ms" << std::endl;
        }


    }


}


#endif