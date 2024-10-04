#ifndef TCP_KIT_BLOCKING_QUEUE_TEST_H
#define TCP_KIT_BLOCKING_QUEUE_TEST_H

#include <gtest/gtest.h>
#include <thread>
#include <chrono>
#include <concurrent/blocking_fifo.hpp>

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

#endif