#include <logger/logger.h>
#include <concurrent/lock_free_queue.h>

namespace tcp_kit {

    namespace lock_free_queue_test {

        void t1_producer(tcp_kit::lock_free_queue<int>& queue, int start, int count) {
            for (int i = start; i < start + count; ++i) {
                queue.push(i);
            }
        }

        void t1_consumer(tcp_kit::lock_free_queue<int>& queue, std::atomic<int>& sum, int consume_count) {
            for (int i = 0; i < consume_count;) {
                auto value = queue.pop();
                if (value) {
                    sum += *value;
                    ++i;
                }
            }
        }

        void t1() {
            tcp_kit::lock_free_queue<int> queue;
            std::atomic<int> sum(0);

            int num_producers = 4;
            int num_consumers = 4;
            int items_per_producer = 1000;

            std::vector<std::thread> producers;
            std::vector<std::thread> consumers;

            // 启动生产者线程
            for (int i = 0; i < num_producers; ++i) {
                producers.emplace_back(t1_producer, std::ref(queue), i * items_per_producer, items_per_producer);
            }

            // 启动消费者线程
            for (int i = 0; i < num_consumers; ++i) {
                consumers.emplace_back(t1_consumer, std::ref(queue), std::ref(sum), (num_producers * items_per_producer) / num_consumers);
            }

            // 等待所有生产者线程完成
            for (auto& t : producers) {
                t.join();
            }

            // 等待所有消费者线程完成
            for (auto& t : consumers) {
                t.join();
            }

            // 检查结果
            int expected_sum = (num_producers * items_per_producer * (items_per_producer * num_producers - 1)) / 2;
            std::cout << "Expected sum: " << expected_sum << "\n";
            std::cout << "Actual   sum: " << sum.load() << "\n";

            if (sum.load() == expected_sum) {
                std::cout << "Test passed: Queue is thread-safe.\n";
            } else {
                std::cout << "Test failed: Queue is not thread-safe.\n";
            }
        }

    }

}