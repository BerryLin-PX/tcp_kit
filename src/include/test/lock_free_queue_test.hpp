#include <logger/logger.h>
#include <concurrent/lock_free_queue.h>

namespace tcp_kit {

    namespace lock_free_queue_test {

        void t1_producer(tcp_kit::lock_free_queue<int>* queue, int start, int count) {
            try{
                for (int i = start; i < start + count; ++i) {
                    queue->push(i);
                }
            } catch (...) {
                log_error("ERROR");
            }
        }

        void t1_consumer(tcp_kit::lock_free_queue<int>* queue, std::atomic<int>& sum, int consume_count) {
            try {
                for (int i = 0; i < consume_count; ++i) {
                    auto value = queue->pop();
                    if (value) {
                        sum += *value;
                    } else {
                        log_info("sum: %d", sum.load());
                    }
                }

            } catch (...) {
                log_error("ERROR");
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
                producers.emplace_back(t1_producer, &queue, i * items_per_producer, items_per_producer);
            }

            // 启动消费者线程
            for (int i = 0; i < num_consumers; ++i) {
                consumers.emplace_back(t1_consumer, &queue, std::ref(sum), (num_producers * items_per_producer) / num_consumers);
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

        void print_timestamp(int el) {
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts); // 获取当前时间（秒 + 纳秒）
            log_info("Pop out: %d - %ld", el, ts.tv_sec);
        }

        void blocking_test() {
            lock_free_queue<int> queue;
            queue.push(1);
            queue.push(2);
            std::thread t([&]{
                for(;;)
                    print_timestamp(*queue.pop());
            });
            std::this_thread::sleep_for(chrono::seconds(3));
            queue.push(3);
            t.join();
        }

        void mtt_producer(tcp_kit::lock_free_queue<int>* queue, int start, int count) {
            for (int i = start; i < start + count; ++i)
                queue->push(i);
        }

        void mtt_consumer(tcp_kit::lock_free_queue<int>* queue, std::atomic<int>& sum, int consume_count) {
            for (int i = 0; i < consume_count; ++i) {
                sum += *queue->pop();
            }
        }

        void multi_thread_test() {
            tcp_kit::lock_free_queue<int> queue;
            std::atomic<int> sum(0);
            int num_producers = 4; int num_consumers = 4; int items_per_producer = 1000;
            std::vector<std::thread> producers;
            std::vector<std::thread> consumers;
            auto producer = [](tcp_kit::lock_free_queue<int>* queue, int start, int count) {
                for (int i = start; i < start + count; ++i)
                    queue->push(i);
            };
            auto consumer = [](tcp_kit::lock_free_queue<int>* queue, std::atomic<int>& sum, int consume_count) {
                for (int i = 0; i < consume_count; ++i) {
                    sum += *queue->pop();
                }
            };
            for (int i = 0; i < num_producers; ++i) {
                producers.emplace_back(producer, &queue, i * items_per_producer, items_per_producer);
            }
            for (int i = 0; i < num_consumers; ++i) {
                consumers.emplace_back(consumer, &queue, std::ref(sum), (num_producers * items_per_producer) / num_consumers);
            }
            for (auto& t : producers) {
                t.join();
            }
            for (auto& t : consumers) {
                t.join();
            }
            log_info("Expected sum: %d, Actual sum: %d",  (num_producers * items_per_producer * (items_per_producer * num_producers - 1)) / 2, sum.load());
        }

        void spsc_performance_test(uint32_t num_elements) {
            lock_free_spsc_queue<int> queue;
            auto start_time = std::chrono::high_resolution_clock::now();
            std::thread producer([&]() {
                for (uint32_t i = 0; i < num_elements; ++i) {
                    queue.push(i);
                }
            });
            std::thread consumer([&]() {
                for (uint32_t i = 0; i < num_elements; ++i) {
                    *queue.pop();
                }
            });

            producer.join();
            consumer.join();

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            std::cout << "元素数量: " << num_elements << ", 用时: " << duration << " ms" << std::endl;
        }

        void performance_test(uint32_t num_elements) {
            lock_free_queue<int> queue;
            std::vector<int> data(num_elements);
            for (uint32_t i = 0; i < num_elements; ++i) {
                data[i] = i;
            }

            auto start_time = std::chrono::high_resolution_clock::now();

            std::thread producer([&]() {
                for (auto el : data) {
                    queue.push(el);
                }
            });

            std::thread consumer([&]() {
                for (uint32_t i = 0; i < num_elements; ++i) {
                    queue.pop();
                }
            });

            producer.join();
            consumer.join();

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            std::cout << "元素数量: " << num_elements << ", 用时: " << duration << " ms" << std::endl;
        }

        void m_performance_test(uint32_t num_elements, uint32_t num_producers, uint32_t num_consumers) {
            lock_free_queue<int> queue;
            std::vector<int> data(num_elements);
            for (uint32_t i = 0; i < num_elements; ++i) {
                data[i] = i;
            }

            auto start_time = std::chrono::high_resolution_clock::now();

            std::vector<std::thread> producers;
            std::vector<std::thread> consumers;

            uint32_t elements_per_producer = num_elements / num_producers;
            uint32_t elements_per_consumer = num_elements / num_consumers;

            for (uint32_t i = 0; i < num_producers; ++i) {
                uint32_t start = i * elements_per_producer;
                uint32_t end = (i == num_producers - 1) ? num_elements : start + elements_per_producer;
                producers.emplace_back([&, start, end]() {
                    for (uint32_t j = start; j < end; ++j) {
                        queue.push(data[j]);
                    }
                });
            }

            for (uint32_t i = 0; i < num_consumers; ++i) {
                consumers.emplace_back([&]() {
                    for (uint32_t j = 0; j < elements_per_consumer; ++j) {
                        queue.pop();
                    }
                });
            }

            for (auto& producer : producers) {
                producer.join();
            }
            for (auto& consumer : consumers) {
                consumer.join();
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            std::cout << "元素数量: " << num_elements  << ", 用时: " << duration << " ms" << std::endl;
        }

    }

}