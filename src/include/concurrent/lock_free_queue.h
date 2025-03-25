#pragma once

#include <atomic>
#include <memory>
#include <concurrent/queue.h>
#include <logger/logger.h>
#include <stdint.h>
#include <thread/interruptible_thread.h>
#include <iostream>

namespace tcp_kit {

    // TODO: 有内存泄漏问题
    template<typename T>
    class lock_free_queue: public queue<T> {
    private:
        struct node;
        struct counted_node_ptr {
            int     external_count;
            node*   ptr;
            uint8_t padding[sizeof(void*) - sizeof(int)];
        };

        std::atomic<counted_node_ptr> _head;
        std::atomic<counted_node_ptr> _tail;
        std::atomic<uint32_t>         _size;
        std::mutex                    _mutex;
        std::condition_variable_any   _not_empty;


        struct node_counter {
            signed   internal_count    : 30;
            unsigned external_counters : 2;
        };

        struct flag_ptr {
            T*      ptr;
            bool    flag;
            uint8_t padding[sizeof(void *) - sizeof(bool)];
        };

        struct node {
            std::atomic<flag_ptr>         data;
            std::atomic<node_counter>     count;
            std::atomic<counted_node_ptr> next;

            node(): data({nullptr, false, {0}}), count({0, 2}), next({0, nullptr, {0}}) {}

            void release_ref() {
                node_counter old_counter = count.load(std::memory_order_relaxed);
                node_counter new_counter;
                do {
                    new_counter = old_counter;
                    --new_counter.internal_count;
                } while (!count.compare_exchange_strong(old_counter, new_counter,
                                                        std::memory_order_acquire,
                                                        std::memory_order_relaxed));
                if (!new_counter.internal_count && !new_counter.external_counters) {
                    delete this;
                }
            }

            bool set_data(T* ptr) {
                flag_ptr expected = {nullptr, false, {0}};
                flag_ptr desired = {ptr, true, {0}};
                return data.compare_exchange_strong(expected, desired);
            }

            T* release_data() {
                flag_ptr released = data.load();
                T* ptr = released.ptr;
                released.ptr = nullptr;
                data.store(released);
                return ptr;
            }

        };

        static void increase_external_count(std::atomic<counted_node_ptr>& counter,
                                            counted_node_ptr& old_counter) {
            counted_node_ptr new_counter;
            do {
                new_counter = old_counter;
                ++new_counter.external_count;
            } while (!counter.compare_exchange_strong(old_counter, new_counter,
                                                      std::memory_order_acquire,
                                                      std::memory_order_relaxed));
            old_counter.external_count = new_counter.external_count;
        }

        static void free_external_counter(counted_node_ptr& old_node_ptr) {
            node* const ptr = old_node_ptr.ptr;
            int const count_increase = old_node_ptr.external_count - 2;
            node_counter old_counter = ptr->count.load(std::memory_order_relaxed);
            node_counter new_counter;
            do {
                new_counter = old_counter;
                --new_counter.external_counters;
                new_counter.internal_count += count_increase;
            } while (!ptr->count.compare_exchange_strong(old_counter, new_counter,
                                                         std::memory_order_acquire,
                                                         std::memory_order_relaxed));
            if (!new_counter.internal_count && !new_counter.external_counters) {
                delete ptr;
            }
        }

        void set_new_tail(counted_node_ptr& old_tail,
                          counted_node_ptr const& new_tail) {
            node* const current_tail_ptr = old_tail.ptr;
            while(!_tail.compare_exchange_weak(old_tail, new_tail) && old_tail.ptr == current_tail_ptr);
            if (old_tail.ptr == current_tail_ptr) {
                for(;;) {
                    uint32_t old_size = _size.load();
                    if (old_size == 0) {
                        std::unique_lock<std::mutex> lock(_mutex);
                        if (_size.fetch_add(1) == 0) {
                            _not_empty.notify_all();
                        }
                        break;
                    } else if (_size.compare_exchange_strong(old_size, old_size + 1)) {
                        break;
                    }
                }
                free_external_counter(old_tail);
            } else {
                current_tail_ptr->release_ref();
            }
        }

    public:
        lock_free_queue(): _size(0), _head({1, new node, {0}}), _tail(_head.load()) {}

        void push(T new_value) {
            std::unique_ptr<T> new_data(new T(new_value));
            counted_node_ptr new_next{1, new node, {0}};
            counted_node_ptr old_tail = _tail.load();
            for (;;) {
                increase_external_count(_tail, old_tail);
                if (old_tail.ptr->set_data(new_data.get())) {
                    counted_node_ptr old_next{0, nullptr, {0}};
                    if (!old_tail.ptr->next.compare_exchange_strong(old_next, new_next)) {
                        delete new_next.ptr;
                        new_next = old_next;
                    }
                    set_new_tail(old_tail, new_next);
                    new_data.release();
                    break;
                } else {
                    counted_node_ptr old_next{0, nullptr, {0}};
                    if (old_tail.ptr->next.compare_exchange_strong(old_next, new_next)) {
                        old_next = new_next;
                        new_next.ptr = new node;
                    }
                    set_new_tail(old_tail, old_next);
                }
            }
        }

        std::unique_ptr<T> pop() {
            for(;;) {
                uint32_t old_size = _size.load();
                if(old_size == 0) {
                    std::unique_lock<std::mutex> lock(_mutex);
                    if(_size.load() == 0) {
                        interruptible_wait_for(_not_empty, lock, std::chrono::seconds(3));
                    }
                    continue;
                }
                if(_size.compare_exchange_strong(old_size, old_size - 1)) {
                    counted_node_ptr old_head = _head.load(std::memory_order_relaxed);
                    for(;;) {
                        increase_external_count(_head, old_head);
                        node* const ptr = old_head.ptr;
                        counted_node_ptr next = ptr->next.load();
                        if (_head.compare_exchange_strong(old_head, next)) {
                            T* const res = ptr->release_data();
                            free_external_counter(old_head);
                            return std::unique_ptr<T>(res);
                        }
                        ptr->release_ref();
                    }
                }
            }
        }

    };

}
