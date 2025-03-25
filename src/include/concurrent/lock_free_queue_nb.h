#pragma once
#include <stdint.h>
#include <atomic>
#include <memory>
#include <logger/logger.h>
#include <thread>

namespace tcp_kit {

    template <typename T>
    class lock_free_queue_nb {
    private:
        struct node;
        struct counted_node_ptr {
            int     external_count;
            node*   ptr;
            uint8_t padding[sizeof(void *) - sizeof(int)];
        };

        std::atomic<counted_node_ptr> _head;
        std::atomic<counted_node_ptr> _tail;
        std::atomic<uint32_t>         _count;

        struct node_counter {
            /*unsigned*/ /* ??? */ signed internal_count : 30;
            unsigned external_counters : 2;
        };

        struct flag_ptr {
            T*      ptr;
            bool    flag;
            uint8_t padding[sizeof(void *) - sizeof(bool)];
        };

        struct node {
//            std::atomic<T *> data;
            std::atomic<flag_ptr> data;
            std::atomic<node_counter> count;
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
                if (!new_counter.internal_count && !new_counter.external_counters){
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
            node *const current_tail_ptr = old_tail.ptr;
            while (!_tail.compare_exchange_weak(old_tail, new_tail) && old_tail.ptr == current_tail_ptr);
            if (old_tail.ptr == current_tail_ptr) {
                free_external_counter(old_tail);
            } else {
                current_tail_ptr->release_ref();
            }
        }

    public:
        lock_free_queue_nb(): _count(0), _head({1, new node, {0}}), _tail(_head.load()) {}

        void push(T new_value) {
            std::unique_ptr<T> new_data(new T(new_value));
            counted_node_ptr new_next{1, new node, {0}};
            counted_node_ptr old_tail = _tail.load();
            for (;;) {
                increase_external_count(_tail, old_tail);
                // T *old_data = nullptr;
                if (/*old_tail.ptr->data.compare_exchange_strong(old_data, new_data.get())*/ old_tail.ptr->set_data(new_data.get())) { // 如果在这里执行之前另一个线程入队又另一个线程出队，又将 data 指针改为 nullptr, 导致 ABA 问题发生
                    counted_node_ptr old_next{0, nullptr, {0}};
                    if (!old_tail.ptr->next.compare_exchange_strong(old_next, new_next)) {
                        delete new_next.ptr;
                        new_next = old_next;
                    } else {
                        ++_count;
                    }
                    set_new_tail(old_tail, new_next);
                    new_data.release();
                    break;
                } else {
                    counted_node_ptr old_next{0, nullptr, {0}};
                    if (old_tail.ptr->next.compare_exchange_strong(old_next, new_next)) {
                        old_next = new_next;
                        new_next.ptr = new node;
                        ++_count;
                    } else {
                    }
                    set_new_tail(old_tail, old_next);
                }
            }
        }

        std::unique_ptr<T> pop() {
            counted_node_ptr old_head = _head.load(std::memory_order_relaxed);
            for (;;) {
                increase_external_count(_head, old_head);
                node *const ptr = old_head.ptr;
                if (ptr == _tail.load().ptr) {
                    ptr->release_ref();
                    log_info("count: %d", _count.load());
                    return std::unique_ptr<T>();
                }
                counted_node_ptr next = ptr->next.load();
                if (_head.compare_exchange_strong(old_head, next)) {
//                    T *const res = ptr->data.exchange(nullptr);
                    T* const res = ptr->release_data();
                    free_external_counter(old_head);
                    return std::unique_ptr<T>(res);
                }
                ptr->release_ref();
            }
        }

        ~lock_free_queue_nb() {
            while(pop());
        }

        uint32_t get_count() {
            return _count.load();
        }

    };

}