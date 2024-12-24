#ifndef TCP_KIT_LOCK_FREE_SPSC_QUEUE_H
#define TCP_KIT_LOCK_FREE_SPSC_QUEUE_H

#include <memory>
#include <atomic>
#include <concurrent/queue.h>

namespace tcp_kit {

    template<typename T>
    class lock_free_spsc_queue: public queue<T> {

    private:
        struct node {
            std::unique_ptr<T> data;
            node* next;
            node(): next(nullptr) {}
        };

        std::atomic<node*> _head;
        std::atomic<node*> _tail;

        node* pop_head() {
            node* const old_head = _head.load();
            if(old_head == _tail.load()) {
                return nullptr;
            }
            _head.store(old_head->next);
            return old_head;
        }

    public:
        lock_free_spsc_queue(): _head(new node()), _tail(_head.load()) {}

        lock_free_spsc_queue(const lock_free_spsc_queue&) = delete;
        lock_free_spsc_queue(lock_free_spsc_queue&&) = delete;
        lock_free_spsc_queue& operator=(const lock_free_spsc_queue&) = delete;
        lock_free_spsc_queue& operator=(lock_free_spsc_queue&&) = delete;

        ~lock_free_spsc_queue() {
            while(node* const old_head = _head.load()) {
                _head.store(old_head->next);
                delete old_head;
            }
        }

        std::unique_ptr<T> pop() override {
            node* popped = pop_head();
            if(!popped) {
                return std::unique_ptr<T>();
            }
            std::unique_ptr<T> res(move(popped->data));
            delete popped;
            return res;
        }

        void push(T new_value) override {
            std::unique_ptr<T> new_data(std::make_unique<T>(new_value));
            node* p = new node;
            node* const old_tail = _tail.load();
            old_tail->data.swap(new_data);
            old_tail->next = p;
            _tail.store(p);
        }

    };

}

#endif
