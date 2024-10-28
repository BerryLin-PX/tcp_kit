#ifndef TCP_KIT_BLOCKING_QUEUE_H
#define TCP_KIT_BLOCKING_QUEUE_H

#include <stdint.h>
#include <queue>
#include <mutex>
#include <utility>
#include <chrono>
#include "thread/interruptible_thread.h"

namespace tcp_kit {

    using namespace std;

    template <typename T>
    class blocking_fifo {

    public:
        explicit blocking_fifo(uint32_t size);
        inline bool empty();
        inline bool full();
        bool try_push(const T& el);
        void push(const T& el);
        void push(T&& el);
        bool try_pop(T& out);
        T pop();
        bool offer(const T& el);
        bool offer(T&& el);
        bool poll(T& out);
        template<typename Duration> bool poll(T& out, Duration duration);
        bool remove(const T& el);

        blocking_fifo(const blocking_fifo<T>&) = delete;
        blocking_fifo<T>& operator=(const blocking_fifo<T>&) = delete;

    private:
        uint32_t  _size;
        deque<T>  _queue;
        mutex     _mutex;
        condition_variable_any  _not_full;
        condition_variable_any  _not_empty;

    };

    template<typename T>
    blocking_fifo<T>::blocking_fifo(uint32_t size): _size(size) {

    }

    template<typename T>
    inline bool blocking_fifo<T>::empty() {
        return _queue.empty();
    }

    template<typename T>
    inline bool blocking_fifo<T>::full() {
        return _queue.size() == _size;
    }

    template<typename T>
    bool blocking_fifo<T>::try_push(const T& el) {
        unique_lock<mutex> lock(_mutex, defer_lock);
        if(lock.try_lock() && !full()) {
            _queue.push_back(el);
            return true;
        }
        return false;
    }

    template<typename T>
    void blocking_fifo<T>::push(const T& el) {
        unique_lock<mutex> lock(_mutex);
        while(full()) {
            interruption_point();
//            _not_full.wait(lock);
            interruptible_wait(_not_full, lock);
            interruption_point();
        }
        _queue.push_back(el);
        _not_empty.notify_one();
    }

    template<typename T>
    void blocking_fifo<T>::push(T&& el) {
        unique_lock<mutex> lock(_mutex);
        while(full()) {
            interruption_point();
//            _not_full.wait(lock);
            interruptible_wait(_not_full, lock);
            interruption_point();
        }
        _queue.push_back(move(el));
        _not_empty.notify_one();
    }

    template<typename T>
    bool blocking_fifo<T>::try_pop(T& out) {
        unique_lock<mutex> lock(_mutex, defer_lock);
        if(lock.try_lock() && !empty()) {
            out = move(_queue.front());
            _queue.pop_front();
            return true;
        }
        return false;
    }

    template<typename T>
    T blocking_fifo<T>::pop() {
        unique_lock<mutex> lock(_mutex);
        while (empty()) {
            interruption_point();
//            _not_empty.wait(lock);
            interruptible_wait(_not_empty, lock);
            interruption_point();
        }
        T pop_out = move(_queue.front());
        _queue.pop_front();
        _not_full.notify_one();
        return move(pop_out);
    }

    template<typename T>
    bool blocking_fifo<T>::offer(T&& el) {
        if(full()) return false;
        else {
            push(move(el));
            return true;
        }
    }

    template<typename T>
    bool blocking_fifo<T>::offer(const T& el) {
        if(full()) return false;
        else {
            push(el);
            return true;
        }
    }

    template<typename T>
    bool blocking_fifo<T>::poll(T& out) {
        if(empty()) return false;
        else {
            unique_lock<mutex> lock(_mutex);
            out = move(_queue.front());
            _queue.pop_front();
            _not_full.notify_one();
            return true;
        }
    }

    template<typename T>
    template<typename Duration>
    bool blocking_fifo<T>::poll(T& out, Duration duration) {
        unique_lock<mutex> lock(_mutex);
        if(empty()) {
            interruption_point();
//            _not_empty.wait_for(lock, duration);
            interruptible_wait_for(_not_empty, lock, duration);
            interruption_point();
        }
        if(empty()) return false;
        else {
            out = move(_queue.front());
            _queue.pop_front();
            _not_full.notify_one();
            return true;
        }
    }

    template<typename T>
    bool blocking_fifo<T>::remove(const T& el) {
        unique_lock<mutex> lock(_mutex);
        auto it = std::find(_queue.begin(), _queue.end(), el);
        if (it != _queue.end()) {
            _queue.erase(it);
            _not_full.notify_one();
            return true;
        }
        return false;
    }

    template<>
    inline bool blocking_fifo<function<void()>>::remove(const function<void()>& el) { // ???
        unique_lock<mutex> lock(_mutex);
        auto it = find_if(_queue.begin(), _queue.end(), [&el](const function<void()>& task) {
            return el.target_type() == task.target_type() && el.target<void()>() == task.target<void()>();
        });
        if (it != _queue.end()) {
            _queue.erase(it);
            _not_full.notify_one();
            return true;
        }
        return false;
    }

}

#endif