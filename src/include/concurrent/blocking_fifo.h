#pragma once

#include <stdint.h>
#include <queue>
#include <mutex>
#include <utility>
#include <chrono>
#include "thread/interruptible_thread.h"
#include <logger/logger.h>

namespace tcp_kit {

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
        uint32_t       _size;
        std::deque<T>  _queue;
        std::mutex     _mutex;
        std::condition_variable_any  _not_full;
        std::condition_variable_any  _not_empty;

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
        std::unique_lock<std::mutex> lock(_mutex, std::defer_lock);
        if(lock.try_lock() && !full()) {
            _queue.push_back(el);
            return true;
        }
        return false;
    }

    template<typename T>
    void blocking_fifo<T>::push(const T& el) {
        std::unique_lock<std::mutex> lock(_mutex);
        while(full()) {
//            _not_full.wait(lock);
            interruptible_wait(_not_full, lock);
        }
        _queue.push_back(el);
        _not_empty.notify_one();
    }

    template<typename T>
    void blocking_fifo<T>::push(T&& el) {
        std::unique_lock<std::mutex> lock(_mutex);
        while(full()) {
//            _not_full.wait(lock);
            interruptible_wait(_not_full, lock);
        }
        _queue.push_back(std::forward<T>(el));
        _not_empty.notify_one();
    }

    template<typename T>
    bool blocking_fifo<T>::try_pop(T &out) {
        std::unique_lock<std::mutex> lock(_mutex, std::defer_lock);
        if(lock.try_lock() && !empty()) {
            out = _queue.front();
            _queue.pop_front();
            return true;
        }
        return false;
    }

    template<typename T>
    T blocking_fifo<T>::pop() {
        std::unique_lock<std::mutex> lock(_mutex);
        while (empty()) {
//            _not_empty.wait(lock);
            interruptible_wait(_not_empty, lock);
        }
        T pop_out = std::move(_queue.front());
        _queue.pop_front();
        _not_full.notify_one();
        return pop_out;
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
            std::unique_lock<std::mutex> lock(_mutex);
            out = move(_queue.front());
            _queue.pop_front();
            _not_full.notify_one();
            return true;
        }
    }

    template<typename T>
    template<typename Duration>
    bool blocking_fifo<T>::poll(T& out, Duration duration) {
        std::unique_lock<std::mutex> lock(_mutex);
        if(empty()) {
            interruptible_wait_for(_not_empty, lock, duration);
        }
        if(empty()) return false;
        else {
            out = std::move(_queue.front());
            _queue.pop_front();
            _not_full.notify_one();
            return true;
        }
    }

    template<typename T>
    bool blocking_fifo<T>::remove(const T& el) {
        std::unique_lock<std::mutex> lock(_mutex);
        auto it = std::find(_queue.begin(), _queue.end(), el);
        if (it != _queue.end()) {
            _queue.erase(it);
            _not_full.notify_one();
            return true;
        }
        return false;
    }

    template<>
    inline bool blocking_fifo<std::function<void()>>::remove(const std::function<void()>& el) { // ???
        std::unique_lock<std::mutex> lock(_mutex);
        auto it = find_if(_queue.begin(), _queue.end(), [&el](const std::function<void()>& task) {
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
