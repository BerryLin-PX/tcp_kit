#ifndef TCP_KIT_BLOCKING_QUEUE_H
#define TCP_KIT_BLOCKING_QUEUE_H

#include <stdint.h>
#include <queue>
#include <mutex>
#include <utility>
#include <chrono>
#include <thread/interruptible_thread.h>

namespace tcp_kit {

    using namespace std;

    template <typename T>
    class blocking_queue {

    public:
        explicit blocking_queue(uint32_t size);
        inline bool empty();
        inline bool full();
        void push(const T& el);
        void push(T&& el);
        T pop();
        bool offer(const T& el);
        bool offer(T&& el);
        bool poll(T& out);
        template<typename Duration> bool poll(T& out, Duration duration);
        bool remove(const T& el);


        blocking_queue(const blocking_queue<T>&) = delete;
        blocking_queue<T>& operator=(const blocking_queue<T>&) = delete;

    private:
        uint32_t           _size;
        deque<T>           _queue;
        mutex              _mutex;
        condition_variable _not_full;
        condition_variable _not_empty;

    };

    template<typename T>
    blocking_queue<T>::blocking_queue(uint32_t size): _size(size) {

    }

    template<typename T>
    inline bool blocking_queue<T>::empty() {
        return _queue.empty();
    }

    template<typename T>
    inline bool blocking_queue<T>::full() {
        return _queue.size() == _size;
    }

    template<typename T>
    void blocking_queue<T>::push(const T& el) {
        unique_lock<mutex> lock(_mutex);
        while(full()) {
            interruption_point();
            _not_full.wait(lock);
            interruption_point();
        }
        _queue.push_back(el);
        _not_empty.notify_one();
    }

    template<typename T>
    void blocking_queue<T>::push(T&& el) {
        unique_lock<mutex> lock(_mutex);
        while(full()) {
            interruption_point();
            _not_full.wait(lock);
            interruption_point();
        }
        _queue.push_back(move(el));
        _not_empty.notify_one();
    }

    template<typename T>
    T blocking_queue<T>::pop() {
        unique_lock<mutex> lock(_mutex);
        while (empty()) {
            interruption_point();
            _not_empty.wait(lock);
            interruption_point();
        }
        T pop_out = move(_queue.front());
        _queue.pop_front();
        _not_full.notify_one();
        return move(pop_out);
    }

    template<typename T>
    bool blocking_queue<T>::offer(T&& el) {
        if(full()) return false;
        else {
            push(move(el));
            return true;
        }
    }

    template<typename T>
    bool blocking_queue<T>::offer(const T& el) {
        if(full()) return false;
        else {
            push(el);
            return true;
        }
    }

    template<typename T>
    bool blocking_queue<T>::poll(T &out) {
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
    bool blocking_queue<T>::poll(T &out, Duration duration) {
        unique_lock<mutex> lock(_mutex);
        if(empty()) {
            interruption_point();
            _not_empty.wait_for(lock, duration);
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
    bool blocking_queue<T>::remove(const T &el) {
        unique_lock<mutex> lock(_mutex);
        auto it = std::find(_queue.begin(), _queue.end(), el);
        if (it != _queue.end()) {
            _queue.erase(it);
            _not_full.notify_one();
            return true;
        }
        return false;
    }


}

#endif