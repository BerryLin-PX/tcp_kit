#ifndef TCP_KIT_BLOCKING_QUEUE_H
#define TCP_KIT_BLOCKING_QUEUE_H

#include <stdint.h>
#include <queue>
#include <mutex>
#include <utility>

namespace tcp_kit {

    using namespace std;

    template <typename T>
    class blocking_queue {

    public:
        explicit blocking_queue(uint32_t size);
        inline bool empty();
        inline bool full();
        void push(const T &el);
        void push(T &&el);
        bool offer(const T &el);
        bool offer(T &&el);
        T pop();

        blocking_queue(const blocking_queue<T> &) = delete;
        blocking_queue<T> &operator=(const blocking_queue<T> &) = delete;

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
    void blocking_queue<T>::push(const T &el) {
        unique_lock<mutex> lock(_mutex);
        while(full())
            _not_full.wait(lock);
        _queue.push_back(el);
        _not_empty.notify_one();
    }

    template<typename T>
    void blocking_queue<T>::push(T &&el) {
        unique_lock<mutex> lock(_mutex);
        while(full())
            _not_full.wait(lock);
        _queue.push_back(move(el));
        _not_empty.notify_one();
    }

    template<typename T>
    T blocking_queue<T>::pop() {
        unique_lock<mutex> lock(_mutex);
        while (empty())
            _not_empty.wait(lock);
        T pop_out = move(_queue.front());
        _queue.pop_front();
        _not_full.notify_one();
        return move(pop_out);
    }

    template<typename T>
    bool blocking_queue<T>::offer(T &&el) {
        unique_lock<mutex> lock(_mutex);
        if(full()) return false;
        else {
            push(move(el));
            return true;
        }
    }

    template<typename T>
    bool blocking_queue<T>::offer(const T &el) {
        unique_lock<mutex> lock(_mutex);
        if(full()) return false;
        else {
            push(el);
            return true;
        }
    }

}

#endif