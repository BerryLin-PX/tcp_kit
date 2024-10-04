// 参考自 Linux 内核 kfifo 实现
// 仅在单生产者-单消费者时保证线程安全
#ifndef TCP_KIT_LOCK_FREE_QUEUE_H
#define TCP_KIT_LOCK_FREE_QUEUE_H
#include <cstddef>

namespace tcp_kit {

    using namespace std;

    // @N: 队列最少容纳 N 个元素, 由于队列将内部存储空间向上取最少所需容量 2 的幂, 实际容量可能会大于 N
    //     假设元素 int32(4个字节), N 为 10, 最少需要 40 个字节, 但因为向上取整, 实际容量 64 个字节,
    //     也就是队列实际可以容纳 16 个 int32 元素
    template<typename T, uint32_t N>
    class lock_free_fifo {

    public:
        explicit lock_free_fifo();
        uint32_t push_by_shallow_copy(const T* el);
        uint32_t pop(T* out);

        lock_free_fifo(const lock_free_fifo<T, N>&) = delete;
        lock_free_fifo(lock_free_fifo<T, N>&&) = delete;
        lock_free_fifo<T, N>& operator=(const lock_free_fifo<T, N>&) = delete;

    private:
        static constexpr uint32_t roundup_pow_of_two(uint32_t x);
        static constexpr uint32_t _min = sizeof(T) * N;
        static constexpr uint32_t _size = (_min & (_min - 1)) ? roundup_pow_of_two(_min) : _min;
        uint8_t  _buffer[_size];
        uint32_t _in;
        uint32_t _out;

    };

    template<typename T, uint32_t N>
    constexpr uint32_t lock_free_fifo<T, N>::roundup_pow_of_two(uint32_t x) {
        x -= 1;
        int r = 32;

        if (!x)
            return 0;
        if (!(x & 0xffff0000u)) {
            x <<= 16;
            r -= 16;
        }
        if (!(x & 0xff000000u)) {
            x <<= 8;
            r -= 8;
        }
        if (!(x & 0xf0000000u)) {
            x <<= 4;
            r -= 4;
        }
        if (!(x & 0xc0000000u)) {
            x <<= 2;
            r -= 2;
        }
        if (!(x & 0x80000000u)) {
            x <<= 1;
            r -= 1;
        }
        return 1 << r;
    }

    template<typename T, uint32_t N>
    lock_free_fifo<T, N>::lock_free_fifo(): _in(0), _out(0) { }

    // 仅对入队元素进行浅拷贝, 若元素持有指向堆内存的指针, 那么这个函数可能引发内存安全问题
    // @return: 本次入队写入的字节数, 无法入队时写入的字节数小于元素大小
    template<typename T, uint32_t N>
    uint32_t lock_free_fifo<T, N>::push_by_shallow_copy(const T* el) {
        uint8_t* el_ptr = (uint8_t*) el;
        uint32_t len = sizeof(T);
        uint32_t l;
        len = min(len, _size - _in + _out);
        l = min(len, _size - (_in & (_size - 1)));
        memcpy(_buffer + (_in & (_size - 1)), el_ptr, l);
        memcpy(_buffer, el_ptr + l, len - l);
        _in += len;
        return len;
    }

    // @return: 写出的字节数, 无法出队时写出的字节数小于元素大小
    template<typename T, uint32_t N>
    uint32_t lock_free_fifo<T, N>::pop(T* out) {
        uint8_t* out_buffer = (uint8_t*) out;
        uint32_t len = sizeof(T);
        uint32_t l;
        len = min(len, _in - _out);
        l = min(len, _size - (_out & (_size - 1)));
        memcpy(out_buffer, _buffer + (_out & (_size - 1)), l);
        memcpy(out_buffer + l, _buffer, len - l);
        _out += len;
        return len;
    }

}

#endif