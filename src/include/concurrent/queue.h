#ifndef TCP_KIT_QUEUE_H
#define TCP_KIT_QUEUE_H

#include <memory>

namespace tcp_kit {

    template<typename T>
    class queue {

    public:
        queue() = default;
        virtual ~queue() = default;

        virtual void push(T new_value) = 0;
        virtual std::unique_ptr<T> pop() = 0;

    };

}

#endif
