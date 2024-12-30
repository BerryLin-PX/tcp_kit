#pragma once

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
