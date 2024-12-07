#ifndef TCP_KIT_ANY_H
#define TCP_KIT_ANY_H

#include <type_traits>
#include <typeinfo>

class any
{
public:
    template<class T>
    using storage_t = typename std::decay<T>::type;

    template<typename U,
            class=typename std::enable_if<!std::is_same<storage_t<U>, any>::value, void>::type>
    any(U&& _value): ptr(new container<storage_t<U>>(std::forward<U>(_value))) { }

    any(): ptr(nullptr) { }

    any(const any& _that): ptr(_that.clone()) { }

    any(any&& _that): ptr(_that.ptr) {
        _that.ptr = nullptr;
    }

    template<class U>
    inline bool is() const {
        typedef storage_t<U> T;

        if (!ptr)
            return false;

        auto derived = dynamic_cast<container<T>*> (ptr);

        return derived != nullptr;
    }

    template<class U>
    inline storage_t<U>& as() const {
        typedef storage_t<U> T;

        if (!ptr)
            throw std::bad_cast();

        auto container_ = dynamic_cast<container<T>*> (ptr);

        if (!container_)
            throw std::bad_cast();

        return container_->value;
    }

    template<class U>
    inline storage_t<U>& value() const {
        return as<storage_t<U>>();
    }

    template<class U>
    inline operator U() const {
        return as<storage_t<U>>();
    }

    any& operator=(const any& a) {
        if (ptr == a.ptr)
            return *this;
        auto old_ptr = ptr;
        ptr = a.clone();
        if (old_ptr)
            delete old_ptr;
        return *this;
    }

    any& operator=(any&& a) {
        if (ptr == a.ptr)
            return *this;
        std::swap(ptr, a.ptr);
        return *this;
    }

    ~any() {
        if (ptr)
            delete ptr;
    }

private:
    class container_base {
    public:
        virtual ~container_base() {}
        virtual container_base* clone() const = 0;
    };

    template<typename T>
    class container: public container_base
    {
    public:
        template<typename U>
        container(U&& value): value(std::forward<U>(value)) { }

        inline container_base* clone() const {
            return new container<T>(value);
        }

        T value;
    };

    inline container_base* clone() const {
        if (ptr)
            return ptr->clone();
        else
            return nullptr;
    }

    container_base* ptr;
};


#endif