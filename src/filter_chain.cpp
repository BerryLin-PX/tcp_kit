#include <network/filter_chain.h>

namespace tcp_kit {

    evbuffer_holder::evbuffer_holder(evbuffer* buffer_): buffer(buffer_) { }

    void empty_connect_chain(ev_context* ctx) {

    }

    void empty_close_chain(ev_context* ctx) {

    }

    std::unique_ptr<msg_buffer> empty_process_chain(msg_context* ctx, std::unique_ptr<msg_buffer> in) {
        return in;
    }

    msg_buffer::msg_buffer(size_t size_): ptr((char*)malloc(size_)), size(size_) {}

    msg_buffer::msg_buffer(char *ptr_, size_t size_): ptr(ptr_), size(size_) {}

}
