#include <network/filter_chain.h>

namespace tcp_kit {

    evbuffer_holder::evbuffer_holder(evbuffer* buffer_): buffer(buffer_) { }

    void empty_connect_chain(ev_context* ctx) {

    }

    std::unique_ptr<evbuffer_holder> empty_process_chain(ev_context* ctx, std::unique_ptr<evbuffer_holder> in) {
        return in;
    }

}
