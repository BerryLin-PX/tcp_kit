#ifndef TCP_KIT_STRING_MSG_H
#define TCP_KIT_STRING_MSG_H

#include <network/filter.h>

namespace tcp_kit {

    namespace filters {

        bufferevent_filter_result string_msg_read(evbuffer* src, evbuffer* dst,ev_ssize_t dst_limit,
                                                  bufferevent_flush_mode mode, void* ctx);

        bufferevent_filter_result string_msg_write(evbuffer* src, evbuffer* dst,ev_ssize_t dst_limit,
                                                  bufferevent_flush_mode mode, void* ctx);

        extern const filter string_msg;

    }

}

#endif
