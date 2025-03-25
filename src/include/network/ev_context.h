#pragma once

#include <event2/bufferevent.h>
#include <event2/event.h>
#include <util/tcp_util.h>
#include <network/server.h>

namespace tcp_kit {

    class ev_handler_base;
    class handler_base;

    struct ev_context {
        static const uint8_t CONNECTED  = 0; // 连接建立
        static const uint8_t READY      = 1; // 准备就绪
        static const uint8_t ACTIVE     = 2; // 工作中
        static const uint8_t CLOSING    = 3; // 因主动或被动关闭连接
        static const uint8_t CLOSED     = 4; // 连接已关闭, bev 不再可用
        static const uint8_t TERMINATED = 5; // 终结

        struct control {
            unsigned error:   1;
            unsigned state:   3;
            unsigned n_async: 12;
        };

        control          ctl;
        socket_t         fd;
        sockaddr        *address;
        int              socklen;
        uint32_t         conn_id;
        ev_handler_base *ev_handler;
        handler_base*    handler;
        bufferevent*     bev;

        ev_context(const ev_context &) = delete;
        ev_context(ev_context &&) = delete;
        ev_context& operator=(const ev_context &) = delete;
        ev_context& operator=(ev_context &&) = delete;

    };

}
