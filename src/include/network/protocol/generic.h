#ifndef TCP_KIT_GENERIC_H
#define TCP_KIT_GENERIC_H

#include <network/server.h>
#include <logger/logger.h>
#include <event2/event.h>
#include <event2/listener.h>
#include <event2/bufferevent.h>

namespace tcp_kit {

    // 作为 server 的通用协议实现
    class generic {
    public:
        class ev_handler: public ev_handler_base {
        public:
            ev_handler();
            ~ev_handler();

            static void listener_callback(evconnlistener* listener, socket_t fd, sockaddr* address, int socklen, void* arg);
            static void read_callback(bufferevent *bev, void *arg);
            static void write_callback(bufferevent *bev, void *arg);
            static void event_callback(bufferevent *bev, short what, void *arg);

        protected:
            server<generic>* _server;
            event_base*      _ev_base;
            mutex            _mutex;
            evconnlistener*  _evc;

            void init(server_base* server_ptr) override;
            void run() override;

        };

        class handler: public handler_base {
        public:
            handler() = default;

        protected:
            void init(server_base* server_ptr) override;
            void run() override;

        };

    };

}

#endif