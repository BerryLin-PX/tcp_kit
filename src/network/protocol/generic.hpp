#ifndef TCP_KIT_GENERIC_HPP
#define TCP_KIT_GENERIC_HPP

#include <network/server.hpp>
#include <logger/logger.h>
#include <event2/event.h>
#include <event2/listener.h>
#include <event2/bufferevent.h>

namespace tcp_kit {

    // 作为 server 的通用协议实现
    //
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

    void generic::ev_handler::listener_callback(evconnlistener* listener, socket_t fd, sockaddr* address, int socklen, void* arg) {
        //log_debug("New event_context");
        generic::ev_handler* ev_handler = (generic::ev_handler *)(arg);
        bufferevent *bev = bufferevent_socket_new(ev_handler->_ev_base, fd, BEV_OPT_CLOSE_ON_FREE);
        if(!bev) {
            log_error("Failed to allocate the bufferevent");
            close_socket(fd);
            return;
        }
        event_context ctx{fd, address, socklen, bev};
        if(!ev_handler->invoke_conn_filters(ctx)) {
            goto err;
        }
        ev_handler->register_read_write_filters(ctx);
        bufferevent_enable(bev, EV_READ | EV_WRITE);
        bufferevent_setcb(bev,
                          read_callback, write_callback, event_callback,
                          ev_handler);
        err:
            bufferevent_free(bev);
    }

    void generic::ev_handler::read_callback(bufferevent *bev, void *arg) {

    }

    void generic::ev_handler::write_callback(bufferevent *bev, void *arg) {

    }

    void generic::ev_handler::event_callback(bufferevent *bev, short what, void *arg) {

    }

    generic::ev_handler::ev_handler() {
        _ev_base = event_base_new();
    }

    void generic::ev_handler::init(server_base* server_ptr) {
        _server = static_cast<server<generic>*>(server_ptr);
        sockaddr_in sin = socket_address(_server->port);
        _evc = evconnlistener_new_bind(
                _ev_base, listener_callback, this,
                LEV_OPT_REUSEABLE | LEV_OPT_CLOSE_ON_FREE | LEV_OPT_REUSEABLE_PORT,
                -1, (sockaddr*) &sin, sizeof(sin));
    }

    void generic::ev_handler::run() {
        event_base_dispatch(_ev_base);
    }

    generic::ev_handler::~ev_handler() {
        evconnlistener_free(_evc);
        event_base_free(_ev_base);
    }

    void generic::handler::init(server_base* server_ptr) {

    }

    void generic::handler::run() {

    }

}

#endif