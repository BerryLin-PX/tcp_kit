#ifndef TCP_KIT_GENERIC_HPP
#define TCP_KIT_GENERIC_HPP

#include <network/server.hpp>
#include <logger/logger.h>
#include <event2/event.h>
#include <event2/listener.h>
#include <event2/bufferevent.h>

namespace tcp_kit {

    class generic {
    public:
        class ev_handler: ev_handler_base<generic> {
        public:
            ev_handler();
            ~ev_handler();

            static void listener_callback(evconnlistener* listener, socket_t fd, sockaddr* address, int socklen, void* arg);
            static void read_callback(bufferevent *bev, void *arg);
            static void write_callback(bufferevent *bev, void *arg);
            static  void event_callback(bufferevent *bev, short what, void *arg);

        protected:
            void bind() override;
            void run() override;

        };

        class handler: handler_base<generic> {
        public:
            handler() = default;
            ~handler();

        protected:
            void init() override;
            void run() override;

        };

    };

    void generic::ev_handler::listener_callback(evconnlistener* listener, socket_t fd, sockaddr* address, int socklen, void* arg) {
        log_debug("New connection");
        generic::ev_handler *ev_handler = (generic::ev_handler *)(arg);
        bufferevent *bev = bufferevent_socket_new(ev_handler->_ev_base, fd, BEV_OPT_CLOSE_ON_FREE);
        bufferevent_enable(bev, EV_READ | EV_WRITE);
        bufferevent_setcb(bev,
                          read_callback, write_callback, event_callback,
                          ev_handler);
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

    generic::ev_handler::~ev_handler() {
        evconnlistener_free(_evc);
        event_base_free(_ev_base);
    }

    void generic::ev_handler::bind() {
        sockaddr_in sin = socket_address(_server->port);
        _evc = evconnlistener_new_bind(
                _ev_base, listener_callback, this,
                LEV_OPT_REUSEABLE | LEV_OPT_CLOSE_ON_FREE | LEV_OPT_REUSEABLE_PORT,
                -1, (sockaddr*) &sin, sizeof(sin));
        _server->try_ready();
        _server->wait_at_least(server<generic>::RUNNING);

    }

    void generic::ev_handler::run() {
        event_base_dispatch(_ev_base);
    }

    generic::handler::~handler() {
        if(compete) delete static_cast<b_fifo*>(fifo.load());
        else        delete static_cast<lf_fifo*>(fifo.load());
    }

    void generic::handler::init() {
        fifo.store(compete ? (void*) new b_fifo(TASK_FIFO_SIZE) : (void*) new lf_fifo(TASK_FIFO_SIZE),
                   memory_order_relaxed);
    }

    void generic::handler::run() {

    }

}

#endif