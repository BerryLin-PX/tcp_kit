#include <network/protocol/generic.h>

#define SUCCESSFUL 0 // libevent API 表示成功的值

namespace tcp_kit {

    void generic::ev_handler::listener_callback(evconnlistener* listener, socket_t fd, sockaddr* address, int socklen, void* arg) {
        generic::ev_handler* ev_handler = (generic::ev_handler *)(arg);
        bufferevent* bev = bufferevent_socket_new(ev_handler->_ev_base, fd, BEV_OPT_CLOSE_ON_FREE);
        if(!bev) {
            log_error("Failed to allocate the bufferevent");
            close_socket(fd);
            return;
        }
        event_context* ctx = new event_context{fd, address, socklen, bev};
        if(ev_handler->call_conn_filters(ctx) &&
           ev_handler->register_read_write_filters(ctx) &&
           bufferevent_enable(ctx->bev, EV_READ | EV_WRITE) == SUCCESSFUL) {
            bufferevent_setcb(ctx->bev,
                              read_callback, write_callback, event_callback,
                              ctx);
        } else {
            log_error("Failed during connection or filter setup");
            bufferevent_free(ctx->bev);
            delete ctx;
        }
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