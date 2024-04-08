#include <logger/logger.h>
#include <util/tcp_util.h>
#include <event2/event.h>
#include <event2/listener.h>

using namespace tcp_server;

void t2_accept_callback(struct evconnlistener *listener, socket_t fd, struct sockaddr *address, int socklen, void *ctx);
void t1_accept_callback(socket_t sock_fd, short events, void *arg);

void test1() {
    int sock_fd = open_socket();
    auto address = cons_sa_in(8000);
    bind_socket(sock_fd, &address);
    auto *ev_base = event_base_new();
    auto *ev = event_new(ev_base, sock_fd, EV_READ, t1_accept_callback, nullptr);
    event_add(ev, NULL);
    listen_socket(sock_fd);
    event_base_dispatch(ev_base);
}

void test2() {
    auto address = cons_sa_in(8000);
    auto *ev_base = event_base_new();
    evconnlistener_new_bind(ev_base,t2_accept_callback,ev_base,
                            LEV_OPT_REUSEABLE | LEV_OPT_CLOSE_ON_FREE, 10,
                            (struct sockaddr *)&address,sizeof(sa_in));
    event_base_dispatch(ev_base);

}

void t1_accept_callback(int sock_fd, short events, void *arg) {
    log_info("ON ACCEPT CALLBACK");
}

void t2_accept_callback(
        struct evconnlistener *listener, socket_t fd,
        struct sockaddr *address, int socklen, void *ctx) {
    log_info("ON ACCEPT CALLBACK");
}
