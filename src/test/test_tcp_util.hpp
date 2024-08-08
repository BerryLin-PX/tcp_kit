#include <logger/logger.h>
#include <util/tcp_util.h>
#include <event2/event.h>
#include <event2/listener.h>
#include <event2/bufferevent.h>
#include <event2/buffer.h>
#include <stdio.h>
#include <arpa/inet.h>

using namespace tcp_kit;

void t1_accept_callback(socket_t sock_fd, short events, void *arg);

void t1() {
    int sock_fd = open_socket();
    auto address = cons_sa_in(8000);
    bind_socket(sock_fd, &address);
    auto *ev_base = event_base_new();
    auto *ev = event_new(ev_base, sock_fd, EV_READ, t1_accept_callback, nullptr);
    event_add(ev, nullptr);
    listen_socket(sock_fd);
    event_base_dispatch(ev_base);
}

void t1_accept_callback(int sock_fd, short events, void *arg) {
    log_info("ON ACCEPT CALLBACK");
}


void t2_accept_callback(struct evconnlistener *listener, socket_t fd, struct sockaddr *address, int socklen, void *arg);
void t2_read_callback(bufferevent *bev, void *arg);
void t2_write_callback(bufferevent *bev, void *arg);
void t2_event_callback(bufferevent *bev, short what, void *arg);

void t2() {
    auto address = cons_sa_in(8000);
    auto *ev_base = event_base_new();
    auto *ev_listener = evconnlistener_new_bind(ev_base,t2_accept_callback,ev_base,
                            LEV_OPT_REUSEABLE | LEV_OPT_CLOSE_ON_FREE, 10,
                            (struct sockaddr *)&address,sizeof(sa_in));
    event_base_dispatch(ev_base);
    evconnlistener_free(ev_listener);
    event_base_free(ev_base);
}

void t2_accept_callback(
        struct evconnlistener *listener, socket_t fd,
        struct sockaddr *address, int socklen, void *arg) {
    log_info("ON ACCEPT CALLBACK");
    char ip[16] = {0};
    sockaddr_in *addr = (sockaddr_in *)address;
    evutil_inet_ntop(AF_INET, &addr->sin_addr, ip, sizeof(ip));
    log_info("client ip: %s", ip);
    event_base *ev_base = (event_base *)(arg);
    bufferevent *bev = bufferevent_socket_new(ev_base, fd, BEV_OPT_CLOSE_ON_FREE);
    bufferevent_enable(bev, EV_READ | EV_WRITE);
    timeval t1 = {10, 0};
    bufferevent_set_timeouts(bev, &t1, 0);
    bufferevent_setcb(bev,
                      t2_read_callback, t2_write_callback, t2_event_callback,
                      ev_base);
}

void t2_read_callback(bufferevent *bev, void *arg) {
    log_info("READ CALLBACK");
    char buf[1024] = {0};
    size_t len = bufferevent_read(bev, buf, sizeof(buf) - 1);
    log_info("read length: %d", len);
    bufferevent_write(bev, "OK", 3);
}

void t2_write_callback(bufferevent *bev, void *arg) {
    log_info("WRITE CALLBACK");
}

void t2_event_callback(bufferevent *bev, short what, void *arg) {

}


void t3_accept_callback(struct evconnlistener *listener, socket_t fd, struct sockaddr *address, int socklen, void *arg);
void t3_read_callback(bufferevent *bev, void *arg);
void t3_write_callback(bufferevent *bev, void *arg);
void t3_event_callback(bufferevent *bev, short what, void *arg);

uint32_t t3_read_ptr = 0;

void t3_server() {
    auto address = cons_sa_in(8000);
    auto *ev_base = event_base_new();
    auto *ev_listener = evconnlistener_new_bind(ev_base,t3_accept_callback,ev_base,
                                                LEV_OPT_REUSEABLE | LEV_OPT_CLOSE_ON_FREE, 10,
                                                (struct sockaddr *)&address,sizeof(sa_in));
    event_base_dispatch(ev_base);
    evconnlistener_free(ev_listener);
    event_base_free(ev_base);
}

void t3_client() {
    struct event_base *ev_base = event_base_new();
    if (!ev_base) return;
    struct bufferevent *bev = bufferevent_socket_new(ev_base, -1, BEV_OPT_CLOSE_ON_FREE);
    if (!bev) {
        event_base_free(ev_base);
        return;
    }
    bufferevent_setcb(bev, NULL, NULL, NULL, NULL);
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(8000);
    inet_pton(AF_INET, "127.0.0.1", &server_addr.sin_addr);
    if (bufferevent_socket_connect(bev, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        bufferevent_free(bev);
        event_base_free(ev_base);
        return;
    }
    const char *msg = "A MESSAGE FROM CLIENT";
    bufferevent_write(bev, msg, strlen(msg));
    char eof = EOF;
    bufferevent_write(bev, &eof, 1);
    event_base_dispatch(ev_base);
    bufferevent_free(bev);
    event_base_free(ev_base);
}

void t3_accept_callback(struct evconnlistener *listener, socket_t fd, struct sockaddr *address, int socklen, void *arg) {
    log_info("ON ACCEPT CALLBACK");
    char ip[16] = {0};
    sockaddr_in *addr = (sockaddr_in *)address;
    evutil_inet_ntop(AF_INET, &addr->sin_addr, ip, sizeof(ip));
    log_info("client ip: %s", ip);
    event_base *ev_base = (event_base *)(arg);
    bufferevent *bev = bufferevent_socket_new(ev_base, fd, BEV_OPT_CLOSE_ON_FREE);
    bufferevent_enable(bev, EV_READ | EV_WRITE);
    timeval t1 = {10, 0};
    bufferevent_set_timeouts(bev, &t1, 0);
    bufferevent_setcb(bev,
                      t3_read_callback, t3_write_callback, t3_event_callback,
                      ev_base);
}

void t3_read_callback(bufferevent *bev, void *arg) {
    auto *ev_buffer = bufferevent_get_input(bev);
    size_t len = evbuffer_get_length(ev_buffer);
    const char *needle = "\r\n\r\n";
    evbuffer_ptr start;
    evbuffer_ptr_set(ev_buffer, &start, t3_read_ptr, EVBUFFER_PTR_SET);
    auto res = evbuffer_search(ev_buffer, needle, 4, &start);
    if(res.pos == -1) t3_read_ptr += len;
    else {
        char *msg = new char[res.pos + 1];
        evbuffer_remove(ev_buffer, msg, res.pos + 1);
        msg[res.pos] = '\0';
        log_info("GET MESSAGE: %s", msg);
        delete[] msg;
    }
}

void t3_write_callback(bufferevent *bev, void *arg) {

}

void t3_event_callback(bufferevent *bev, short what, void *arg) {

}
