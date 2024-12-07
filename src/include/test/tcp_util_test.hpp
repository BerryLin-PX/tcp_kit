#ifndef TCP_KIT_TCP_UTIL_TEST_H
#define TCP_KIT_TCP_UTIL_TEST_H

#include <logger/logger.h>
#include <util/tcp_util.h>
#include <event2/event.h>
#include <event2/listener.h>
#include <event2/bufferevent.h>
#include <event2/buffer.h>
#include <stdio.h>
#include <arpa/inet.h>

namespace tcp_kit {

    namespace tcp_util_test {
        void t1_accept_callback(socket_t sock_fd, short events, void* arg);

        void t1() {
            int sock_fd = open_socket();
            auto address = socket_address(8000);
            bind_socket(sock_fd, &address);
            auto* ev_base = event_base_new();
            auto* ev = event_new(ev_base, sock_fd, EV_READ, t1_accept_callback, nullptr);
            event_add(ev, nullptr);
            listen_socket(sock_fd);
            event_base_dispatch(ev_base);
        }

        void t1_accept_callback(int sock_fd, short events, void* arg) {
            log_info("ON ACCEPT CALLBACK");
        }


        void t2_accept_callback(struct evconnlistener* listener, socket_t fd, struct sockaddr* address, int socklen, void* arg);
        void t2_read_callback(bufferevent* bev, void* arg);
        void t2_write_callback(bufferevent* bev, void* arg);
        void t2_event_callback(bufferevent* bev, short what, void* arg);

        void t2() {
            auto address = socket_address(8000);
            auto* ev_base = event_base_new();
            auto* ev_listener = evconnlistener_new_bind(ev_base,t2_accept_callback,ev_base,
                                                        LEV_OPT_REUSEABLE | LEV_OPT_CLOSE_ON_FREE, 10,
                                                        (struct sockaddr*) &address,sizeof(sockaddr_in));
            event_base_dispatch(ev_base);
            evconnlistener_free(ev_listener);
            event_base_free(ev_base);
        }

        void t2_accept_callback(
                struct evconnlistener* listener, socket_t fd,
                struct sockaddr* address, int socklen, void* arg) {
            log_info("ON ACCEPT CALLBACK");
            char ip[16] = {0};
            sockaddr_in* addr = (sockaddr_in*) address;
            evutil_inet_ntop(AF_INET, &addr->sin_addr, ip, sizeof(ip));
            log_info("client ip: %s", ip);
            event_base* ev_base = (event_base*)(arg);
            bufferevent* bev = bufferevent_socket_new(ev_base, fd, BEV_OPT_CLOSE_ON_FREE);
            bufferevent_enable(bev, EV_READ | EV_WRITE);
            timeval t1 = {10, 0};
            bufferevent_set_timeouts(bev, &t1, 0);
            bufferevent_setcb(bev,
                              t2_read_callback, t2_write_callback, t2_event_callback,
                              ev_base);
        }

        void t2_read_callback(bufferevent* bev, void* arg) {
            log_info("READ CALLBACK");
            char buf[1024] = {0};
            size_t len = bufferevent_read(bev, buf, sizeof(buf) - 1);
            log_info("read length: %d", len);
            bufferevent_write(bev, "OK", 3);
        }

        void t2_write_callback(bufferevent* bev, void* arg) {
            log_info("WRITE CALLBACK");
        }

        void t2_event_callback(bufferevent* bev, short what, void* arg) {

        }


        void t3_accept_callback(struct evconnlistener* listener, socket_t fd, struct sockaddr* address, int socklen, void* arg);
        void t3_read_callback(bufferevent* bev, void* arg);
        void t3_write_callback(bufferevent* bev, void* arg);
        void t3_event_callback(bufferevent* bev, short what, void* arg);

        uint32_t t3_read_ptr = 0;

        void t3_server() {
            auto address = socket_address(8000);
            auto *ev_base = event_base_new();
            auto *ev_listener = evconnlistener_new_bind(ev_base,t3_accept_callback,ev_base,
                                                        LEV_OPT_REUSEABLE | LEV_OPT_CLOSE_ON_FREE, 10,
                                                        (struct sockaddr *)&address,sizeof(sockaddr_in));
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

        static enum bufferevent_filter_result
        uppercase_filter(struct evbuffer *src, struct evbuffer *dst, ev_ssize_t lim, bufferevent_flush_mode mode, void *ctx) {
            char buf[1024];
            int n;
            while ((n = evbuffer_remove(src, buf, sizeof(buf))) > 0) {
                for (int i = 0; i < n; i++) {
                    buf[i] = toupper(buf[i]);
                }
                evbuffer_add(dst, buf, n);
            }
            return BEV_OK;
        }

        static enum bufferevent_filter_result
        space_to_underscore_filter(struct evbuffer *src, struct evbuffer *dst, ev_ssize_t lim, bufferevent_flush_mode mode, void *ctx) {
            char buf[1024];
            int n;
            while ((n = evbuffer_remove(src, buf, sizeof(buf))) > 0) {
                for (int i = 0; i < n; i++) {
                    if (buf[i] == ' ') {
                        buf[i] = '_';
                    }
                }
                evbuffer_add(dst, buf, n);
            }
            return BEV_OK;
        }

        void t4() {
            struct event_base *base = event_base_new();
            struct bufferevent *bev_raw = bufferevent_socket_new(base, -1, BEV_OPT_CLOSE_ON_FREE);

            // 添加第一个过滤器：将输入转换为大写
            struct bufferevent *bev_uppercase = bufferevent_filter_new(bev_raw, uppercase_filter, NULL,
                                                                       BEV_OPT_CLOSE_ON_FREE, NULL, NULL);

            // 添加第二个过滤器：将空格替换为下划线
            struct bufferevent *bev_final = bufferevent_filter_new(bev_uppercase, space_to_underscore_filter, NULL,
                                                                   BEV_OPT_CLOSE_ON_FREE, NULL, NULL);

            // 这里 bev_final 是我们实际操作的数据流，它已经经过两个过滤器的处理

            // 模拟向过滤链中写入数据
            const char *input_data = "hello world";
            bufferevent_write(bev_final, input_data, strlen(input_data));

            // 读取过滤后的数据
            struct evbuffer *output = bufferevent_get_input(bev_final);
            char output_data[1024];
            int len = evbuffer_remove(output, output_data, sizeof(output_data) - 1);
            output_data[len] = '\0';  // 添加字符串结束符

            printf("Filtered output: %s\n", output_data);  // 输出应为：HELLO_WORLD

            // 释放资源
            bufferevent_free(bev_final);
            event_base_free(base);
        }

        static enum bufferevent_filter_result
        first_filter(struct evbuffer *src, struct evbuffer *dst, ev_ssize_t lim, bufferevent_flush_mode mode, void *ctx) {
            log_info("Last filter");
            char buf[1024];
            int n;
            while ((n = evbuffer_remove(src, buf, sizeof(buf))) > 0) {
                for (int i = 0; i < n; i++) {
                    buf[i] = toupper(buf[i]);
                }
                evbuffer_add(dst, buf, n);
            }
            return BEV_OK;
        }

        static enum bufferevent_filter_result
        second_filter(struct evbuffer *src, struct evbuffer *dst, ev_ssize_t lim, bufferevent_flush_mode mode, void *ctx) {
            log_info("Second filter");
            char buf[1024];
            int n;
            while ((n = evbuffer_remove(src, buf, sizeof(buf))) > 0) {
                for (int i = 0; i < n; i++) {
                    if (buf[i] == ' ') {
                        buf[i] = '_';
                    }
                }
                evbuffer_add(dst, buf, n);
            }
            return BEV_OK;
        }

        void t5_accept_callback(struct evconnlistener *listener, socket_t fd, struct sockaddr *address, int socklen, void *arg) {
            struct event_base *ev_base = (struct event_base *)arg;
            struct bufferevent *bev = bufferevent_socket_new(ev_base, fd, BEV_OPT_CLOSE_ON_FREE);
            struct bufferevent *filtered_bev_1 = bufferevent_filter_new(
                    bev, first_filter, nullptr, BEV_OPT_CLOSE_ON_FREE, nullptr, nullptr);
            struct bufferevent *filtered_bev_2 = bufferevent_filter_new(
                    filtered_bev_1, second_filter, nullptr, BEV_OPT_CLOSE_ON_FREE, nullptr, nullptr);
            bufferevent_setcb(filtered_bev_2, nullptr, nullptr, nullptr, nullptr);
            bufferevent_enable(filtered_bev_2, EV_READ | EV_WRITE);
        }

        void t5() {
            auto address = socket_address(3000);
            auto *ev_base = event_base_new();
            auto *ev_listener = evconnlistener_new_bind(ev_base,t5_accept_callback,ev_base,
                                                        LEV_OPT_REUSEABLE | LEV_OPT_CLOSE_ON_FREE, 10,
                                                        (struct sockaddr *)&address,sizeof(sockaddr_in));
            event_base_dispatch(ev_base);
            evconnlistener_free(ev_listener);
            event_base_free(ev_base);
        }

    }

}

#endif
