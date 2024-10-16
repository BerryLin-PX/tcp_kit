// non-blocking
#ifndef _TCP_SERVER_TCP_UTIL_
#define _TCP_SERVER_TCP_UTIL_

#include <stdint.h>
#include <event2/util.h>
#include <netinet/in.h>

#define socket_t evutil_socket_t

namespace tcp_kit {

    socket_t open_socket();

    sockaddr_in socket_address(uint16_t port);

    int bind_socket(socket_t socket_fd, sockaddr_in* addr);

    int accept_conn(socket_t socket_fd, sockaddr_in* addr, socklen_t* len);

    int listen_socket(socket_t socket_fd);

    int close_socket(socket_t socket_fd);

}

#endif
