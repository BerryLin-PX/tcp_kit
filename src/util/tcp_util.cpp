#include <util/tcp_util.h>
#include <logger/logger.h>

namespace tcp_server {

#if defined(__APPLE__) || defined(__linux__)
#include <sys/socket.h>
#include <sys/types.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <stdlib.h>
#include <strings.h>

    #ifdef __APPLE__
    #include <fcntl.h>
    socket_t open_socket() {
        socket_t socket_fd = socket(AF_INET, SOCK_STREAM, 0);
        if(socket_fd < 0) {
            log_error("failed to open socket: %d", socket_fd);
            exit(1);
        }
        int flags = fcntl(socket_fd, F_GETFL, 0) | O_NONBLOCK;
        fcntl(socket_fd, F_SETFL, flags);
        return socket_fd;
    }
    #else
    socket_t open_socket() {
        socket_t socket_fd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
        if(socket_fd < 0) {
            log_error("failed to open socket: %d", socket_fd);
            exit(1);
        }
        return socket_fd;
    }
    #endif

    sa_in cons_sa_in(uint16_t port) {
        sa_in addr;
        bzero(&addr, sizeof(sa_in));
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
        addr.sin_port = htons(port);
        return addr;
    }

    int bind_socket(socket_t socket_fd, sa_in *address) {
        int ret = bind(socket_fd, (struct sockaddr*) address, sizeof(sa_in));
        if(ret < 0) log_error("socket bind failed: %d", ret);
        return ret;
    }

    int listen_socket(socket_t socket_fd) {
        return listen(socket_fd, SOMAXCONN);
    }

    int close_socket(socket_t socket_fd) {
        return close(socket_fd);
    }

#elif _WIN32

#endif

}
