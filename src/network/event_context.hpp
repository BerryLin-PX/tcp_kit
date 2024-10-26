#ifndef TCP_KIT_EVENT_CONTEXT_HPP
#define TCP_KIT_EVENT_CONTEXT_HPP

#include <network/server.hpp>
#include <vector>
#include <network/filter/filter.hpp>

namespace tcp_kit {

    class event_context {

    public:
        socket_t     fd;
        sockaddr*    address;
        int          socklen;
        bufferevent* bev;

        event_context(int fd, sockaddr *address, int socklen, bufferevent *bev);

    };

    event_context::event_context(int fd, sockaddr *address, int socklen, bufferevent *bev)
                    : fd(fd), address(address), socklen(socklen), bev(bev) { }

}

#endif