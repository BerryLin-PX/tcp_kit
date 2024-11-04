#ifndef TCP_KIT_EVENT_CONTEXT_H
#define TCP_KIT_EVENT_CONTEXT_H

#include <event2/bufferevent.h>
#include <util/tcp_util.h>

namespace tcp_kit {

    class event_context {

    public:
        socket_t     fd;
        sockaddr*    address;
        int          socklen;
        bufferevent* bev;

        event_context(socket_t fd, sockaddr* address, int socklen, bufferevent* bev): fd(fd), address(address),
                                                                                      socklen(socklen), bev(bev) { };

    };



}

#endif