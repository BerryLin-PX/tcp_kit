#include "network/event_context.h"
#include "util/tcp_util.h"
#include <event2/bufferevent.h>

namespace tcp_kit {

    event_context::event_context(socket_t fd, sockaddr* address, int socklen, bufferevent* bev)
            : fd(fd), address(address), socklen(socklen), bev(bev) { }

}