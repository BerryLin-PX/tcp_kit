#include <network/ev_context.h>
#include <event2/bufferevent.h>

namespace tcp_kit {


    ev_context::ev_context(const ev_context::control &ctl_, int fd_, sockaddr* address_, int socklen_,
                           ev_handler_base* ev_handler_, handler_base* handler_, bufferevent* bev_) : ctl(ctl_), fd(fd_),
                                                                                                      address(address_),
                                                                                                      socklen(socklen_),
                                                                                                      ev_handler(ev_handler_),
                                                                                                      handler(handler_),
                                                                                                      bev(bev_) {}
}