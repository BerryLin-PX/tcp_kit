#ifndef TCP_KIT_SSL_FILTER_H
#define TCP_KIT_SSL_FILTER_H

#include <network/filter/filter.hpp>

namespace tcp_kit {

    namespace filters {

        // 依赖于 libevent 提供的 OpenSSL 过滤器实现

        bool conn_filter(bufferevent*& bev) {

        }

        extern const filter ssl_filter = {conn_filter, nullptr, nullptr};

    }

}


#endif