#ifndef TCP_KIT_SSL_H
#define TCP_KIT_SSL_H

#include "filter.h"
#include <event2/bufferevent_ssl.h>
#include <openssl/ssl.h>
#include <openssl/rand.h>
#include <pthread.h>

namespace tcp_kit {

    namespace filters {

        // 依赖于 libevent 提供的 OpenSSL 过滤器实现

        class ssl_ctx_guard {

        public:
            static ssl_ctx_guard singleton;
            SSL_CTX* ctx;

            ssl_ctx_guard(const ssl_ctx_guard&) = delete;
            ssl_ctx_guard(ssl_ctx_guard&&) = delete;
            ssl_ctx_guard& operator=(const ssl_ctx_guard&) = delete;

        private:
            static pthread_mutex_t* ssl_locks;
            static int ssl_num_locks;

            ssl_ctx_guard();
            static void thread_lock_cb(int mode, int which, const char * f, int l);
            static void init_ssl_locking();

            ~ssl_ctx_guard();

        };

        inline void ssl_connect(event_context* ctx);

        extern const filter ssl;

    }

}


#endif