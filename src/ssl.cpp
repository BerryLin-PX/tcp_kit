#include "include/network/ssl.h"
#include <logger/logger.h>
#include <openssl/crypto.h>
#include <stdlib.h>
#include <error/errors.h>

#define _SSLtid pthread_self().p

namespace tcp_kit {

    namespace filters {

        pthread_mutex_t* ssl_ctx_guard::ssl_locks = nullptr;
        int ssl_ctx_guard::ssl_num_locks = 0;
        ssl_ctx_guard ssl_ctx_guard::singleton;

#define SSL_PKEY_FILE "/Users/linruixin/Desktop/pkey" // TODO
#define SSL_CERT_FILE "/Users/linruixin/Desktop/cert" // TODO

#if OPENSSL_VERSION_NUMBER < 0x10000000L
        static unsigned long get_thread_id_cb() {
            return _SSLtid;
        }
#else
        static void get_thread_id_cb(CRYPTO_THREADID *id) {
            CRYPTO_THREADID_set_numeric(id, _SSLtid);
        }
#endif

        ssl_ctx_guard::ssl_ctx_guard() {
            init_ssl_locking();
            SSL_load_error_strings();
            SSL_library_init();
            if(!RAND_poll()) {
                log_error("Failed to gather sufficient entropy for SSL initialization");
                exit(EXIT_FAILURE);
            }
            ctx = SSL_CTX_new(SSLv23_server_method());
            if (! SSL_CTX_use_certificate_chain_file(ctx, SSL_CERT_FILE) ||
                ! SSL_CTX_use_PrivateKey_file(ctx, SSL_PKEY_FILE, SSL_FILETYPE_PEM)) {
                log_error("Couldn't read 'pkey' or 'cert' file.  To generate a key\n"
                     "and self-signed certificate, run:\n"
                     "  openssl genrsa -out pkey 2048\n"
                     "  openssl req -new -key pkey -out cert.req\n"
                     "  openssl x509 -req -days 365 -in cert.req -signkey pkey -out cert");
                exit(EXIT_FAILURE);
            }
            SSL_CTX_set_options(ctx, SSL_OP_NO_SSLv2);
        }

        void ssl_ctx_guard::thread_lock_cb(int mode, int which, const char * f, int l) {
            if (which < ssl_num_locks) {
                if (mode & CRYPTO_LOCK) {
                    pthread_mutex_lock(&(ssl_locks[which]));
                } else {
                    pthread_mutex_unlock(&(ssl_locks[which]));
                }
            }
        }

        void ssl_ctx_guard::init_ssl_locking() {
            int i;
            ssl_num_locks = CRYPTO_num_locks();
            ssl_locks = static_cast<pthread_mutex_t *>(malloc(ssl_num_locks * sizeof(pthread_mutex_t)));
            if (ssl_locks == NULL) {
                log_error("Failed to initialize OpenSSL locking");
                exit(EXIT_FAILURE);
            }
            for (i = 0; i < ssl_num_locks; i++) {
                pthread_mutex_init(&(ssl_locks[i]), NULL);
            }
#if OPENSSL_VERSION_NUMBER < 0x10000000L
            CRYPTO_set_id_callback(get_thread_id_cb);
#else
            CRYPTO_THREADID_set_callback(get_thread_id_cb);
#endif
            CRYPTO_set_locking_callback(thread_lock_cb);
        }

        ssl_ctx_guard::~ssl_ctx_guard() {
            delete[] ssl_locks;
            SSL_CTX_free(ctx);
        }

        void openssl_filter::connect(ev_context* ctx) {
            bufferevent* ssl_bev = bufferevent_openssl_filter_new(bufferevent_get_base(ctx->bev),
                                                                  ctx->bev,
                                                                  SSL_new(ssl_ctx_guard::singleton.ctx),
                                                                  BUFFEREVENT_SSL_ACCEPTING,
                                                                  BEV_OPT_CLOSE_ON_FREE);
            if(ssl_bev) {
                ctx->bev = ssl_bev;
            } else {
                throw generic_error<CONS_BEV_FAILED>("Failed to create the SSL filter");
            }
        }


    }

}
