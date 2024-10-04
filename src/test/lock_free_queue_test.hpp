#include <concurrent/lock_free_fifo.hpp>
#include <logger/logger.h>

namespace tcp_kit {

    namespace lock_free_queue_test {

        void t1() {
            lock_free_fifo<int, 10> queue;
            for(int i = 0; i <= 20; ++i) {
                if(queue.push_by_shallow_copy(&i) < sizeof(int)) {
                    log_error("Push failed");
                }
            }
            for(int i = 0; i <= 20; ++i) {
                int j = 0;
                if(queue.pop(&j) < sizeof(int)) {
                    log_error("Pop failed");
                } else {
                    log_info("Pop out: %d", j);
                }
            }
        }

    }

}