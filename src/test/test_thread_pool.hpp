#include <thread/thread_pool.h>

namespace tcp_kit {

    void t1() {
        blocking_queue<runnable> q(10);
        thread_pool tp(1, 1, 0, &q);
        tp.execute([] {});
    }

}

