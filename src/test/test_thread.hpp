#include <thread>
#include <logger/logger.h>

using namespace std;

void t1() {
    thread t([]{
        log_debug("THE NEW THREAD RUNNING");
    });
    t.join();
}