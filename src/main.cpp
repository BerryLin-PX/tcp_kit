#include <test/test_logger.hpp>
#include <test/test_tcp_util.hpp>
#include <string.h>
#include <thread>

void function() {
    std::this_thread::sleep_for(std::chrono::seconds(3));
    t3_client();
}

int main() {
//    test_log();
//    t1();
//    t2();
    std::thread t(function);
    t3_server();
    return 0;
}
