#include <util/system_util.h>
#include <logger/logger.h>

using namespace tcp_kit;

void t1() {
    log_info("n of processor: %lld", processor_numb());
}