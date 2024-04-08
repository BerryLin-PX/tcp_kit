#include <logger/logger.h>

void test_log() {
    log_debug("Hello %s", "world");
    log_info("Hello %s", "world");
    log_warn("Hello %s", "world");
    log_error("Hello %s", "world");
    log_fatal("Hello %s", "world");
}