#include <util/system_util.h>

#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>

int64_t tcp_kit::numb_of_processor() {
    uint32_t numb;
    size_t len = 32;
    sysctlbyname("hw.physicalcpu", &numb, &len, nullptr, 0);
    return numb;
}

#elif defined(__linux__)
#include <unistd.h>

int64_t processor_numb() {
    return sysconf(_SC_NPROCESSORS_ONLN);
}

#elif defined(_WIN32)
int64_t tcp_kit::processor_numb() {
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    return sysinfo.dwNumberOfProcessors;
}

#endif
