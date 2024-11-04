#include <network/filter.h>

namespace tcp_kit {

    bool filter::operator==(const filter& filter_) const {
        return connect == filter_.connect &&
               read == filter_.read &&
               write == filter_.write &&
               process == filter_.process;
    }

}
