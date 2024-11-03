#include <network/filter.h>

namespace tcp_kit {

    raw_buffer::raw_buffer(bufferevent *bev_): bev(bev_) {

    }

    filter::filter(connect_filter connect_, bufferevent_filter_cb read_, bufferevent_filter_cb write_)
                                                : connect(connect_),
                                                  read(read_),
                                                  write(write_) {
    }

    bool filter::operator==(const filter& filter_) const {
        return connect == filter_.connect &&
               read == filter_.read &&
               write == filter_.write &&
               process == filter_.process;
    }


}
