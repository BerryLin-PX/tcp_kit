#include <network/builtin/string_msg.h>
#include <event2/buffer.h>

namespace tcp_kit {

    namespace filters {

        bufferevent_filter_result string_msg_read(evbuffer* src, evbuffer* dst,ev_ssize_t dst_limit,
                                                  bufferevent_flush_mode mode, void* ctx) {
            size_t length = 0;
            length = evbuffer_search(src, "\n", 1, nullptr).pos;
            if (length == -1) {
                if (mode == BEV_FLUSH) {
                    evbuffer_add_buffer(dst, src);
                    return BEV_OK;
                }
                return BEV_NEED_MORE;
            }
            // 从 src 中移出消息并直接写入 dst，包含 '\0'
            if (evbuffer_remove_buffer(src, dst, length + 1) < 0) {
                return BEV_ERROR;
            }
            return BEV_OK;
        }

        bufferevent_filter_result string_msg_write(evbuffer* src, evbuffer* dst,ev_ssize_t dst_limit,
                                                  bufferevent_flush_mode mode, void* ctx) {
            return BEV_OK;
        }

        const filter string_msg = filter::make(nullptr, string_msg_read, string_msg_write);

    }

}