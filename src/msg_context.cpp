#include <network/msg_context.h>
#include <event2/event.h>

namespace tcp_kit {

    void msg_context::done() {
        event_active(done_ev, 0, 0);
    }

}
