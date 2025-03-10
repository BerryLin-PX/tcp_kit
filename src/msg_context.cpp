#include <network/msg_context.h>
#include <event2/event.h>
#include <assert.h>

namespace tcp_kit {

    void msg_context::done() {
        assert(!callback_fired);
        event_active(done_ev, 0, 0);
        event_fired = true;
    }

    void msg_context::error() {
        assert(!callback_fired);
        event_active(error_ev, 0, 0);
        event_fired = true;
    }

}
