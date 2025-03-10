#pragma once
#include <string>

namespace tcp_kit {

    template<typename Protocols>
    class client {
    public:
        explicit client(std::string& address);
        client(client&) = delete;
        client(client&&) = delete;
        client& operator=(client&) = delete;

    };

    template<typename Protocols>
    client<Protocols>::client(std::string& address) {

    }

}