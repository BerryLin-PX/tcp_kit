#tcp_kit
tcp_kit is a **lightweight, cross-platform** network protocol development kit based on the C++11 standard. It adopts the Reactor event-driven model and supports multithreading and asynchronous I/O operations. This kit is designed to provide developers with:

- 🚀 High-performance network processing: Event loop driven by libevent 
- 📦 Out-of-the-box protocol stack: Built-in common encoding/decoding and encryption/decryption protocols
- 🖥️ Cross-platform support: Compatible with Linux and macOS

## ⚡️ Quick Start

### 🌍 Environment
- Compiler：GCC 4.8.5+
- Dependencies: libevent 2.1+, Protobuf 3.0+
- Build System: CMake 2.8+

###🛠 Installation

####Clone the Repository
```shell
git clone https://github.com/LinRuiiXin/tcp_kit.git
cd tcp_kit
```
####Install Dependencies
   - CentOS
      ```shell
      sudo yum install -y libevent-devel protobuf-devel
      ```
   - macOS
      ```shell
      brew install libevent protobuf
      ```
####Build tcp_kit
```shell
mkdir build && cd build
cmake ..
sudo make install
```
### 📢 Echo Server
```c++
#include <tcp_kit/network/server.h>
#include <tcp_kit/network/generic.h>

int main() {
    tcp_kit::server<generic, 3000> svr;
    svr.api("echo", [](std::string msg) {
        return msg;
    });
    svr.start();
}
```

