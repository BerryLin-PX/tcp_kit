cmake_minimum_required(VERSION 2.8)
project(tcp_kit)

add_definitions(-DNDEBUG)
add_definitions(-levent -levent_pthreads)


set(CMAKE_CXX_STANDARD 11)

set(LIBEVENT_INCLUDE_DIR /usr/local/opt/libevent/include)
set(LIBEVENT_LIBRARIES /usr/local/opt/libevent/lib/libevent.a)
#                       /usr/local/opt/libevent/lib/libevent_pthreads.a)
set(LIBEVENT_OPEN_SSL_LIBRARIES /usr/local/opt/libevent/lib/libevent_openssl.a)

set(GTEST_LIB /usr/local/opt/googletest)
find_package(GTest REQUIRED)
find_package(OpenSSL REQUIRED)

set(Protobuf_INCLUDE_DIR "/usr/local/Cellar/protobuf@3/3.20.3/include")
set(Protobuf_LIBRARIES "/usr/local/Cellar/protobuf@3/3.20.3/lib/libprotobuf.a")
set(Protobuf_PROTOC_EXECUTABLE "/usr/local/Cellar/protobuf@3/3.20.3/bin/protoc")
find_package(Protobuf REQUIRED)

file(GLOB_RECURSE ALL_FILES
        "${CMAKE_SOURCE_DIR}/src/*.h"
        "${CMAKE_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_SOURCE_DIR}/src/*.hpp")

add_executable(tcp_kit ${ALL_FILES})

include_directories(src)
include_directories(src/include)
include_directories(${GTEST_INCLUDE_DIRS})
include_directories(${LIBEVENT_INCLUDE_DIR})
target_include_directories(tcp_kit PRIVATE ${Protobuf_INCLUDE_DIR})

target_link_libraries(tcp_kit ${GTEST_LIBRARIES})
target_link_libraries(tcp_kit ${LIBEVENT_LIBRARIES})
target_link_libraries(tcp_kit ${LIBEVENT_OPEN_SSL_LIBRARIES})
target_link_libraries(tcp_kit OpenSSL::SSL OpenSSL::Crypto)
target_link_libraries(tcp_kit ${Protobuf_LIBRARIES})

