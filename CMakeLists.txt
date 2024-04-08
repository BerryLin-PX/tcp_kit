cmake_minimum_required(VERSION 2.8)
project(tcp_server)

set(CMAKE_CXX_STANDARD 11)

find_library(LIBEVENT_LIB /usr/local/Cellar/libevent/2.1.12_1)

file(GLOB_RECURSE ALL_FILES "${CMAKE_SOURCE_DIR}/src/*.h" "${CMAKE_SOURCE_DIR}/src/*.cpp" "${CMAKE_SOURCE_DIR}/src/*.hpp")

add_executable(tcp_server ${ALL_FILES})

include_directories(src)
include_directories(/usr/local/Cellar/libevent/2.1.12_1/include)
target_link_libraries(tcp_server ${LIBEVENT_LIB})

