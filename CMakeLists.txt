cmake_minimum_required(VERSION 2.8)
project(tcp_kit)

set(CMAKE_CXX_STANDARD 11)

#set(LIBEVENT_ROOT /usr/local/opt/libevent)
#find_package(Libevent REQUIRED)

set(LIBEVENT_INCLUDE_DIR /usr/local/opt/libevent/include)
set(LIBEVENT_LIBRARIES /usr/local/opt/libevent/lib/libevent.a)

set(GTEST_LIB /usr/local/opt/googletest)
find_package(GTest REQUIRED)

file(GLOB_RECURSE ALL_FILES
        "${CMAKE_SOURCE_DIR}/src/*.h"
        "${CMAKE_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_SOURCE_DIR}/src/*.hpp")

add_executable(tcp_kit ${ALL_FILES})

include_directories(src)
include_directories(${GTEST_INCLUDE_DIRS})
include_directories(${LIBEVENT_INCLUDE_DIR})

target_link_libraries(tcp_kit ${GTEST_LIBRARIES})
target_link_libraries(tcp_kit ${LIBEVENT_LIBRARIES})
