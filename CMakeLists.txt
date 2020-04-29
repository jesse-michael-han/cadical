cmake_minimum_required(VERSION 3.0)
project(CaDiCaL C CXX)

# set(CMAKE_PREFIX_PATH "$ENV{HOME}/org/projects/rl-lbd/cadical/pybind11/")

add_subdirectory(pybind11)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -O3 -DNDEBUG -I/usr/include/python3.8 -fPIC")
set(CMAKE_CXX_STANDARD 17)

set(VERSION "0.0")

include_directories(${CMAKE_CURRENT_BINARY_DIR})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})


file(GLOB cadical_source_files "src/*.hpp" "src/*.cpp" "*.hpp")
list(REMOVE_ITEM cadical_source_files "src/cadical.cpp")
list(REMOVE_ITEM cadical_source_files "src/mobical.cpp")
list(REMOVE_ITEM cadical_source_files "src/satenv.cpp")
list(REMOVE_ITEM cadical_source_files "src/satenv.hpp")

add_library(cadical_lib ${cadical_source_files})
target_link_libraries(cadical_lib "${ZLIB_LIBRARY}")
target_link_libraries(cadical_lib pybind11)

pybind11_add_module(satenv src/satenv.cpp)
target_link_libraries(satenv PRIVATE cadical_lib)
