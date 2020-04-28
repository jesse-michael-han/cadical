cmake_minimum_required(VERSION 3.0)
project(CaDiCaL C CXX)

set(CMAKE_PREFIX_PATH "$ENV{HOME}/Downloads/libtorch")

find_package(Torch REQUIRED)
find_package(ATen REQUIRED)
find_package(OpenMP REQUIRED)

# set(CMAKE_CXX_COMPILER $ENV{HOME}/.local/bin/g++)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -O3 -DNDEBUG -I/usr/include/python3.8")
set(CMAKE_CXX_STANDARD 17)

set(VERSION "0.0")

include_directories(${CMAKE_CURRENT_BINARY_DIR})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# add_custom_command ( OUTPUT build.hpp
#                      COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/make-build-header.sh
#                      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/)

# add_custom_target(
#   build_hpp ALL
#   # DEPENDS doc.pdf
# )

file(GLOB cadical_source_files "src/*.hpp" "src/*.cpp" "*.hpp")
list(REMOVE_ITEM cadical_source_files "src/cadical.cpp")
list(REMOVE_ITEM cadical_source_files "src/mobical.cpp")
# list(REMOVE_ITEM cadical_source_files "src/logging.hpp" "src/logging.cpp")

# add_library(cadical_lib2 "src/logging.hpp" "src/logging.cpp")

# add_library (clindices "src/clindices.hpp")
# set_target_properties(clindices PROPERTIES LINKER_LANGUAGE CXX)
# target_link_libraries(clindices ${TORCH_LIBRARIES})
# target_link_libraries(clindices -Wl,--no-as-needed "$ENV{HOME}/org/projects/neuro-minisat/torchscript/build/libmy_ops.so")
# target_link_libraries(cadical_lib clindices)

add_library(cadical_lib ${cadical_source_files})
target_link_libraries(cadical_lib "${TORCH_LIBRARIES}")
target_link_libraries(cadical_lib "${ZLIB_LIBRARY}")
target_link_libraries(cadical_lib -Wl,--no-as-needed "$ENV{HOME}/org/projects/neuro-minisat/torchscript/build/libmy_ops.so")
target_link_libraries(cadical_lib OpenMP::OpenMP_CXX)

# add_executable(cadical src/cadical.cpp)
# target_link_libraries(cadical cadical_lib)

add_executable(satenv src/satenv.cpp)
target_link_libraries(satenv cadical_lib)
