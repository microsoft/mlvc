cmake_minimum_required(VERSION 3.23)
project(msrtc_rans VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

option(BUILD_PYTHON_MODULE "build python module" ON)
option(INSTALL_ONLY_PYTHON_MODULE "install only python module" OFF)

if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    add_compile_options(/W4 /WX)
else()
    add_compile_options(-Wall -Werror)
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(SRCS
    include/msrtc_rans/span.h
    include/msrtc_rans/EntropyCoder.h
    private/include/msrtc_rans/rans.h
    src/EntropyCoder.cpp
)

add_library(msrtc_rans STATIC ${SRCS})
target_include_directories(
    msrtc_rans
    PUBLIC include
    PRIVATE private/include
)

if (NOT INSTALL_ONLY_PYTHON_MODULE)
    install(TARGETS msrtc_rans)
endif()

if (BUILD_PYTHON_MODULE)
    add_subdirectory(python/cpp)
endif()
