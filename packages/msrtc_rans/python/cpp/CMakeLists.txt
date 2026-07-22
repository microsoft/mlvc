find_package (Python3 COMPONENTS Interpreter Development)

set(SRCS
    PyRANS.cpp
    PyWrapper.h
)

Python3_add_library(msrtc_rans_pyd MODULE WITH_SOABI ${SRCS})
set_target_properties(
    msrtc_rans_pyd
    PROPERTIES
    OUTPUT_NAME _msrtc_rans
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/$<CONFIG>/msrtc/rans
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/$<CONFIG>/msrtc/rans
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/$<CONFIG>/msrtc/rans
)
target_link_libraries(msrtc_rans_pyd PRIVATE msrtc_rans)

install(TARGETS msrtc_rans_pyd LIBRARY DESTINATION msrtc/rans)
