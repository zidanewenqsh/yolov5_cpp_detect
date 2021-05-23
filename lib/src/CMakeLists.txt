cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 14) # important
project(detector)
# set(LIBUTILS_SRC utils.cpp)
# set(LIBDETECT_SRC detector.cpp utils.cpp)
# set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
# add_subdirectory(src)
if (UNIX)
    set(CMAKE_PREFIX_PATH ~/libtorchcpu)
    # set(LIBRARY_OUTPUT_PATH ~/lib)
    set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
    set(CMAKE_INSTALL_PREFIX install)
    # link_directories(~/lib)
elseif(WIN32)
    set(LIBRARY_OUTPUT_PATH E:/lib)
    set(Torch_DIR E:/lib/libtorch_debug_17/share/cmake/Torch)
    set(OpenCVPath D:/lib/opencv452/build/x64/vc15)
    set(OpenCV_DIR ${OpenCVPath}/lib)
    link_directories(E:/lib/Debug/)

endif (UNIX)
# set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
include_directories(../include)
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
# set(SRC_LIST main.cpp detector.cpp utils.cpp)
# add_executable(${PROJECT_NAME} ${SRC_LIST})
# target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES} ${OpenCV_LIBS})
# set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)

set(LIBUTILS_SRC utils.cpp)
set(LIBDETECT_SRC detector.cpp utils.cpp)
add_library(utils SHARED ${LIBUTILS_SRC})
add_library(detector SHARED ${LIBDETECT_SRC})
add_library(utils_static STATIC ${LIBUTILS_SRC})
add_library(detector_static STATIC ${LIBDETECT_SRC})
target_link_libraries(utils ${TORCH_LIBRARIES} ${OpenCV_LIBS})
target_link_libraries(utils_static ${TORCH_LIBRARIES} ${OpenCV_LIBS})
target_link_libraries(detector ${TORCH_LIBRARIES} ${OpenCV_LIBS})
target_link_libraries(detector_static ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_target_properties(utils_static PROPERTIES OUTPUT_NAME utils)
set_target_properties(detector_static PROPERTIES OUTPUT_NAME detector)
if (UNIX)
# install(TARGETS ${PROJECT_NAME} 
# 	DESTINATION bin 
# 	PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
#                     GROUP_READ GROUP_EXECUTE)

install(TARGETS detector utils detector_static utils_static
    #    EXPORT ${PROJECT_NAME}Targets # 执行完后，build文件夹中会生成一个名为PangolinTargets.cmake的文件，主要内容为Pangolin库所链接的其它库的绝对路径，以及pangolin库在build树中的位置。这样一来，外部工程直接导入build树中的库文件就变的十分方便。
       LIBRARY DESTINATION lib  # 动态库安装路径
       ARCHIVE DESTINATION lib  # 静态库安装路径
       RUNTIME DESTINATION bin  # 可执行文件安装路径
    #    PUBLIC_HEADER DESTINATION include  # 头文件安装路径
       )
install(DIRECTORY include
        DESTINATION include
)

endif(UNIX)

if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET ${PROJECT_NAME}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:${PROJECT_NAME}>)
    file(GLOB OPENCV_DLLS "${OpenCVPath}/bin/*.dll")
    add_custom_command(TARGET ${PROJECT_NAME}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${OPENCV_DLLS}
            $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif ()