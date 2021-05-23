# CMakeList.txt: CMakeProject1 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 14) # important
project(yolo)

if (UNIX)
        # set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
        set(CMAKE_PREFIX_PATH /home/wen/libtorchcpu)
        # link_directories(/home/wen/Projects/yolov5cpp_05/lib/build/install/lib)
        # link_directories(/usr/local/lib)
        # include_directories(/home/wen/Projects/yolov5cpp_05/lib/build/install/include)
        # include_directories(include)
        include_directories(/usr/local/include/yolov5)
        MESSAGE(STATUS This is TorchDIR, ${Torch_DIR})
        MESSAGE(STATUS This is CMAKE_PREFIX_PATH, ${CMAKE_PREFIX_PATH})
elseif(WIN32)
#        set(CUDA_PREFIX C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2)
        set(CUDA_TOOLKIT_ROOT_DIR C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0)
        set(Torch_DIR D:/Libs/libtorch_171_110_debug/share/cmake/Torch)
#        set(Torch_DIR D:/Libs/libtorch_102_18_debug/share/cmake/Torch)
        MESSAGE(STATUS This is TorchDIR, ${Torch_DIR})
        set(OpenCVPath D:/Libs/opencv/build/x64/vc15)
        set(OpenCV_DIR ${OpenCVPath}/lib)
        link_directories(E:/lib/yolo)
        include_directories(lib/include)
endif (UNIX)

# add_subdirectory(lib)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
# include_directories(include)
# include_directories(${CUDA_INCLUDE_DIRS})
# link_directories(${CUDA_LIBRARIES})

# find_package(CUDA REQUIRED)
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
AUX_SOURCE_DIRECTORY(src SRC_LIST)
# MESSAGE(STATUS This is SRC_LIST, ${SRC_LIST})
# set(SRC_LIST main.cpp detector.cpp utils.cpp)
# add_executable(${PROJECT_NAME} main.cpp ${SRC_LIST})
# target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES} ${OpenCV_LIBS})

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES} ${OpenCV_LIBS} detector)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)

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
