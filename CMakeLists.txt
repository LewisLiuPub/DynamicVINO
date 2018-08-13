# Copyright (c) 2018 Intel Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required (VERSION 2.8)

project(dynamic_vino_lib)

list (APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# environment variable InferenceEngine_DIR can be use instead of relaive path to specify location of configuration file
set(InferenceEngine_DIR /opt/intel/computer_vision_sdk_2018.2.299/deployment_tools/inference_engine/share)
set(OpenCV_DIR /opt/intel/computer_vision_sdk_2018.2.299/opencv/share/OpenCV)


message(STATUS "Looking for inference engine configuration file at: ${CMAKE_PREFIX_PATH}")
find_package(InferenceEngine 1.1)

if (NOT InferenceEngine_FOUND)
    message(FATAL_ERROR "")
endif()

set(CMAKE_BUILD_TYPE "Release")
if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    message(STATUS "CMAKE_BUILD_TYPE not defined, 'Release' will be used")
    set(CMAKE_BUILD_TYPE "Release")
endif()

if (NOT(BIN_FOLDER))
    if("${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
        set (ARCH intel64)
    else()
        set (ARCH ia32)
    endif()

    set (BIN_FOLDER ${ARCH})
endif()

if (NOT (IE_MAIN_SOURCE_DIR))
    set(NEED_EXTENSIONS TRUE)
    if (WIN32)
        set (IE_MAIN_SOURCE_DIR ${CMAKE_SOURCE_DIR}/../bin/)
    else()
        set (IE_MAIN_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR})
    endif()
endif()

if(NOT(UNIX))
    set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${IE_MAIN_SOURCE_DIR}/${BIN_FOLDER})
    set (CMAKE_LIBRARY_PATH ${IE_MAIN_SOURCE_DIR}/${BIN_FOLDER})
    set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${IE_MAIN_SOURCE_DIR}/${BIN_FOLDER})
    set (CMAKE_COMPILE_PDB_OUTPUT_DIRECTORY ${IE_MAIN_SOURCE_DIR}/${BIN_FOLDER})
    set (CMAKE_PDB_OUTPUT_DIRECTORY ${IE_MAIN_SOURCE_DIR}/${BIN_FOLDER})
    set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${IE_MAIN_SOURCE_DIR}/${BIN_FOLDER})
    set (LIBRARY_OUTPUT_DIRECTORY ${IE_MAIN_SOURCE_DIR}/${BIN_FOLDER})
    set (LIBRARY_OUTPUT_PATH ${LIBRARY_OUTPUT_DIRECTORY}) # compatibility issue: linux uses LIBRARY_OUTPUT_PATH, windows uses LIBRARY_OUTPUT_DIRECTORY
else ()
    set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${IE_MAIN_SOURCE_DIR}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE}/lib)
    set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${IE_MAIN_SOURCE_DIR}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE}/lib)
    set (CMAKE_COMPILE_PDB_OUTPUT_DIRECTORY ${IE_MAIN_SOURCE_DIR}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE})
    set (CMAKE_PDB_OUTPUT_DIRECTORY ${IE_MAIN_SOURCE_DIR}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE})
    set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${IE_MAIN_SOURCE_DIR}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE})
    set (LIBRARY_OUTPUT_DIRECTORY ${IE_MAIN_SOURCE_DIR}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE}/lib)
    set (LIBRARY_OUTPUT_PATH ${LIBRARY_OUTPUT_DIRECTORY}/lib)
endif()

set(CMAKE_CXX_FLAGS "-std=c++14 ${CMAKE_CXX_FLAGS}")
if (WIN32)
    if(NOT "${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
        message(FATAL_ERROR "Only 64-bit supported on Windows")
    endif()

    set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS _CRT_SECURE_NO_WARNINGS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_SCL_SECURE_NO_WARNINGS -DNOMINMAX")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc") #no asynchronous structured exception handling
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LARGEADDRESSAWARE")
    if (ENABLE_OMP)
        find_package(OpenMP)
        if (OPENMP_FOUND)
            set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
            set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        endif()
    endif()
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Werror=return-type ")
    if (APPLE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=unused-command-line-argument")
    elseif(UNIX)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wuninitialized -Winit-self -Wmaybe-uninitialized")
    endif()
endif()

include(feature_defs OPTIONAL)

# Find OpenCV libray if exists
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})
if(OpenCV_FOUND)
    add_definitions(-DUSE_OPENCV)
endif()

####################################
## to use C++14
set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-std=c++14 ${CMAKE_CXX_FLAGS}")
####################################

set(GFLAGS_IS_SUBPROJECT TRUE)
add_subdirectory(thirdparty/gflags)
# add_subdirectory(thirdparty/common/format_reader)

# Properties->C/C++->General->Additional Include Directories
include_directories (
    # ${CMAKE_CURRENT_SOURCE_DIR}/common/format_reader
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${InferenceEngine_INCLUDE_DIRS}
    ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/gflags/include
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty
)

if (UNIX)
    SET(LIB_DL dl)
endif()

set(DEPENDENCIES realsense2 ${OpenCV_LIBS} cpu_extension)

if (NEED_EXTENSIONS)
    add_subdirectory(thirdparty/extension)
endif()

set(HEADER_FILES
        include/openvino_service/data_struct.h
        include/openvino_service/factory.h
        include/openvino_service/pipeline.h
        include/openvino_service/engines/engine.h
        include/openvino_service/inferences/base_inference.h
        include/openvino_service/inferences/age_gender_recognition.h
        include/openvino_service/inferences/emotions_recognition.h
        include/openvino_service/inferences/face_detection.h
        include/openvino_service/inferences/head_pose_recognition.h
        include/openvino_service/inputs/base_input.h
        include/openvino_service/inputs/realsense_camera.h
        include/openvino_service/inputs/standard_camera.h
        include/openvino_service/inputs/video_input.h
        include/openvino_service/models/base_model.h
        include/openvino_service/models/age_gender_detection_model.h
        include/openvino_service/models/face_detection_model.h
        include/openvino_service/models/head_pose_detection_model.h
        include/openvino_service/models/emotion_detection_model.h
        include/openvino_service/outputs/base_output.h
        include/openvino_service/outputs/image_window_output.h
        )


add_library(${PROJECT_NAME} SHARED
        lib/factory.cpp
        lib/pipeline.cpp
        lib/engines/engine.cpp
        lib/inferences/base_inference.cpp
        lib/inferences/age_gender_recognition.cpp
        lib/inferences/emotions_recognition.cpp
        lib/inferences/face_detection.cpp
        lib/inferences/head_pose_recognition.cpp
        lib/inputs/realsense_camera.cpp
        lib/inputs/standard_camera.cpp
        lib/inputs/video_input.cpp
        lib/models/base_model.cpp
        lib/models/head_pose_detection_model.cpp
        lib/models/emotion_detection_model.cpp
        lib/models/age_gender_detection_model.cpp
        lib/models/face_detection_model.cpp
        include/openvino_service/models/head_pose_detection_model
        lib/outputs/image_window_output.cpp
        )
set_target_properties(${PROJECT_NAME} PROPERTIES
        PUBLIC_HEADER
        "${HEADER_FILES}"
        )
target_link_libraries(${PROJECT_NAME} ${DEPENDENCIES})
add_subdirectory(sample)
# include(GNUInstallDirs)
# install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#         PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
#         )
# install(FILES ${CMAKE_BINARY_DIR}/${PROJECT_NAME}.pc
#         DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig)