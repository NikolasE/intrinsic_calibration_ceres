cmake_minimum_required(VERSION 3.0.0)
project(ceres_intrinsic_calib VERSION 0.1.0)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the C++17 flag to the CMAKE_CXX_FLAGS variable
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")


include(CTest)
enable_testing()

find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(Boost REQUIRED) # not really required, could be removed if required 

include_directories(${OpenCV_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})


include_directories(${CMAKE_SOURCE_DIR}/include)

add_library(ceres_intrinsic_calib src/ceres_intrinsic_calib.cpp)
target_link_libraries(ceres_intrinsic_calib ${OpenCV_LIBS} ${CERES_LIBRARIES} ${Boost_LIBRARIES})

add_executable(run_calib main.cpp)
target_link_libraries(run_calib ceres_intrinsic_calib)


add_executable(capture_images src/image_capture.cpp)
target_link_libraries(capture_images ${OpenCV_LIBS})


# set(CPACK_PROJECT_NAME ${PROJECT_NAME})
# set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
# include(CPack)

