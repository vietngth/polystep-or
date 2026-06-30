cmake_minimum_required(VERSION 3.13.0 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(EVALUATOR LANGUAGES CXX)
#set(CMAKE_PREFIX_PATH /home/cheahm/.julia/artifacts/fb412eee87eae845b84a799f0cabf241142406d7)

find_package(JlCxx REQUIRED)
get_target_property(JlCxx_location JlCxx::cxxwrap_julia LOCATION)
get_filename_component(JlCxx_location ${JlCxx_location} DIRECTORY)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${JlCxx_location}")
message(STATUS "Found JlCxx at ${JlCxx_location}")

include_directories(${RoutingKit_INCLUDE_DIR})

add_library(CostEvaluator SHARED
    Evmain.cpp
    Block.cpp
    District.cpp
    Params.cpp
    Point.cpp
    Polygon.cpp
)

target_link_libraries(CostEvaluator PUBLIC 
    JlCxx::cxxwrap_julia
)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(CostEvaluator PUBLIC OpenMP::OpenMP_CXX)
endif()
