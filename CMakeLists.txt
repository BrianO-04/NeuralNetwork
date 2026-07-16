cmake_minimum_required(VERSION 3.10.0)
project(NeuralNetwork VERSION 0.1.0 LANGUAGES C CXX)


include_directories(include)
include_directories(src)

add_executable(NeuralNetwork 
    src/main.cpp)

find_package(Eigen3 REQUIRED)
target_link_libraries(NeuralNetwork PRIVATE Eigen3::Eigen)