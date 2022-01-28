cmake_minimum_required(VERSION 3.10)
project(literatelb LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(SFML 2.5 COMPONENTS graphics window system REQUIRED)
find_package(ImGui-SFML REQUIRED)
find_package(glm REQUIRED)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --default-stream per-thread --extended-lambda")
set(CMAKE_CUDA_ARCHITECTURES 86)

include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/tangle
	${CMAKE_CURRENT_BINARY_DIR})

include(assets.cmake)
serialize_assets(${CMAKE_CURRENT_SOURCE_DIR}/tangle/asset
  ${CMAKE_CURRENT_BINARY_DIR}/assets.cc
  ${CMAKE_CURRENT_BINARY_DIR}/assets.h)
add_library(assets ${CMAKE_CURRENT_BINARY_DIR}/assets.cc)

link_libraries(
  CUDA::cuda_driver
	sfml-graphics
	sfml-window
	sfml-system
	ImGui-SFML::ImGui-SFML
  assets)

file(GLOB EXAMPLES ${CMAKE_CURRENT_SOURCE_DIR}/tangle/*.cu)
foreach(examplefile ${EXAMPLES})
  get_filename_component(examplename ${examplefile} NAME_WE)
  add_executable(${examplename} ${examplefile})
endforeach()
