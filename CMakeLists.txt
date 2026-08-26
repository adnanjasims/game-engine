cmake_minimum_required(VERSION 3.20)

project(EngineOptimizationCore
  VERSION 0.1.0
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "build type" FORCE)
endif()

option(EOC_BUILD_TESTS "build googletest suites" ON)
option(EOC_BUILD_BENCH "build standalone benchmark executable" ON)

find_package(Threads REQUIRED)

set(EOC_PUBLIC_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/include")

set(EOC_CORE_SOURCES
  src/core/module_interface.cpp
  src/core/task_graph.cpp
  src/core/memory_allocators.cpp
  src/core/profiler.cpp
  src/ai/inference_engine.cpp
  src/ai/tensor_buffer.cpp
  src/multimedia/audio_pipeline.cpp
  src/multimedia/video_decoder_stub.cpp
  src/telemetry/metrics_collector.cpp
)

add_library(eoc_core SHARED ${EOC_CORE_SOURCES})
add_library(eoc::core ALIAS eoc_core)

set_target_properties(eoc_core PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  VERSION ${PROJECT_VERSION}
  SOVERSION 0
)

target_include_directories(eoc_core PUBLIC
  "$<BUILD_INTERFACE:${EOC_PUBLIC_INCLUDE}>"
  "$<INSTALL_INTERFACE:include>"
)

target_compile_definitions(eoc_core PRIVATE EOC_BUILDING_SHARED)

target_link_libraries(eoc_core PUBLIC Threads::Threads)

set(EOC_WARNINGS -Wall -Wextra -Werror -pedantic -O3)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(eoc_core PRIVATE ${EOC_WARNINGS})
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    target_compile_options(eoc_core PRIVATE -mavx2)
  endif()
endif()

if(EOC_BUILD_BENCH)
  add_executable(eoc_bench src/main.cpp)
  target_link_libraries(eoc_bench PRIVATE eoc_core)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(eoc_bench PRIVATE ${EOC_WARNINGS})
  endif()
endif()

if(EOC_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
    GIT_SHALLOW TRUE
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()
  include(GoogleTest)

  add_executable(eoc_tests
    tests/test_allocators.cpp
    tests/test_task_graph.cpp
    tests/test_inference_pipeline.cpp
  )
  target_link_libraries(eoc_tests PRIVATE eoc_core GTest::gtest_main)
  target_include_directories(eoc_tests SYSTEM PRIVATE
    "${googletest_SOURCE_DIR}/googletest/include"
  )
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(eoc_tests PRIVATE ${EOC_WARNINGS})
  endif()
  gtest_discover_tests(eoc_tests)
endif()
