if(MSVC)
  # Windows build uses newer features
  cmake_minimum_required(VERSION 3.21)
else()
  cmake_minimum_required(VERSION 3.3)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

project(spectacularAI_orbbec_example)

if(MSVC)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
else()
  set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
endif()

find_package(Threads REQUIRED)
find_package(spectacularAI_orbbecPlugin REQUIRED)
find_package(OrbbecSDK REQUIRED PATHS "${OrbbecSDK_DIR}")

if(MSVC) # Must be after project() is called
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /Gy")
else()
  set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
  SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--exclude-libs=ALL")
endif()

set(EXAMPLE_LIBS
  spectacularAI::orbbecPlugin
  OrbbecSDK::OrbbecSDK)

# enables searching for dynamic libraries from the relative path ../lib
if(NOT MSVC)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath='$ORIGIN/../lib:$ORIGIN/../lib/3rdparty'")
endif()

# Minimal example
add_executable(vio_jsonl vio_jsonl.cpp)
target_link_libraries(vio_jsonl ${EXAMPLE_LIBS})

if(MSVC)
  add_custom_command(TARGET vio_jsonl POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:vio_jsonl> $<TARGET_FILE_DIR:vio_jsonl>
    COMMAND_EXPAND_LISTS
  )
endif()

set(BUILD_MAPPER ON CACHE STRING "Build VIO mapper example (requires OpenCV)")
if (BUILD_MAPPER)
  find_package(Threads REQUIRED)
  set(USE_STATIC_OPENCV OFF CACHE STRING "Use OpenCV as statically linked library (internal and unsupported flag, do not use)")
  if (USE_STATIC_OPENCV)
    find_package(mobile-cv-suite REQUIRED)
    set(MAPPER_LIBS mobile-cv-suite::static mobile-cv-suite::imgcodecs Threads::Threads)
  else()
    find_package(OpenCV REQUIRED)
    set(MAPPER_LIBS "${OpenCV_LIBS}" Threads::Threads)
  endif()

  add_executable(vio_mapper vio_mapper.cpp)
  target_link_libraries(vio_mapper ${EXAMPLE_LIBS} ${MAPPER_LIBS})
  if(MSVC)
    add_custom_command(TARGET vio_mapper POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:vio_mapper> $<TARGET_FILE_DIR:vio_mapper>
      COMMAND_EXPAND_LISTS
    )
  endif()
endif()

set(BUILD_MAPPER ON CACHE STRING "Build global point cloud example (requires OpenCV)")
if (BUILD_MAPPER)
  find_package(Threads REQUIRED)
  set(USE_STATIC_OPENCV OFF CACHE STRING "Use OpenCV as statically linked library (internal and unsupported flag, do not use)")
  if (USE_STATIC_OPENCV)
    find_package(mobile-cv-suite REQUIRED)
    set(MAPPER_LIBS mobile-cv-suite::static mobile-cv-suite::imgcodecs Threads::Threads)
  else()
    find_package(OpenCV REQUIRED)
    set(MAPPER_LIBS "${OpenCV_LIBS}" Threads::Threads)
  endif()

  add_executable(global_pcd global_pcd.cpp)
  target_link_libraries(global_pcd ${EXAMPLE_LIBS} ${MAPPER_LIBS})
  if(MSVC)
    add_custom_command(TARGET global_pcd POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:global_pcd> $<TARGET_FILE_DIR:global_pcd>
      COMMAND_EXPAND_LISTS
    )
  endif()
endif()
