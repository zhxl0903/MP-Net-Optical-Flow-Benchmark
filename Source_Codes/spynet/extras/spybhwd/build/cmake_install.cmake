# Install script for directory: /home/zhang205/Github/MP-Net-Optical-Flow-Benchmark/Source_Codes/spynet/extras/spybhwd

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/zhang205/torch/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/spybhwd/scm-1/lib/libspy.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/spybhwd/scm-1/lib/libspy.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/spybhwd/scm-1/lib/libspy.so"
         RPATH "$ORIGIN/../lib:/home/zhang205/torch/install/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/spybhwd/scm-1/lib" TYPE MODULE FILES "/home/zhang205/Github/MP-Net-Optical-Flow-Benchmark/Source_Codes/spynet/extras/spybhwd/build/libspy.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/spybhwd/scm-1/lib/libspy.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/spybhwd/scm-1/lib/libspy.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/spybhwd/scm-1/lib/libspy.so"
         OLD_RPATH "/home/zhang205/torch/install/lib:::::::::::::::"
         NEW_RPATH "$ORIGIN/../lib:/home/zhang205/torch/install/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/spybhwd/scm-1/lib/libspy.so")
    endif()
  endif()
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/spybhwd/scm-1/lua/spy" TYPE FILE FILES
    "/home/zhang205/Github/MP-Net-Optical-Flow-Benchmark/Source_Codes/spynet/extras/spybhwd/test.lua"
    "/home/zhang205/Github/MP-Net-Optical-Flow-Benchmark/Source_Codes/spynet/extras/spybhwd/init.lua"
    "/home/zhang205/Github/MP-Net-Optical-Flow-Benchmark/Source_Codes/spynet/extras/spybhwd/ScaleBHWD.lua"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/spybhwd/scm-1/lib/libcuspy.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/spybhwd/scm-1/lib/libcuspy.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/spybhwd/scm-1/lib/libcuspy.so"
         RPATH "$ORIGIN/../lib:/home/zhang205/torch/install/lib:/usr/local/cuda-8.0/lib64")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/spybhwd/scm-1/lib" TYPE MODULE FILES "/home/zhang205/Github/MP-Net-Optical-Flow-Benchmark/Source_Codes/spynet/extras/spybhwd/build/libcuspy.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/spybhwd/scm-1/lib/libcuspy.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/spybhwd/scm-1/lib/libcuspy.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/spybhwd/scm-1/lib/libcuspy.so"
         OLD_RPATH "/home/zhang205/torch/install/lib:/usr/local/cuda-8.0/lib64:::::::::::::::"
         NEW_RPATH "$ORIGIN/../lib:/home/zhang205/torch/install/lib:/usr/local/cuda-8.0/lib64")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/spybhwd/scm-1/lib/libcuspy.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/zhang205/Github/MP-Net-Optical-Flow-Benchmark/Source_Codes/spynet/extras/spybhwd/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
