link_libraries(stdc++ dl m pthread)

# Check necessary dependencies
include(FindPkgConfig)

find_package(XercesC REQUIRED)

find_package(Boost REQUIRED COMPONENTS system filesystem thread python)
link_libraries(Boost::system Boost::filesystem Boost::thread)

find_package(Eigen3 REQUIRED CONFIG)
include_directories(${EIGEN3_INCLUDE_DIRS})

find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${OpenMP_SHARED_LINKER_FLAGS}")

set(OpenGL_GL_PREFERENCE LEGACY)
find_package(OpenGL REQUIRED)
pkg_check_modules(GLEWMX REQUIRED glewmx)
pkg_check_modules(Xxf86vm REQUIRED xxf86vm)
find_package(X11 REQUIRED)

set(GL_LIBRARIES
  OpenGL::GL
  OpenGL::GLU
  ${GLEWMX_LIBRARIES}
  ${Xxf86vm_LIBRARIES}
  ${X11_LIBRARIES})

# Check optional Dependencies
find_package(PNG)
if(NOT PNG_FOUND)
  message(WARNING "PNG library not found -- PNG I/O not available")
else()
  add_compile_definitions(MTS_HAS_LIBPNG=1)
endif()

find_package(JPEG)
if(NOT JPEG_FOUND)
  message(WARNING "JPEG library not found -- JPEG I/O not available")
else()
  add_compile_definitions(MTS_HAS_LIBJPEG=1)
endif()

pkg_check_modules(OpenEXR OpenEXR)
if(NOT OpenEXR_FOUND)
  message(WARNING "OpenEXR library not found -- OpenEXR I/O not available")
else()
  add_compile_definitions(MTS_HAS_OPENEXR=1)
  include_directories(${OpenEXR_INCLUDE_DIRS})
  link_libraries(${OpenEXR_LIBRARIES})
endif()

find_package(COLLADA_DOM COMPONENTS 1.4.1 CONFIG)
if(NOT COLLADA_DOM_FOUND)
  message(WARNING "Collada library not found -- Collada importer is disabled")
else()
  add_compile_definitions(MTS_HAS_COLLADA=1)
endif()

pkg_check_modules(FFTW fftw3)
if(NOT FFTW_FOUND)
  message(WARNING "FFTW library not found -- Fast image convolution not available")
else()
  add_compile_definitions(MTS_HAS_FFTW=1)
  set(FFTW_LIBRARIES fftw3_threads fftw3)
endif()

find_package(Qt5 COMPONENTS Gui Widgets Core OpenGL Xml XmlPatterns Network)
if(NOT Qt5_FOUND)
  message(WARNING "Qt5 library not found -- Mitsuba GUI not available")
else()
  set(Qt5_LIBRARIES Qt5::Gui Qt5::Widgets Qt5::Core Qt5::OpenGL Qt5::Xml Qt5::XmlPatterns Qt5::Network)
endif()

find_package(Python COMPONENTS Interpreter Development)
if(NOT Python_FOUND)
  message(WARNING "Python interpreter or library not found -- Python API not available")
endif()
