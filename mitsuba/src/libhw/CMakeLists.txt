add_mts_library(mitsuba-hw
  session.cpp device.cpp gputexture.cpp gpugeometry.cpp
  gpuprogram.cpp renderer.cpp glrenderer.cpp glprogram.cpp
  glgeometry.cpp gltexture.cpp gpusync.cpp glsync.cpp
  vpl.cpp font.cpp viewer.cpp basicshader.cpp shadow.cpp)

target_compile_definitions(mitsuba-hw PRIVATE MTS_BUILD_MODULE=MTS_MODULE_HW)

if(${CMAKE_SYSTEM_NAME} MATCHES Linux)
  target_sources(mitsuba-hw PRIVATE
    x11session.cpp x11device.cpp glxdevice.cpp glxrenderer.cpp)
endif()

target_link_libraries(mitsuba-hw ${GL_LIBRARIES})
