add_mts_library(mitsuba-core
  class.cpp object.cpp statistics.cpp thread.cpp brent.cpp
  logger.cpp appender.cpp formatter.cpp lock.cpp qmc.cpp
  random.cpp timer.cpp  util.cpp properties.cpp half.cpp
  transform.cpp spectrum.cpp aabb.cpp stream.cpp math.cpp
  fstream.cpp plugin.cpp triangle.cpp bitmap.cpp
  fmtconv.cpp serialization.cpp sstream.cpp cstream.cpp
  mstream.cpp sched.cpp sched_remote.cpp sshstream.cpp
  zstream.cpp shvector.cpp fresolver.cpp rfilter.cpp
  quad.cpp mmap.cpp chisquare.cpp warp.cpp vmf.cpp
  tls.cpp ssemath.cpp spline.cpp track.cpp)

target_compile_definitions(mitsuba-core PRIVATE MTS_BUILD_MODULE=MTS_MODULE_CORE)
target_link_libraries(mitsuba-core ${PNG_LIBRARIES} ${JPEG_LIBRARIES} ${FFTW_LIBRARIES})
