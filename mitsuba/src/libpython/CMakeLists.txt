if(Python_FOUND)
  add_library(mitsuba-python SHARED core.cpp render.cpp)

  target_compile_options(mitsuba-python PRIVATE -fno-strict-aliasing)
  target_compile_definitions(mitsuba-python PRIVATE MTS_BUILD_MODULE=MTS_MODULE_PYTHON)
  target_include_directories(mitsuba-python PRIVATE ${Python_INCLUDE_DIRS})
  target_link_directories(mitsuba-python PRIVATE
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/plugins)
  target_link_libraries(mitsuba-python
    mitsuba-bidir
    mitsuba-converter
    mitsuba-core
    mitsuba-hw
    mitsuba-render
    Boost::python
    ${Python_LIBRARIES})

  set_target_properties(mitsuba-python PROPERTIES
    LIBRARY_OUTPUT_NAME mitsuba
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python${Python_VERSION}
    PREFIX "")
endif()
