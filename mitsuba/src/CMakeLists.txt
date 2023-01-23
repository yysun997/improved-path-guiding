# Utility functions
function(add_mts_plugin)
  list(POP_FRONT ARGV PLUGIN_NAME)
  add_library(${PLUGIN_NAME} SHARED ${ARGV})
  target_link_libraries(${PLUGIN_NAME}
    mitsuba-bidir
    mitsuba-core
    mitsuba-hw
    mitsuba-render)
  set_target_properties(${PLUGIN_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/plugins
    PREFIX "")
endfunction()

function(add_mts_executable)
  list(POP_FRONT ARGV EXECUTABLE_NAME)
  add_executable(${EXECUTABLE_NAME} ${ARGV})
  set_target_properties(${EXECUTABLE_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
endfunction()

function(add_mts_library)
  list(POP_FRONT ARGV LIBRARY_NAME)
  add_library(${LIBRARY_NAME} SHARED ${ARGV})
  set_target_properties(${LIBRARY_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
endfunction()

# Internal libraries dependent by other modules
add_subdirectory(libbidir)
add_subdirectory(libconverter)
add_subdirectory(libcore)
add_subdirectory(libhw)
add_subdirectory(librender)

# Python bindings
add_subdirectory(libpython)

# Shipped plugins
add_subdirectory(bsdfs)
add_subdirectory(emitters)
add_subdirectory(films)
add_subdirectory(integrators)
add_subdirectory(medium)
add_subdirectory(phase)
add_subdirectory(rfilters)
add_subdirectory(samplers)
add_subdirectory(sensors)
add_subdirectory(shapes)
add_subdirectory(subsurface)
add_subdirectory(tests)
add_subdirectory(textures)
add_subdirectory(utils)
add_subdirectory(volume)

# Command-line tools
add_subdirectory(mitsuba)

# GUI
add_subdirectory(mtsgui)
