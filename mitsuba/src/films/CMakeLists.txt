add_mts_plugin(mfilm mfilm.cpp cnpy.cpp)
add_mts_plugin(ldrfilm ldrfilm.cpp)
add_mts_plugin(hdrfilm hdrfilm.cpp)

if(OpenEXR_FOUND)
  add_mts_plugin(tiledhdrfilm tiledhdrfilm.cpp)
endif()
