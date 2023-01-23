link_libraries(
  mitsuba-core
  mitsuba-hw
  mitsuba-render)

if(COLLADA_DOM_FOUND)
  add_mts_executable(mtsimport mtsimport.cpp)
  target_link_libraries(mtsimport mitsuba-converter)
endif()

link_libraries(
  ${XercesC_LIBRARIES}
  ${GL_LIBRARIES})

add_mts_executable(mtsutil mtsutil.cpp)
add_mts_executable(mtssrv mtssrv.cpp)
add_mts_executable(mitsuba mitsuba.cpp)
