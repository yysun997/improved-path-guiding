add_mts_library(mitsuba-render
  bsdf.cpp film.cpp integrator.cpp emitter.cpp sensor.cpp
  skdtree.cpp medium.cpp renderjob.cpp imageproc.cpp
  rectwu.cpp renderproc.cpp imageblock.cpp particleproc.cpp
  renderqueue.cpp scene.cpp  subsurface.cpp texture.cpp
  shape.cpp trimesh.cpp sampler.cpp util.cpp irrcache.cpp
  testcase.cpp photonmap.cpp gatherproc.cpp volume.cpp
  vpl.cpp shader.cpp scenehandler.cpp intersection.cpp
  common.cpp phase.cpp noise.cpp photon.cpp)

target_compile_definitions(mitsuba-render PRIVATE MTS_BUILD_MODULE=MTS_MODULE_RENDER)
target_link_libraries(mitsuba-render ${XercesC_LIBRARIES})
