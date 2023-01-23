add_mts_plugin(ao direct/ao.cpp)
add_mts_plugin(direct direct/direct.cpp)
add_mts_plugin(path path/path.cpp)
add_mts_plugin(volpath path/volpath.cpp)
add_mts_plugin(volpath_simple path/volpath_simple.cpp)
add_mts_plugin(ptracer ptracer/ptracer.cpp ptracer/ptracer_proc.cpp)

# Photon mapping-based techniques
add_mts_plugin(photonmapper photonmapper/photonmapper.cpp photonmapper/bre.cpp)
add_mts_plugin(ppm photonmapper/ppm.cpp)
add_mts_plugin(sppm photonmapper/sppm.cpp)

# Miscellaneous
add_mts_plugin(vpl vpl/vpl.cpp)
add_mts_plugin(adaptive misc/adaptive.cpp)
add_mts_plugin(irrcache misc/irrcache.cpp misc/irrcache_proc.cpp)
add_mts_plugin(multichannel misc/multichannel.cpp)
add_mts_plugin(field misc/field.cpp)
add_mts_plugin(motion misc/motion.cpp)

# Bidirectional techniques
add_mts_plugin(bdpt bdpt/bdpt.cpp bdpt/bdpt_wr.cpp bdpt/bdpt_proc.cpp)
add_mts_plugin(pssmlt pssmlt/pssmlt.cpp pssmlt/pssmlt_sampler.cpp pssmlt/pssmlt_proc.cpp)
add_mts_plugin(mlt mlt/mlt.cpp mlt/mlt_proc.cpp)
add_mts_plugin(erpt erpt/erpt.cpp erpt/erpt_proc.cpp)
