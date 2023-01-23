# Basic library of smooth and rough materials
add_mts_plugin(diffuse diffuse.cpp)
add_mts_plugin(dielectric dielectric.cpp)
add_mts_plugin(conductor conductor.cpp)
add_mts_plugin(plastic plastic.cpp)
add_mts_plugin(roughdiffuse roughdiffuse.cpp)
add_mts_plugin(roughdielectric roughdielectric.cpp)
add_mts_plugin(roughconductor roughconductor.cpp)
add_mts_plugin(roughplastic roughplastic.cpp)

# Materials that act as modifiers
add_mts_plugin(twosided twosided.cpp)
add_mts_plugin(mask mask.cpp)
add_mts_plugin(mixturebsdf mixturebsdf.cpp)
add_mts_plugin(blendbsdf blendbsdf.cpp)
add_mts_plugin(coating coating.cpp)
add_mts_plugin(roughcoating roughcoating.cpp)
add_mts_plugin(bumpmap bumpmap.cpp)
add_mts_plugin(normalmap normalmap.cpp)

# Other materials
add_mts_plugin(ward ward.cpp)
add_mts_plugin(phong phong.cpp)
add_mts_plugin(difftrans difftrans.cpp)
add_mts_plugin(hk hk.cpp)
add_mts_plugin(null null.cpp)
add_mts_plugin(thindielectric thindielectric.cpp)
add_mts_plugin(irawan irawan.cpp)
