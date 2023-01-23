set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_mts_plugin(obj obj.cpp)
add_mts_plugin(ply ply.cpp ply/ply_parser.cpp)
add_mts_plugin(serialized serialized.cpp)
add_mts_plugin(rectangle rectangle.cpp)
add_mts_plugin(disk disk.cpp)
add_mts_plugin(sphere sphere.cpp)
add_mts_plugin(cylinder cylinder.cpp)
add_mts_plugin(hair hair.cpp)
add_mts_plugin(shapegroup shapegroup.cpp)
add_mts_plugin(instance instance.cpp)
add_mts_plugin(cube cube.cpp)
add_mts_plugin(heightfield heightfield.cpp)

# This plugin is broken
#add_mts_plugin(deformable deformable.cpp)
