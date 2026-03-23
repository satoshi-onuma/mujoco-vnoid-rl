include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_cnoid_simple_controller(vnoid_rl_controller main.cpp shm_interface.h)

# リアルタイム拡張ライブラリをリンク（共有メモリ用）
target_link_libraries(vnoid_rl_controller rt)
