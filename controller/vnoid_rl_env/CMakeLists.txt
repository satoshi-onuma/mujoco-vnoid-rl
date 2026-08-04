# CMakeのバージョン指定
cmake_minimum_required(VERSION 3.10)
project(vnoid_rl_env)

# 親ディレクトリのsrcを参照
include_directories("../../src")

# 必要なパッケージを探す
find_package(pybind11 REQUIRED)
find_package(mujoco REQUIRED)
find_package(glfw3 REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Pythonがインポートできるモジュールとしてライブラリをビルド
add_library(vnoid_rl_env MODULE
    bindings.cpp
    myrobot.cpp
    vnoid_env_lifecycle.cpp
    vnoid_env_reward.cpp
    vnoid_env_episode.cpp
    vnoid_env_terrain.cpp
    vnoid_env_logging.cpp
    vnoid_env_render.cpp
)
set_target_properties(vnoid_rl_env PROPERTIES PREFIX "")

# C++の規格を指定
set_target_properties(vnoid_rl_env PROPERTIES CXX_STANDARD 17)

# インクルードパスの設定
target_include_directories(vnoid_rl_env PUBLIC
    ${pybind11_INCLUDE_DIRS}
    "../../src"
    "."
)

target_include_directories(vnoid_rl_env PRIVATE
    ${Python3_INCLUDE_DIRS}
)

# リンク設定
target_link_libraries(vnoid_rl_env PRIVATE
    vnoid_lib
    mujoco::mujoco
    glfw
    Python3::Python    # ★ 追加
)
