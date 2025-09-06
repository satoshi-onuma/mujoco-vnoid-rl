# CMakeのバージョン指定
cmake_minimum_required(VERSION 3.10)
project(vnoid_rl_env)

# 親ディレクトリのsrcを参照
include_directories("../../src")

# 必要なパッケージを探す
find_package(pybind11 REQUIRED)
find_package(mujoco REQUIRED)
find_package(glfw3 REQUIRED)

# Pythonがインポートできるモジュールとしてライブラリをビルド
add_library(vnoid_rl_env MODULE bindings.cpp myrobot.cpp)
set_target_properties(vnoid_rl_env PROPERTIES PREFIX "")

# C++の規格を指定
set_target_properties(vnoid_rl_env PROPERTIES CXX_STANDARD 17)

# インクルードパスの設定
target_include_directories(vnoid_rl_env PUBLIC
    ${pybind11_INCLUDE_DIRS}
    "../../src"
    "."
)

# リンク設定
target_link_libraries(vnoid_rl_env PRIVATE
    vnoid_lib
    mujoco::mujoco
    glfw
)
