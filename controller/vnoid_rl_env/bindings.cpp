#include "vnoid_env.h"

// ★★★ pybind11モジュール定義（インタラクティブ版） ★★★
PYBIND11_MODULE(vnoid_rl_env, m) {
    py::class_<VnoidEnv>(m, "VnoidEnv")
        .def(py::init<const std::string&>())  // デフォルト：レンダリングなし
        .def(py::init<const std::string&, bool>())  // レンダリング有効化オプション
        .def("step", &VnoidEnv::step)
        .def("reset", &VnoidEnv::reset)
        .def("get_observation", &VnoidEnv::get_observation)
        .def("set_walk_command", &VnoidEnv::set_walk_command)
        .def("set_reward_weights", &VnoidEnv::set_reward_weights)
        .def("set_terrain_config", &VnoidEnv::set_terrain_config)
        .def("set_seed", &VnoidEnv::set_seed)
        .def("should_close", &VnoidEnv::should_close)
        .def("get_control_log", &VnoidEnv::get_control_log)
        .def("clear_control_log", &VnoidEnv::clear_control_log)
        .def("start_paper_log", &VnoidEnv::start_paper_log)
        .def("stop_paper_log", &VnoidEnv::stop_paper_log)
        .def("get_terrain_switch_at", &VnoidEnv::get_terrain_switch_at)
        .def("is_terrain_switched", &VnoidEnv::is_terrain_switched)
        .def("set_camera", &VnoidEnv::set_camera);
}
