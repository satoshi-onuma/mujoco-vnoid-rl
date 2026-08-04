#include "vnoid_env.h"

void VnoidEnv::apply_terrain(double friction,
                   double sr0, double sr1,
                   double si0, double si1, double si2) {
    if (!m) return;
    int fid = mj_name2id(m, mjOBJ_GEOM, "floor");
    if (fid < 0) return;
    m->geom_friction[fid * 3 + 0] = friction;
    m->geom_solref[fid * 2 + 0] = sr0;
    m->geom_solref[fid * 2 + 1] = sr1;
    m->geom_solimp[fid * 5 + 0] = si0;
    m->geom_solimp[fid * 5 + 1] = si1;
    m->geom_solimp[fid * 5 + 2] = si2;
    std::cout << "地盤パラメータ変更: friction=" << friction
              << " solref=(" << sr0 << ", " << sr1 << ")"
              << " solimp=(" << si0 << ", " << si1 << ", " << si2 << ")"
              << " step=" << control_cycle_count << std::endl;
}

void VnoidEnv::apply_hard_terrain() { apply_terrain(1.0, 0.001, 100.0, 0.9, 0.99, 0.003); }
void VnoidEnv::apply_soft_terrain() { apply_terrain(1.0, 0.1, 2.0, 0.7, 0.85, 0.003); }
void VnoidEnv::apply_debug_terrain() { apply_terrain(1.0, 0.2, 1., 0.6, 0.75, 0.003); }

void VnoidEnv::apply_random_terrain() {
    double friction = terrain_friction_dist(terrain_rng);
    double sr0 = terrain_sr0_dist(terrain_rng);
    double sr1 = terrain_sr1_dist(terrain_rng);
    double si0 = terrain_si0_dist(terrain_rng);
    double si1 = terrain_si1_dist(terrain_rng);
    double si2 = terrain_si2_dist(terrain_rng);
    si0 = std::min(si0, si1 - 0.05);
    apply_terrain(friction, sr0, sr1, si0, si1, si2);
}

// Pythonから受け取った地盤設定を保持するだけ。実際の適用は step() の切替タイミング
void VnoidEnv::set_terrain_config(const py::dict& cfg) {
    if (cfg.contains("mode")) {
        terrain_switch_mode = cfg["mode"].cast<std::string>();
    }
    static const char* keys[6] = {
        "friction", "solref0", "solref1", "solimp0", "solimp1", "solimp2"
    };
    bool given = false;
    for (int i = 0; i < 6; ++i) {
        if (cfg.contains(keys[i])) {
            terrain_params[i] = cfg[keys[i]].cast<double>();
            given = true;
        }
    }
    terrain_params_given = given;
    std::cout << "切り替え先地盤の設定: mode=" << terrain_switch_mode
              << (given ? " (明示パラメータあり)" : "") << std::endl;
}

void VnoidEnv::apply_switch_terrain() {
    if (terrain_params_given) {
        apply_terrain(terrain_params[0], terrain_params[1], terrain_params[2],
                      terrain_params[3], terrain_params[4], terrain_params[5]);
        return;
    }
    if (terrain_switch_mode == "hard") {
        apply_hard_terrain();
    } else if (terrain_switch_mode == "soft") {
        apply_soft_terrain();
    } else if (terrain_switch_mode == "debug") {
        apply_debug_terrain();
    } else if (terrain_switch_mode == "random") {
        apply_random_terrain();
    } else {
        std::cerr << "不明な地盤モード: " << terrain_switch_mode
                  << " (hard/soft/debug/random) → softを使用" << std::endl;
        apply_soft_terrain();
    }
}

// std::uniform_real_distribution<double> terrain_friction_dist{0.8, 1.0};
// std::uniform_real_distribution<double> terrain_sr0_dist{0.001, 0.15};
// std::uniform_real_distribution<double> terrain_sr1_dist{1.0, 150.0};
// std::uniform_real_distribution<double> terrain_si0_dist{0.6, 0.95};
// std::uniform_real_distribution<double> terrain_si1_dist{0.80, 0.99};
// std::uniform_real_distribution<double> terrain_si2_dist{0.001, 0.005};
