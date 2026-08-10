#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "myrobot.h"
#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <thread>
#include <fstream>
#include <cmath>
#include <random>
#include <algorithm>
#include <array>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace cnoid::vnoid;

// ========== VNOID_REWARD_LOG_DEBUG ==========
// reward_tracking 検証用の一時ログ。不要になったら:
//   1. この #define を削除（または 0 にする）
//   2. ファイル内の「VNOID_REWARD_LOG_DEBUG」マーカー付き #if ブロックをすべて削除
//   3. python_scripts/plot_reward_log.py を削除
#define VNOID_REWARD_LOG_DEBUG 1
// =============================================

struct FootState {
    double pos[3];
    double vel[6];  // linvel(3) + angvel(3), world frame
};

// ★★★ シンプル設計のVnoidEnv ★★★
class VnoidEnv {
private:
    // MuJoCoのデータ
    mjModel* m = nullptr;
    mjData* d = nullptr;
    
    // vnoidのロボットクラス
    std::unique_ptr<MyRobot> robot;
    double previous_x = 0.0;
    double previous_y = 0.0;
    double step_start_yaw = 0.0;  // 歩開始時点の胴体yaw（両足の中間角）
    Vector3 prev_com_pos_for_reward;  // 報酬計算用の前回CoM位置
    int frame_skip;

    // レンダリング用のオブジェクト（オプショナル）
    mjvCamera cam;
    mjvOption opt;
    mjvScene scn;
    mjrContext con;
    GLFWwindow* window = nullptr;
    
    // 初期化状態を追跡
    bool initialized = false;
    bool rendering_enabled = false;
    bool glfw_initialized = false;
    bool scene_initialized = false;
    bool context_initialized = false;

    // CSVファイルストリーム
    std::ofstream csv_file;
    bool csv_opened = false;
#if VNOID_REWARD_LOG_DEBUG
    std::ofstream reward_csv_file;
    bool reward_csv_opened = false;
    int reward_step_index = 0;
    // reward_tracking() が計算した中間量をログ用に退避（報酬式自体には使わない）
    double last_dx = 0.0, last_dy = 0.0;
    double last_ex_disp = 0.0, last_ey_disp = 0.0;
    double last_ex = 0.0, last_ey = 0.0;
#endif

    // RLアクション保存用
    std::array<double, 2> last_rl_action = {0.0, 0.0};
    Vector3 prev_com_pos;
    Vector3 com_vel_actual;
    bool first_step = true;
    bool logging_completed = false;
    int reset_count = 0;  // reset()が呼ばれた回数
    
    // 角運動量の時間微分計算用
    Vector3 prev_angular_momentum_com = Vector3(0.0, 0.0, 0.0);
    bool first_log = true;

    // 地盤切り替え
    int control_cycle_count = 0;
    int terrain_switch_at = -1;
    std::mt19937 terrain_rng{std::random_device{}()};
    //std::random_device{}()をシードとして入れる
    //terrain_rngはシードから乱数列を作り続けるオブジェクト
    //高品質な乱数生成器　メルセンヌ・ツイスタ
    //１から2^32-1までの整数を生成
    std::uniform_real_distribution<double> terrain_friction_dist{0.8, 1.0};
    std::uniform_real_distribution<double> terrain_sr0_dist{0.02, 0.2};
    std::uniform_real_distribution<double> terrain_sr1_dist{0.9, 110.0};
    std::uniform_real_distribution<double> terrain_si0_dist{0.6, 0.95};
    std::uniform_real_distribution<double> terrain_si1_dist{0.70, 0.99};
    std::uniform_real_distribution<double> terrain_si2_dist{0.002, 0.004};

    // MuJoCo 足 body ID（ログ・デバッグ用）
    int bid_r_foot = -1;
    int bid_l_foot = -1;

    // reward params (1歩スケール)
    double w_track = 0.0;
    double tracking_sigma = 0.02; // 1歩変位誤差のスケール
    double w_act = 0.1;          // まずは0から導入（弱いペナルティ）
    double w_healthy = 1.0;

    // 歩行途中の切り替え先地盤（Pythonから注入。切り替えタイミングはC++側が決める）
    std::string terrain_switch_mode = "debug";
    bool terrain_params_given = false;
    // friction, solref0, solref1, solimp0, solimp1, solimp2
    // 数値6個を直接渡されたとき用の退避先
    double terrain_params[6] = {1.0, 0.1, 2.0, 0.7, 0.85, 0.003};

private:
    double reward_tracking();
    double reward_action_penalty();
    double reward_healthy();

#if VNOID_REWARD_LOG_DEBUG
    void log_reward_step(double tracking, double action_penalty,
                         double healthy, double total);
#endif

public:
    VnoidEnv(const std::string& model_path, bool enable_rendering = false);
    ~VnoidEnv();

    // ★ ログ取得メソッドはCSVファイルから読み込む必要がある場合は削除
    // CSVファイルは外部スクリプトで読み込むことを想定
    py::dict get_control_log();
    
    // ★ ログをクリアするメソッド（CSVファイルは削除しない）
    void clear_control_log();

    py::array_t<double> get_observation();
    double compute_reward();

    // コピー・ムーブを禁止
    VnoidEnv(const VnoidEnv&) = delete;
    VnoidEnv& operator=(const VnoidEnv&) = delete;

    // ★★★ コールバック用のアクセサ ★★★
    mjModel* GetModel() { return m; }
    mjData* GetData() { return d; }
    mjvCamera* GetCamera() { return &cam; }
    mjvScene* GetScene() { return &scn; }

private:
    // 地盤パラメータ適用ヘルパー
    void apply_terrain(double friction,
                       double sr0, double sr1,
                       double si0, double si1, double si2);
    void apply_hard_terrain();
    void apply_soft_terrain();
    void apply_debug_terrain();
    void apply_random_terrain();
    // 歩行途中の切り替え時に、設定された地盤を適用する
    void apply_switch_terrain();

    void initFootBodyIds();
    FootState get_foot_state(int bid);
    // ★ CoM速度を数値微分で計算
    Vector3 calc_com_velocity();
    Vector3 calc_dcm_actual(const Vector3& com_vel);
    // ★ 重心周りの角運動量を計算（MuJoCoから取得）
    Vector3 calc_angular_momentum_around_com();
    void log_control_data();
    void initializeRobot();
    // ★★★ sample_controller_mujocoと同じGLFW初期化 ★★★
    void initializeGLFW();
    // ★★★ sample_controller_mujocoと同じレンダリング初期化 ★★★
    void initializeRenderer();
    void cleanup();

public:
    // 報酬重み・切り替え先地盤を Python から設定
    void set_reward_weights(double w_track_, double w_act_,
                            double w_healthy_, double tracking_sigma_);
    // 地盤設定を辞書で受け取って保持するだけ。適用は step() 内の切替タイミング
    void set_terrain_config(const py::dict& cfg);

    py::array_t<double> reset();
    // Phase3: 外部から歩容コマンドを注入
    void set_walk_command(double stride, double sway, double turn);
    py::tuple step(py::array_t<double> action);

    // ★★★ sample_controller_mujocoと同じ表示更新 ★★★
    void updateDisplay();
    // ★★★ 録画用レンダリング（画面表示とは別） ★★★
    py::array_t<unsigned char> render();

    bool is_rendering_enabled() const {
        return rendering_enabled;
    }

    bool should_close() const {
        return rendering_enabled ? glfwWindowShouldClose(window) : false;
    }

private:
    // GLFWコールバック関数
    static void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods);
    static void mouse_button(GLFWwindow* window, int button, int act, int mods);
    static void mouse_move(GLFWwindow* window, double xpos, double ypos);
    static void scroll(GLFWwindow* window, double xoffset, double yoffset);
};
