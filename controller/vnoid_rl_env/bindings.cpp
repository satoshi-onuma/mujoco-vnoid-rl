#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "myrobot.h"
#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>
#include <mutex>
#include <memory>
#include <stdexcept>

namespace py = pybind11;
using namespace cnoid::vnoid;

// ★★★ GLFW管理のスレッドセーフティを強化 ★★★
static std::mutex g_glfw_mutex;
static int g_glfw_init_count = 0;
static bool g_glfw_initialized = false;

// C++側のGym環境を管理するクラス
class VnoidEnv {
private:
    // MuJoCoのデータ
    mjModel* m = nullptr;
    mjData* d = nullptr;
    
    // vnoidのロボットクラス
    std::unique_ptr<MyRobot> robot;
    double previous_x = 0.0;

    // レンダリング用のオブジェクト
    mjvCamera cam;
    mjvOption opt;
    mjvScene scn;
    mjrContext con;
    mjrRect viewport;
    GLFWwindow* window = nullptr;
    
    // 初期化状態を追跡
    bool initialized = false;
    bool scene_initialized = false;
    bool context_initialized = false;

public:
    int frame_skip;

    // コンストラクタ
    VnoidEnv(const std::string& model_path) {
        try {
            initializeGLFW();
            loadModel(model_path);
            initializeRobot();
            initializeRenderer();
            initialized = true;
            const double video_fps = 30.0;
            this->frame_skip = (1.0 / video_fps) / m->opt.timestep;
            printf("DEBUG INFO: Frame skip set to %d\n", this->frame_skip);
        } catch (...) {
            cleanup();
            throw;
        }
    }

    // デストラクタ
    ~VnoidEnv() {
        cleanup();
    }

    // コピー・ムーブを禁止
    VnoidEnv(const VnoidEnv&) = delete;
    VnoidEnv& operator=(const VnoidEnv&) = delete;
    VnoidEnv(VnoidEnv&&) = delete;
    VnoidEnv& operator=(VnoidEnv&&) = delete;

private:
    void initializeGLFW() {
        std::lock_guard<std::mutex> lock(g_glfw_mutex);
        
        if (!g_glfw_initialized) {
            if (!glfwInit()) {
                throw std::runtime_error("GLFWの初期化に失敗しました。");
            }
            g_glfw_initialized = true;
        }
        g_glfw_init_count++;
        
        // 不可視ウィンドウを作成
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE);
        window = glfwCreateWindow(1280, 720, "Headless MuJoCo", nullptr, nullptr);
        
        if (!window) {
            if (--g_glfw_init_count == 0 && g_glfw_initialized) {
                glfwTerminate();
                g_glfw_initialized = false;
            }
            throw std::runtime_error("GLFWウィンドウの作成に失敗しました。");
        }
        
        glfwMakeContextCurrent(window);
    }

    void loadModel(const std::string& model_path) {
        char error[1000] = {0};
        m = mj_loadXML(model_path.c_str(), nullptr, error, 1000);
        
        if (!m) {
            throw std::runtime_error("モデルファイルの読み込みに失敗しました: " + std::string(error));
        }
        
        d = mj_makeData(m);
        if (!d) {
            throw std::runtime_error("MuJoCoデータの作成に失敗しました。");
        }
        
        printf("DEBUG INFO: nq = %d, nv = %d, Total Obs Size = %d\n", m->nq, m->nv, m->nq + m->nv);
    }

    void initializeRobot() {
        robot = std::make_unique<MyRobot>();
        robot->Init(m, d);
    }

    void initializeRenderer() {
        // カメラとオプションの初期化
        mjv_defaultCamera(&cam);
        mjv_defaultOption(&opt);

        // ★★★ カメラ位置をロボット中心に調整 ★★★
        // ロボットを追跡するカメラ設定
        cam.type = mjCAMERA_TRACKING;  // 追跡カメラモード
        cam.trackbodyid = 0;           // 胴体（通常はindex 0）を追跡
        cam.distance = 3.0;            // ロボットからの距離（メートル）
        cam.elevation = -20;           // 仰角（度）- 少し上から見下ろす
        cam.azimuth = 45;              // 方位角（度）- 斜め前から
        
        // または固定カメラの場合：
        // cam.type = mjCAMERA_FREE;
        // cam.lookat[0] = 0.0;  // ロボットの位置を見る（X座標）
        // cam.lookat[1] = 0.0;  // Y座標
        // cam.lookat[2] = 1.0;  // Z座標（ロボットの腰あたりの高さ）
        
        // シーンの初期化
        memset(&scn, 0, sizeof(mjvScene));
        mjv_makeScene(m, &scn, 2000);
        scene_initialized = true;
        
        // レンダリングコンテキストの初期化
        memset(&con, 0, sizeof(mjrContext));
        mjr_makeContext(m, &con, mjFONTSCALE_150);
        context_initialized = true;
        
        // ★★★ ビューポートのサイズを大きくして高画質化 ★★★
        viewport = {0, 0, 1280, 720};  // HD解像度に変更
    }

    void cleanup() {
        // レンダリング関連のクリーンアップ
        if (scene_initialized) {
            mjv_freeScene(&scn);
            scene_initialized = false;
        }
        
        if (context_initialized) {
            mjr_freeContext(&con);
            context_initialized = false;
        }
        
        // MuJoCoデータのクリーンアップ
        if (d) {
            mj_deleteData(d);
            d = nullptr;
        }
        
        if (m) {
            mj_deleteModel(m);
            m = nullptr;
        }
        
        // ロボットのクリーンアップ
        robot.reset();
        
        // GLFWのクリーンアップ
        std::lock_guard<std::mutex> lock(g_glfw_mutex);
        if (window) {
            glfwDestroyWindow(window);
            window = nullptr;
        }
        
        if (--g_glfw_init_count == 0 && g_glfw_initialized) {
            glfwTerminate();
            g_glfw_initialized = false;
        }
    }

public:
    py::array_t<double> reset() {
    if (!initialized) {
        throw std::runtime_error("環境が初期化されていません。");
    }
    
    printf("DEBUG: Reset開始\n");
    
    // 1. MuJoCoの物理状態を完全にリセット
    mj_resetData(m, d);

    double com_height = 0.8;  // デフォルト値
    
    // 2. ロボットの初期姿勢を明示的に設定（MuJoCoのqpos配列を直接操作）
    // ベース位置 (floating base: x, y, z, qw, qx, qy, qz)
    d->qpos[0] = 0.0;  // x位置
    d->qpos[1] = 0.0;  // y位置  
    d->qpos[2] = com_height;  // z位置（重心高さ）
    d->qpos[3] = 1.0;  // qw (クォータニオンw成分)
    d->qpos[4] = 0.0;  // qx
    d->qpos[5] = 0.0;  // qy
    d->qpos[6] = 0.0;  // qz
    
    // 関節角度をすべて0にリセット（7番目以降が関節角度）
    for (int i = 7; i < m->nq; ++i) {
        d->qpos[i] = 0.0;
    }
    
    // 速度もすべて0にリセット
    for (int i = 0; i < m->nv; ++i) {
        d->qvel[i] = 0.0;
    }
    
    // 加速度も0にリセット
    for (int i = 0; i < m->nv; ++i) {
        d->qacc[i] = 0.0;
    }
    
    // 制御入力も0にリセット
    for (int i = 0; i < m->nu; ++i) {
        d->ctrl[i] = 0.0;
    }
    
    // 3. MuJoCoの順運動学を実行して、位置・速度・力を更新
    mj_forward(m, d);
    
    // 4. vnoidの内部状態を完全にリセット
    robot->ResetState();
    
    // 5. 追跡用変数もリセット
    previous_x = d->qpos[0];
    
    printf("DEBUG: Reset完了 - MuJoCo状態とVnoid状態がリセットされました\n");
    printf("DEBUG: 初期位置 = (%.3f, %.3f, %.3f)\n", d->qpos[0], d->qpos[1], d->qpos[2]);
    printf("DEBUG: 初期クォータニオン = (%.3f, %.3f, %.3f, %.3f)\n", 
           d->qpos[3], d->qpos[4], d->qpos[5], d->qpos[6]);
    
    return get_observation();
}

    py::tuple step(py::array_t<double> action) {
        if (!initialized) {
            throw std::runtime_error("環境が初期化されていません。");
        }
        
        auto buf = action.request();
        if (buf.ndim != 1 || buf.size < 2) {
            throw std::runtime_error("アクションの次元またはサイズが不正です。");
        }
        
        double* ptr = static_cast<double*>(buf.ptr);
        RLParams rl_params;
        rl_params.foot_offset.x() = ptr[0];
        rl_params.foot_offset.y() = ptr[1];

        // vnoidの制御サイクルとMuJoCoのシミュレーション
        // ★★★ フレームスキップを実装 ★★★
        for (int i = 0; i < this->frame_skip; ++i) {
            robot->Control(rl_params);
            mj_step(m, d);
        }

        py::array_t<double> obs = get_observation();
        double reward = compute_reward();
        bool terminated = check_termination();
        
        return py::make_tuple(obs, reward, terminated, py::dict());
    }

    py::array_t<unsigned char> render() {
        if (!initialized || !scene_initialized || !context_initialized) {
            throw std::runtime_error("レンダリング環境が初期化されていません。");
        }
        
        try {

            for (int i = 0; i < 3; ++i) {
            cam.lookat[i] = d->qpos[i];
        }
            // シーンデータを更新
            mjv_updateScene(m, d, &opt, nullptr, &cam, mjCAT_ALL, &scn);
            
            // シーンを描画バッファにレンダリング
            mjr_render(viewport, &scn, &con);
            
            // ピクセルデータを読み出し
            size_t buffer_size = viewport.width * viewport.height * 3;
            auto buffer = std::make_unique<unsigned char[]>(buffer_size);
            mjr_readPixels(buffer.get(), nullptr, viewport, &con);
            
            // Python配列に変換（スマートポインタを使用してメモリ安全性を確保）
            py::capsule free_when_done(buffer.release(), [](void *f) {
                delete[] static_cast<unsigned char *>(f);
            });
            
            return py::array_t<unsigned char>(
                {viewport.height, viewport.width, 3},
                {viewport.width * 3, 3, 1},
                static_cast<unsigned char*>(free_when_done.get_pointer()),
                free_when_done
            );
        } catch (const std::exception& e) {
            throw std::runtime_error("レンダリング中にエラーが発生しました: " + std::string(e.what()));
        }
    }

private:
    py::array_t<double> get_observation() {
        const int obs_size = m->nq + m->nv;
        py::array_t<double> obs(obs_size);
        auto obs_ptr = obs.mutable_data();

        memcpy(obs_ptr, d->qpos, m->nq * sizeof(double));
        memcpy(obs_ptr + m->nq, d->qvel, m->nv * sizeof(double));

        return obs;
    }

    double compute_reward() {
        double current_x = d->qpos[0];
        double forward_reward = current_x - previous_x;
        previous_x = current_x;
        
        double healthy_reward = 1.0;
        return forward_reward * 5.0 + healthy_reward;
    }

    bool check_termination() {
        double hips_z_position = d->qpos[2];
        return (hips_z_position < 0.5);
    }
};

// pybind11モジュール定義
PYBIND11_MODULE(vnoid_rl_env, m) {
    py::class_<VnoidEnv>(m, "VnoidEnv")
        .def(py::init<const std::string&>())
        .def("step", &VnoidEnv::step)
        .def("reset", &VnoidEnv::reset)
        .def("render", &VnoidEnv::render);
}