// ★★★ sample_controller_mujocoベース：インタラクティブ表示版 ★★★

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

namespace py = pybind11;
using namespace cnoid::vnoid;

// ★★★ コールバック関数の前方宣言 ★★★
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods);
void mouse_button(GLFWwindow* window, int button, int act, int mods);
void mouse_move(GLFWwindow* window, double xpos, double ypos);
void scroll(GLFWwindow* window, double xoffset, double yoffset);

// ★★★ sample_controller_mujocoと同じグローバル変数（インタラクション用） ★★★
// これらは1つのVnoidEnvインスタンスでのみ使用される
static bool button_left = false;
static bool button_middle = false;
static bool button_right = false;
static double lastx = 0;
static double lasty = 0;



// ★★★ シンプル設計のVnoidEnv ★★★
class VnoidEnv {
private:
    // MuJoCoのデータ
    mjModel* m = nullptr;
    mjData* d = nullptr;
    
    // vnoidのロボットクラス
    std::unique_ptr<MyRobot> robot;
    double previous_x = 0.0;
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

public:
    

    VnoidEnv(const std::string& model_path, bool enable_rendering = false) 
        : rendering_enabled(enable_rendering) {
               std::cout << "VnoidEnv初期化開始 (レンダリング: " 
                  << (enable_rendering ? "有効" : "無効") << ")" << std::endl;
        
        // MuJoCoモデル読み込み（常に実行）
        char error[1000] = "Could not load model";
        m = mj_loadXML(model_path.c_str(), nullptr, error, 1000);
        if (!m) {
            throw std::runtime_error("モデル読み込み失敗: " + std::string(error));
        }
        
        d = mj_makeData(m);
        if (!d) {
            throw std::runtime_error("データ初期化失敗");
        }
        
        // ロボットコントローラ初期化
        robot = std::make_unique<MyRobot>();
        robot->Init(m, d);
        
        // ★★★ sample_controller_mujocoと同じ60fps制御 ★★★
        // 1/60秒 = 0.01667秒をシミュレーション時間で進める
        const double render_period = 1.0 / 60.0;  // 60fps
        frame_skip = static_cast<int>(render_period / m->opt.timestep);
        
        std::cout << "フレームスキップ設定: " << frame_skip 
                  << " (60fps制御, MuJoCo timestep=" << m->opt.timestep << "秒)" << std::endl;
        
        // ★★★ レンダリングが有効な場合のみOpenGL初期化 ★★★
        if (rendering_enabled) {
            try {
                initializeGLFW();
                initializeRenderer();
                std::cout << "✅ OpenGL初期化成功" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "⚠️ OpenGL初期化失敗: " << e.what() << std::endl;
                std::cerr << "⚠️ レンダリングを無効化して続行します" << std::endl;
                rendering_enabled = false;
                
                // 部分的に初期化された状態をクリーンアップ
                if (window) {
                    glfwDestroyWindow(window);
                    window = nullptr;
                }
                if (glfw_initialized) {
                    glfwTerminate();
                    glfw_initialized = false;
                }
            }
        }
        
        initialized = true;
        previous_x = d->qpos[0];
        
        std::cout << "✅ VnoidEnv初期化完了" << std::endl;
    }

    ~VnoidEnv() {
        cleanup();
    }

    py::array_t<double> get_observation() {
        std::vector<double> obs;
        obs.reserve(m->nq + m->nv);
        
        for (int i = 0; i < m->nq; ++i) obs.push_back(d->qpos[i]);
        for (int i = 0; i < m->nv; ++i) obs.push_back(d->qvel[i]);
        
        return py::array_t<double>(obs.size(), obs.data());
    }

    double compute_reward() {
        double current_x = d->qpos[0];
        double forward_reward = current_x - previous_x;
        previous_x = current_x;
        
        double healthy_reward = 1.0;
        return forward_reward * 5.0 + healthy_reward;
    }

    // コピー・ムーブを禁止
    VnoidEnv(const VnoidEnv&) = delete;
    VnoidEnv& operator=(const VnoidEnv&) = delete;

    // ★★★ コールバック用のアクセサ ★★★
    mjModel* GetModel() { return m; }
    mjData* GetData() { return d; }
    mjvCamera* GetCamera() { return &cam; }
    mjvScene* GetScene() { return &scn; }

private:
    

    void initializeRobot() {
        robot = std::make_unique<MyRobot>();
        robot->Init(m, d);
    }

    // ★★★ sample_controller_mujocoと同じGLFW初期化 ★★★
    void initializeGLFW() {
        // init GLFW (sample_controller_mujocoと同じ)
        if (!glfwInit()) {
            throw std::runtime_error("Could not initialize GLFW");
        }
        glfw_initialized = true;

        // ★★★ sample_controller_mujocoと同じ可視ウィンドウ作成 ★★★
        window = glfwCreateWindow(1200, 900, "VnoidEnv Interactive", NULL, NULL);
        
        if (!window) {
            glfwTerminate();
            glfw_initialized = false;
            throw std::runtime_error("GLFWウィンドウの作成に失敗しました。");
        }
        
        glfwMakeContextCurrent(window);
        glfwSwapInterval(1);

        // ★★★ sample_controller_mujocoと同じコールバック設定 ★★★
        glfwSetWindowUserPointer(window, this);  // thisポインタを保存
        glfwSetKeyCallback(window, keyboard);
        glfwSetCursorPosCallback(window, mouse_move);
        glfwSetMouseButtonCallback(window, mouse_button);
        glfwSetScrollCallback(window, scroll);
    }

    // ★★★ sample_controller_mujocoと同じレンダリング初期化 ★★★
    void initializeRenderer() {
        // ★★★ sample_controller_mujocoと同じ初期化 ★★★
        mjv_defaultCamera(&cam);
        mjv_defaultOption(&opt);
        mjv_defaultScene(&scn);     // ← これが足りない！
mjr_defaultContext(&con);
        
        // create scene and context (sample_controller_mujocoと同じ)
        mjv_makeScene(m, &scn, 2000);
        scene_initialized = true;
        
        mjr_makeContext(m, &con, mjFONTSCALE_150);
        context_initialized = true;
    }

    void cleanup() {
        // ロボットのクリーンアップ
        robot.reset();
        
        // ★★★ sample_controller_mujocoと同じクリーンアップ順序 ★★★
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
        
        // ★★★ sample_controller_mujocoと同じGLFWクリーンアップ ★★★
        if (window) {
            glfwDestroyWindow(window);
            window = nullptr;
        }
        
        if (glfw_initialized) {
            // terminate GLFW (sample_controller_mujocoと同じコメント付き)
            #if defined(__APPLE__) || defined(_WIN32)
            glfwTerminate();
            #endif
            glfw_initialized = false;
        }
    }

public:
    py::array_t<double> reset() {
    if (!initialized) {
        throw std::runtime_error("環境が初期化されていません。");
    }
    
    // データを完全に削除して再作成
    mj_deleteData(d);
    d = mj_makeData(m);
    
    // ロボットも完全に再初期化
    robot = std::make_unique<MyRobot>();
    robot->Init(m, d);
    
    mj_forward(m, d);
    previous_x = d->qpos[0];
    
    return get_observation();
}

    py::tuple step(py::array_t<double> action) {
        if (!initialized) {
            throw std::runtime_error("環境が初期化されていません。");
        }
         // ★★★ デバッグ用のログ出力を追加 ★★★
    // どのプロセスがstepを呼び出したか確認
    // C++11のスレッドIDを使って、簡易的にワーカーを識別
    std::cout << "[Worker " << std::this_thread::get_id() << "] Python step() called." << std::endl;
        
        auto buf = action.request();
        if (buf.ndim != 1 || buf.size < 5) {
            throw std::runtime_error("アクションの次元またはサイズが不正です。");
        }
        
        double* ptr = static_cast<double*>(buf.ptr);
        RLParams rl_params;
         
        // actionから設定
        rl_params.foot_offset.x() = ptr[0];
        rl_params.foot_offset.y() = ptr[1];
        rl_params.foot_angle_offset.x() = ptr[2];
        rl_params.foot_angle_offset.y() = ptr[3];
        rl_params.foot_angle_offset.z() = ptr[4];

        

        bool step_completed = false;
        int prev_support_leg = robot->footstep_buffer.steps[0].side;
        int step_counter = 0;

        const int MAX_ITERATIONS = 1000;
        const double MIN_HEIGHT = 0.5;
        bool terminated = false;
        bool timeout = false;

        // ★★★ sample_controller_mujocoと同じ制御パターン ★★★
        while (!step_completed && !terminated) {
            // ★ 制御サイクル実行（frame_skip回のmj_step）
            for (int i = 0; i < this->frame_skip; ++i) {
                robot->Control(rl_params);
                mj_step(m, d);

                 // 毎ステップ転倒チェック
                double hips_z = d->qpos[2];
                if (hips_z < MIN_HEIGHT) {
                    terminated = true;
                    std::cout << "[INFO] Robot fell (height=" << hips_z << ")" << std::endl;
                    break;
                }
                
            }

            if (terminated) break;
        
            // レンダリング更新（既存コードと同じ）
            if (rendering_enabled && !glfwWindowShouldClose(window)) {
                updateDisplay();
            }
        
            // ★ 1歩完了を検出
            int current_support_leg = robot->footstep_buffer.steps[0].side;
            step_completed = (current_support_leg != prev_support_leg);
        
            if (step_completed) {
                std::cout << "[INFO] Support leg changed: " << prev_support_leg 
                        << " -> " << current_support_leg << std::endl;
            }

            step_counter++;

            if (step_counter  >= MAX_ITERATIONS) {
                terminated = true;
                timeout = true;  // タイムアウトフラグをセット
                std::cerr << "[WARNING] Step timeout" << std::endl;
                break;
            }
    
        
        
        }

        if (step_completed && !terminated) {
            std::cout << "[INFO] Step completed after " << step_counter << " iterations" << std::endl;
        }

        py::array_t<double> obs = get_observation();
        double reward = compute_reward();

        if (timeout) {
            reward = -10.0;
        }
        
            return py::make_tuple(obs, reward, terminated, py::dict());
        }

    // ★★★ sample_controller_mujocoと同じ表示更新 ★★★
    void updateDisplay() {
        if (!rendering_enabled || !window) return;

        // get framebuffer viewport (sample_controller_mujocoと同じ)
        mjrRect viewport = {0, 0, 0, 0};
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

        // update scene and render (sample_controller_mujocoと同じ)
        mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
        mjr_render(viewport, &scn, &con);

        // swap OpenGL buffers (sample_controller_mujocoと同じ)
        glfwSwapBuffers(window);

        // process pending GUI events (sample_controller_mujocoと同じ)
        glfwPollEvents();
    }

    // ★★★ 録画用レンダリング（画面表示とは別） ★★★
    py::array_t<unsigned char> render() {
        if (!rendering_enabled || !scene_initialized || !context_initialized) {
            throw std::runtime_error("レンダリングが無効化されています。");
        }
        
        try {
            // 録画用の固定ビューポート
            mjrRect viewport = {0, 0, 1280, 720};
            
            // シーン更新・レンダリング
            mjv_updateScene(m, d, &opt, nullptr, &cam, mjCAT_ALL, &scn);
            mjr_render(viewport, &scn, &con);
            
            // ピクセルデータを読み出し
            size_t buffer_size = viewport.width * viewport.height * 3;
            auto buffer = std::make_unique<unsigned char[]>(buffer_size);
            mjr_readPixels(buffer.get(), nullptr, viewport, &con);
            
            // Python配列に変換
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

    bool is_rendering_enabled() const {
        return rendering_enabled;
    }

    bool should_close() const {
        return rendering_enabled ? glfwWindowShouldClose(window) : false;
    }

private:
    // GLFWコールバック関数
    static void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods) {
        if (act == GLFW_PRESS && key == GLFW_KEY_BACKSPACE) {
            VnoidEnv* env = static_cast<VnoidEnv*>(glfwGetWindowUserPointer(window));
            if (env) {
                env->reset();
            }
        }
    }

    static void mouse_button(GLFWwindow* window, int button, int act, int mods) {
        button_left = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
        button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
        button_right = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);
        glfwGetCursorPos(window, &lastx, &lasty);
    }

    static void mouse_move(GLFWwindow* window, double xpos, double ypos) {
        VnoidEnv* env = static_cast<VnoidEnv*>(glfwGetWindowUserPointer(window));
        if (!env || !button_left && !button_middle && !button_right) return;

        double dx = xpos - lastx;
        double dy = ypos - lasty;
        lastx = xpos;
        lasty = ypos;

        int width, height;
        glfwGetWindowSize(window, &width, &height);

        bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                          glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);

        mjtMouse action;
        if (button_right) {
            action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
        } else if (button_left) {
            action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
        } else {
            action = mjMOUSE_ZOOM;
        }

        mjv_moveCamera(env->GetModel(), action, dx/height, dy/height, env->GetScene(), env->GetCamera());
    }

    static void scroll(GLFWwindow* window, double xoffset, double yoffset) {
        VnoidEnv* env = static_cast<VnoidEnv*>(glfwGetWindowUserPointer(window));
        if (!env) return;
        mjv_moveCamera(env->GetModel(), mjMOUSE_ZOOM, 0, -0.05*yoffset, env->GetScene(), env->GetCamera());
    }

    
};



// ★★★ pybind11モジュール定義（インタラクティブ版） ★★★
PYBIND11_MODULE(vnoid_rl_env, m) {
    py::class_<VnoidEnv>(m, "VnoidEnv")
        .def(py::init<const std::string&>())  // デフォルト：レンダリングなし
        .def(py::init<const std::string&, bool>())  // レンダリング有効化オプション
        .def("step", &VnoidEnv::step)
        .def("reset", &VnoidEnv::reset)
        .def("render", &VnoidEnv::render)
        .def("update_display", &VnoidEnv::updateDisplay)
        .def("should_close", &VnoidEnv::should_close)
        .def("is_rendering_enabled", &VnoidEnv::is_rendering_enabled);
}