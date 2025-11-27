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
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

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

    struct ControlLog {
        std::vector<double> time;
        
        // 実測値
        std::vector<double> com_pos_x, com_pos_y, com_pos_z;
        std::vector<double> com_vel_x, com_vel_y, com_vel_z;
        std::vector<double> zmp_x, zmp_y, zmp_z;
        std::vector<double> dcm_x, dcm_y, dcm_z;
        
        // 目標値
        std::vector<double> com_pos_ref_x, com_pos_ref_y, com_pos_ref_z;
        std::vector<double> zmp_ref_x, zmp_ref_y, zmp_ref_z;
        std::vector<double> dcm_ref_x, dcm_ref_y, dcm_ref_z;
        
        void clear() {
            time.clear();
            com_pos_x.clear(); com_pos_y.clear(); com_pos_z.clear();
            com_vel_x.clear(); com_vel_y.clear(); com_vel_z.clear();
            zmp_x.clear(); zmp_y.clear(); zmp_z.clear();
            dcm_x.clear(); dcm_y.clear(); dcm_z.clear();
            com_pos_ref_x.clear(); com_pos_ref_y.clear(); com_pos_ref_z.clear();
            zmp_ref_x.clear(); zmp_ref_y.clear(); zmp_ref_z.clear();
            dcm_ref_x.clear(); dcm_ref_y.clear(); dcm_ref_z.clear();
        }
        
        void reserve(size_t n) {
            time.reserve(n);
            com_pos_x.reserve(n); com_pos_y.reserve(n); com_pos_z.reserve(n);
            com_vel_x.reserve(n); com_vel_y.reserve(n); com_vel_z.reserve(n);
            zmp_x.reserve(n); zmp_y.reserve(n); zmp_z.reserve(n);
            dcm_x.reserve(n); dcm_y.reserve(n); dcm_z.reserve(n);
            com_pos_ref_x.reserve(n); com_pos_ref_y.reserve(n); com_pos_ref_z.reserve(n);
            zmp_ref_x.reserve(n); zmp_ref_y.reserve(n); zmp_ref_z.reserve(n);
            dcm_ref_x.reserve(n); dcm_ref_y.reserve(n); dcm_ref_z.reserve(n);
        }
    };

    ControlLog control_log;

    Vector3 prev_com_pos;
    Vector3 com_vel_actual;
    bool first_step = true;

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
        
        // 1/60秒 = 0.01667秒をシミュレーション時間で進める
        const double render_period = 1.0 / 60.0;  // 60fps
        frame_skip = static_cast<int>(render_period / m->opt.timestep);
        
        std::cout << "フレームスキップ設定: " << frame_skip 
                  << " (60fps制御, MuJoCo timestep=" << m->opt.timestep << "秒)" << std::endl;
        
        // ★★★ レンダリングが有効な場合のみOpenGL初期化 ★★★ログも初期化
        if (rendering_enabled) {

            control_log.clear();
            control_log.reserve(100000);
            std::cout << "📊 制御データのロギング自動開始" << std::endl;

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

        prev_com_pos = Vector3(0.0, 0.0, 0.0);
        com_vel_actual = Vector3(0.0, 0.0, 0.0);
        first_step = true;
        
        std::cout << "✅ VnoidEnv初期化完了" << std::endl;
    }

    ~VnoidEnv() {

        if (rendering_enabled &&!control_log.time.empty()) {
            try {
                generate_plots();
            } catch (const std::exception& e) {
                std::cerr << "⚠️ グラフ生成エラー: " << e.what() << std::endl;
            }
        }
        cleanup();
    }

    py::array_t<double> get_observation() {
        std::vector<double> obs;
        
        obs.push_back(robot->base.angvel[0]);
        obs.push_back(robot->base.angvel[1]);
        obs.push_back(robot->base.angvel[2]);

        obs.push_back(robot->base.ori.w()); 
        obs.push_back(robot->base.ori.x()); 
        obs.push_back(robot->base.ori.y()); 
        obs.push_back(robot->base.ori.z()); 

        obs.push_back(robot->base.acc[0]);    // 加速度
        obs.push_back(robot->base.acc[1]);
        obs.push_back(robot->base.acc[2]);
        
        obs.push_back(robot->foot[0].contact_ref ? 1.0 : 0.0);
        obs.push_back(robot->foot[1].contact_ref ? 1.0 : 0.0);

        obs.push_back(robot->foot[0].pos_ref[2]);
        obs.push_back(robot->foot[1].pos_ref[2]);

        double left_foot_sink = robot->foot[0].pos[2] - robot->foot[0].pos_ref[2]  ;//実測-予測　予想より下にいたら-,上にいたら+
        double right_foot_sink = robot->foot[0].pos[2] - robot->foot[1].pos_ref[2] ;//実測足の位置使っているのでここだけジャイロ以外使用

        obs.push_back(left_foot_sink); 
        obs.push_back(right_foot_sink); 

        /*
        DCMの修正した大きさを入れる　デルタ入れたらいい?
        Vector3 dcm_error = centroid.dcm_ref - centroid.dcm_target;
        obs.push_back(dcm_error[0]);
        obs.push_back(dcm_error[1]);
        */
        


        //設計思想に合わせて追加
    
        return py::array_t<double>(obs.size(), obs.data());
    }

    double compute_reward() {
        double current_x = d->qpos[0];
        double forward_reward = current_x - previous_x;
        double healthy_reward = 1.0;
        double total_reward = forward_reward * 5.0 + healthy_reward;
    
        // ★デバッグログ
        std::cout << "[REWARD] current_x=" << current_x 
                 << " | prev_x=" << previous_x 
                << " | forward=" << forward_reward 
                 << " | total=" << total_reward << std::endl;

        //ここで返してるトータルリワードとPython側で受け取ってるRewardの値が若干違う
        //並列環境減らすなどして試す
        //現時点では50stepで終わってないほうが凶悪
    
        previous_x = current_x;
        return total_reward;
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


    // ★ CoM速度を数値微分で計算
    Vector3 calc_com_velocity() {
        if (!robot) {
            return Vector3(0.0, 0.0, 0.0);
        }
        
        Vector3 current_com_pos = robot->centroid.com_pos;
        
        if (first_step) {
            // 初回は速度ゼロ
            prev_com_pos = current_com_pos;
            com_vel_actual = Vector3(0.0, 0.0, 0.0);
            first_step = false;
        } else {
            // 数値微分： v = (x[t] - x[t-1]) / dt
            double dt = robot->timer.dt;
            if (dt > 1e-10) {  // ゼロ除算回避
                com_vel_actual = (current_com_pos - prev_com_pos) / dt;
            }
            prev_com_pos = current_com_pos;
        }
        
        return com_vel_actual;
    }

    Vector3 calc_dcm_actual() {
        // DCM = CoM_pos + T * CoM_vel
        // T = sqrt(h/g) はLIPMの時定数
        double T = robot->param.T;
        Vector3 com_vel = calc_com_velocity();
        return robot->centroid.com_pos + T * com_vel;
        // Note: com_velの実測値がない場合はcom_vel_refを使う
        // より正確にはFK計算からcom_velを取得すべき
    }

    void log_control_data() {
        if (!rendering_enabled|| !robot) return;
        
        // CoM速度を計算（キャッシュされる）
        Vector3 com_vel = calc_com_velocity();

        // DCM実測値を計算
        Vector3 dcm_actual = calc_dcm_actual();
        
        control_log.time.push_back(robot->timer.time);
        
        // 実測値
        control_log.com_pos_x.push_back(robot->centroid.com_pos.x());
        control_log.com_pos_y.push_back(robot->centroid.com_pos.y());
        control_log.com_pos_z.push_back(robot->centroid.com_pos.z());
        control_log.com_vel_x.push_back(com_vel.x());
        control_log.com_vel_y.push_back(com_vel.y());
        control_log.com_vel_z.push_back(com_vel.z());
        control_log.zmp_x.push_back(robot->centroid.zmp.x());
        control_log.zmp_y.push_back(robot->centroid.zmp.y());
        control_log.zmp_z.push_back(robot->centroid.zmp.z());
        control_log.dcm_x.push_back(dcm_actual.x());
        control_log.dcm_y.push_back(dcm_actual.y());
        control_log.dcm_z.push_back(dcm_actual.z());
        
        // 目標値
        control_log.com_pos_ref_x.push_back(robot->centroid.com_pos_ref.x());
        control_log.com_pos_ref_y.push_back(robot->centroid.com_pos_ref.y());
        control_log.com_pos_ref_z.push_back(robot->centroid.com_pos_ref.z());
        control_log.zmp_ref_x.push_back(robot->centroid.zmp_ref.x());
        control_log.zmp_ref_y.push_back(robot->centroid.zmp_ref.y());
        control_log.zmp_ref_z.push_back(robot->centroid.zmp_ref.z());
        control_log.dcm_ref_x.push_back(robot->centroid.dcm_ref.x());
        control_log.dcm_ref_y.push_back(robot->centroid.dcm_ref.y());
        control_log.dcm_ref_z.push_back(robot->centroid.dcm_ref.z());
    }

    // ★ グラフ生成（C++のみで完結）
    void generate_plots() {

        if (control_log.time.empty()) {
        std::cout << "⚠️ ログデータが空です" << std::endl;
        return;
    }
        std::cout << "\n📊 グラフ生成中..." << std::endl;

        plt::backend("Agg");  // ★ この1行を最初に追加
        
        // 6パネルのグラフ
        plt::figure_size(1500, 1000);
        
        // --- パネル1: CoM Position ---
        plt::subplot(3, 2, 1);
        plt::named_plot("Actual X", control_log.time, control_log.com_pos_x, "r-");
        plt::named_plot("Ref X", control_log.time, control_log.com_pos_ref_x, "r--");
        plt::named_plot("Actual Y", control_log.time, control_log.com_pos_y, "g-");
        plt::named_plot("Ref Y", control_log.time, control_log.com_pos_ref_y, "g--");
        plt::named_plot("Actual Z", control_log.time, control_log.com_pos_z, "b-");
        plt::named_plot("Ref Z", control_log.time, control_log.com_pos_ref_z, "b--");
        plt::xlabel("Time [s]");
        plt::ylabel("Position [m]");
        plt::title("CoM Position");
        plt::legend();
        plt::grid(true);
        
        // --- パネル2: CoM Velocity ---
        plt::subplot(3, 2, 2);
        plt::named_plot("X", control_log.time, control_log.com_vel_x, "r-");
        plt::named_plot("Y", control_log.time, control_log.com_vel_y, "g-");
        plt::named_plot("Z", control_log.time, control_log.com_vel_z, "b-");
        plt::xlabel("Time [s]");
        plt::ylabel("Velocity [m/s]");
        plt::title("CoM Velocity");
        plt::legend();
        plt::grid(true);
        
        // --- パネル3: ZMP ---
        plt::subplot(3, 2, 3);
        plt::named_plot("Actual X", control_log.time, control_log.zmp_x, "r-");
        plt::named_plot("Ref X", control_log.time, control_log.zmp_ref_x, "r--");
        plt::named_plot("Actual Y", control_log.time, control_log.zmp_y, "g-");
        plt::named_plot("Ref Y", control_log.time, control_log.zmp_ref_y, "g--");
        plt::xlabel("Time [s]");
        plt::ylabel("Position [m]");
        plt::title("ZMP (Zero Moment Point)");
        plt::legend();
        plt::grid(true);
        
        // --- パネル4: DCM ---
        plt::subplot(3, 2, 4);
        plt::named_plot("Actual X", control_log.time, control_log.dcm_x, "r-");
        plt::named_plot("Ref X", control_log.time, control_log.dcm_ref_x, "r--");
        plt::named_plot("Actual Y", control_log.time, control_log.dcm_y, "g-");
        plt::named_plot("Ref Y", control_log.time, control_log.dcm_ref_y, "g--");
        plt::xlabel("Time [s]");
        plt::ylabel("Position [m]");
        plt::title("DCM (Divergent Component of Motion)");
        plt::legend();
        plt::grid(true);
        
        // --- パネル5: CoM Trajectory (XY平面) ---
        plt::subplot(3, 2, 5);
        plt::named_plot("Actual", control_log.com_pos_x, control_log.com_pos_y, "b-");
        plt::named_plot("Ref", control_log.com_pos_ref_x, control_log.com_pos_ref_y, "b--");
        plt::plot({control_log.com_pos_x[0]}, {control_log.com_pos_y[0]}, "go");  // Start
        plt::plot({control_log.com_pos_x.back()}, {control_log.com_pos_y.back()}, "rx");  // End
        plt::xlabel("X [m]");
        plt::ylabel("Y [m]");
        plt::title("CoM Trajectory (Top View)");
        plt::legend();
        plt::grid(true);
        
        // --- パネル6: ZMP & DCM Trajectory (XY平面) ---
        plt::subplot(3, 2, 6);
        plt::named_plot("ZMP Actual", control_log.zmp_x, control_log.zmp_y, "r-");
        plt::named_plot("ZMP Ref", control_log.zmp_ref_x, control_log.zmp_ref_y, "r--");
        plt::named_plot("DCM Actual", control_log.dcm_x, control_log.dcm_y, "b-");
        plt::named_plot("DCM Ref", control_log.dcm_ref_x, control_log.dcm_ref_y, "b--");
        plt::xlabel("X [m]");
        plt::ylabel("Y [m]");
        plt::title("ZMP & DCM Trajectory (Top View)");
        plt::legend();
        plt::grid(true);
        
        // 保存
        plt::save("control_analysis.png");
        std::cout << "✅ グラフ保存完了: control_analysis.png (" 
                  << control_log.time.size() << " サンプル)" << std::endl;
    }
    

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
        //並列環境で変数が共有されてる？

        // ★ CoM速度計算をリセット
        first_step = true;
        prev_com_pos = Vector3(0.0, 0.0, 0.0);
        com_vel_actual = Vector3(0.0, 0.0, 0.0);
    
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
        rl_params.stride_offset = ptr[0];
        rl_params.turn_offset = ptr[1];
        rl_params.spacing_offset = ptr[2];
        rl_params.climb_offset = ptr[3];
        rl_params.duration_offset = ptr[4];

        bool step_completed = false;
        int prev_support_leg = robot->footstep_buffer.steps[0].side;
        //並列環境で変数が共有されてる？
        int step_counter = 0;

        const int MAX_ITERATIONS = 1000;
        const double MIN_HEIGHT = 0.5;
        bool terminated = false;
        bool timeout = false;

        std::vector<py::array_t<unsigned char>> frame_list;

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

                // ★ 毎制御サイクルでログ記録
                log_control_data();
                
            }

            if (terminated) break;
        
            // レンダリング更新（既存コードと同じ）
            if (rendering_enabled && !glfwWindowShouldClose(window)) {
                updateDisplay();

                //ここで直接描画
                try {
                    frame_list.push_back(render());
                } catch (const std::exception& e) {
                    std::cerr << "[WARNING] Frame capture failed: " << e.what() << std::endl;
                }
            }
        
            // 1歩完了を検出
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
        
        return py::make_tuple(obs, reward, terminated, py::dict(),frame_list);
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

            // カメラをロボットに追従させる、必要ないときは消せ
            cam.lookat[0] = d->qpos[0];  // X座標
            cam.lookat[1] = d->qpos[1];  // Y座標
            cam.lookat[2] = d->qpos[2];  // Z座標


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
        .def("should_close", &VnoidEnv::should_close);
}