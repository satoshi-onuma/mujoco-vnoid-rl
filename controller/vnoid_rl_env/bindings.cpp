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
            // CSVファイルを開く
            std::string csv_filename = "control_log.csv";
            csv_file.open(csv_filename);
            
            if (!csv_file.is_open()) {
                std::cerr << "⚠️ CSVファイルを開けませんでした: " << csv_filename << std::endl;
                csv_opened = false;
            } else {
                csv_opened = true;
                std::cout << "📝 CSVロギング開始: " << csv_filename << std::endl;
                
                // ヘッダー行を書き込む
                csv_file << "time,"
                         << "com_pos_x,com_pos_y,com_pos_z,"
                         << "com_pos_ref_x,com_pos_ref_y,com_pos_ref_z,"
                         << "com_vel_x,com_vel_y,com_vel_z,"
                         << "zmp_ref_x,zmp_ref_y,"
                         << "dcm_x,dcm_y,dcm_z,"
                         << "dcm_ref_x,dcm_ref_y,dcm_ref_z,"
                         << "dcm_offset_actual_x,dcm_offset_actual_y,"
                         << "dcm_offset_desired_x,dcm_offset_desired_y,"
                         << "support_foot_actual_x,support_foot_actual_y,"
                         << "next_step_dcm_x,next_step_dcm_y,"
                         << "next_step_support_foot_x,next_step_support_foot_y,"
                         << "land_dcm_x,land_dcm_y,land_dcm_z,"
                         << "obs_angvel_x,obs_angvel_y,obs_angvel_z,"
                         << "obs_foot_sink_right,obs_foot_sink_left,"
                         << "obs_contact_right,obs_contact_left,"
                         << "rl_action_foot_offset_x,rl_action_foot_offset_y,"
                         << "base_angle_roll,base_angle_pitch,base_angle_yaw,"
                         << "base_angle_ref_roll,base_angle_ref_pitch,base_angle_ref_yaw,"
                         << "recovery_moment_desired_x,recovery_moment_desired_y,recovery_moment_desired_z,"
                         << "angular_moment_x,angular_moment_y,angular_moment_z,"
                         << "moment_diff_x,moment_diff_y,moment_diff_z,"
                         << "zmp_local_x,zmp_local_y,zmp_local_z,"
                         << "obs_ori_w,obs_ori_x,obs_ori_y,obs_ori_z,"
                         << "obs_acc_x,obs_acc_y,obs_acc_z,"
                         << "obs_foot_height_right,obs_foot_height_left"
                         << std::endl;
                
                csv_file.flush();  // 即座に書き込む
            }

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
        prev_com_pos_for_reward = Vector3(0.0, 0.0, 0.0);
        com_vel_actual = Vector3(0.0, 0.0, 0.0);
        first_step = true;
        
        std::cout << "✅ VnoidEnv初期化完了" << std::endl;
    }

    ~VnoidEnv() {
        // CSVファイルを閉じる
        if (csv_opened) {
            csv_file.close();
            std::cout << "✅ CSVファイル保存完了: control_log.csv" << std::endl;
        }
        
        cleanup();
    }


    // ★ ログ取得メソッドはCSVファイルから読み込む必要がある場合は削除
    // CSVファイルは外部スクリプトで読み込むことを想定
    py::dict get_control_log() {
        py::dict log;
        // CSVファイルから読み込む機能は実装しない（外部スクリプトで処理）
        // 後方互換性のため空の辞書を返す
        return log;
    }
    
    // ★ ログをクリアするメソッド（CSVファイルは削除しない）
    void clear_control_log() {
        // CSVファイルは削除しない（外部で管理）
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

        double right_foot_sink = robot->foot[0].pos[2] - robot->foot[0].pos_ref[2];  // 実測-予測　予想より下にいたら-,上にいたら+
        double left_foot_sink = robot->foot[1].pos[2] - robot->foot[1].pos_ref[2];   // 実測足の位置使っているのでここだけジャイロ以外使用

        obs.push_back(right_foot_sink); 
        obs.push_back(left_foot_sink); 

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
        if (!robot) {
            return 0.0;
        }
        
        // 現在のCoM位置を取得
        Vector3 current_com_pos = robot->centroid.com_pos;
        
        // 初回は前方向報酬を0にする
        double forward_reward = 0.0;
        if (prev_com_pos_for_reward.norm() > 1e-6) {  // 前回の位置が有効な場合
            // 絶対座標での移動ベクトル
            Vector3 delta_global = current_com_pos - prev_com_pos_for_reward;
            
            // ロボットのローカル座標系に変換（クォータニオンの逆変換）
            // ローカル座標系での前方向はMuJoCoではx方向
            Vector3 delta_local = robot->base.ori.conjugate() * delta_global;
            
            // ローカル座標系での前方向（x方向）の成分を取得
            forward_reward = delta_local.x();
        }
        
        /*
         double current_x = d->qpos[0];
        double forward_reward = current_x - previous_x;
        */
        double healthy_reward = 1.0;
        double total_reward = forward_reward * 5.0 + healthy_reward;
        /*
         // ★デバッグログ
        std::cout << "[REWARD] forward_reward=" << forward_reward 
                 << " | total=" << total_reward << std::endl;
        */
        //ここで返してるトータルリワードとPython側で受け取ってるRewardの値が若干違う
        //並列環境減らすなどして試す
        //現時点では50stepで終わってないほうが凶悪
    
        // 次回のために現在の位置を保存
        prev_com_pos_for_reward = current_com_pos;
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

    // ★ 重心周りの角運動量を計算（MuJoCoから取得）
    Vector3 calc_angular_momentum_around_com() {
        if (!m || !d || !robot) {
            return Vector3(0.0, 0.0, 0.0);
        }
        mj_subtreeVel(m, d);
        // MuJoCoのsubtree_angmomを使用
        // ルートボディ（body ID 0）のsubtree_angmomは全体の角運動量
        // または、cinert[0]の角運動量成分（3:6）を使用
        // subtree_angmomは重心周りの角運動量を返す
        Vector3 angular_momentum;
        int hips_body_id = 1;  // XMLの<body name="HIPS">に対応
        
        // 方法1: subtree_angmomを使用（推奨）
        // ルートボディ（body ID 0）のサブツリー角運動量 = 全体の角運動量
        if (m->nbody > hips_body_id && d->subtree_angmom) {
            // subtree_angmomは各ボディのサブツリー角運動量（3次元ベクトル）
            // ルートボディ（ID 0）のサブツリーが全体なので、それを使用
            angular_momentum = Vector3(
                d->subtree_angmom[3*hips_body_id],
                d->subtree_angmom[3*hips_body_id+1],
                d->subtree_angmom[3*hips_body_id+2]
            );

            /*
            // デバッグ出力（最初の数回だけ）
        static int debug_count = 0;
        if (debug_count++ < 5) {
            std::cout << "Angular momentum [" << debug_count << "]: ["
                      << angular_momentum.x() << ", "
                      << angular_momentum.y() << ", "
                      << angular_momentum.z() << "]" << std::endl;
            */
        }
        else {
            angular_momentum = Vector3(0.0, 0.0, 0.0);
        }
        
        return angular_momentum;
    }

    void log_control_data() {
        if (!rendering_enabled || !robot || !csv_opened || logging_completed) return;
        
        // CoM速度を計算
        Vector3 com_vel = calc_com_velocity();

        // DCM実測値を計算
        Vector3 dcm_actual = calc_dcm_actual();
        
        double time = robot->timer.time;
        
        // DCM Offset計算
        int sup = robot->footstep_buffer.steps[0].side;
        double support_foot_actual_x = robot->foot[sup].pos[0];
        double support_foot_actual_y = robot->foot[sup].pos[1];
        double dcm_offset_actual_x = dcm_actual.x() - support_foot_actual_x;
        double dcm_offset_actual_y = dcm_actual.y() - support_foot_actual_y;
        
        double dcm_offset_desired_x = 0.0;
        double dcm_offset_desired_y = 0.0;
        double next_step_dcm_x = 0.0;
        double next_step_dcm_y = 0.0;
        double next_step_support_foot_x = 0.0;
        double next_step_support_foot_y = 0.0;
        if (robot->footstep.steps.size() >= 2) {
            const Step& st1 = robot->footstep.steps[1];
            int swg = !st1.side;
            next_step_dcm_x = st1.dcm.x();
            next_step_dcm_y = st1.dcm.y();
            next_step_support_foot_x = st1.foot_pos[swg].x();
            next_step_support_foot_y = st1.foot_pos[swg].y();
            dcm_offset_desired_x = next_step_dcm_x - next_step_support_foot_x;
            dcm_offset_desired_y = next_step_dcm_y - next_step_support_foot_y;
        }
        
        // 観測値
        double right_foot_sink = robot->foot[0].pos[2] - robot->foot[0].pos_ref[2];
        double left_foot_sink = robot->foot[1].pos[2] - robot->foot[1].pos_ref[2];
        
        // 回復モーメントの所望量を計算（stabilizer.cppのCalcDcmDynamicsと同じ計算）
        Vector3 theta = robot->base.angle - robot->base.angle_ref;
        Vector3 omega = robot->base.angvel - robot->base.angvel_ref;
        Vector3 omegadd_local(
            -(robot->stabilizer.orientation_ctrl_gain_p*theta.x() + robot->stabilizer.orientation_ctrl_gain_d*omega.x()),
            -(robot->stabilizer.orientation_ctrl_gain_p*theta.y() + robot->stabilizer.orientation_ctrl_gain_d*omega.y()),
            0.0
        );
        Vector3 Ld_local(
            robot->param.nominal_inertia.x()*omegadd_local.x(),
            robot->param.nominal_inertia.y()*omegadd_local.y(),
            robot->param.nominal_inertia.z()*omegadd_local.z()
        );
        // limit recovery moment for safety
        for(int i = 0; i < 3; i++){
            Ld_local[i] = std::min(std::max(-robot->stabilizer.recovery_moment_limit, Ld_local[i]), robot->stabilizer.recovery_moment_limit);
        }
        
        // 重心周りの角運動量を計算（MuJoCoから取得）
        Vector3 angular_momentum_com = calc_angular_momentum_around_com();
        
        // 角運動量の時間微分を計算（モーメント）
        Vector3 angular_moment = Vector3(0.0, 0.0, 0.0);
        double dt = robot->timer.dt;
        if (!first_log && dt > 1e-6) {
            angular_moment = (angular_momentum_com - prev_angular_momentum_com) / dt;
        }
        prev_angular_momentum_com = angular_momentum_com;
        if (first_log) first_log = false;
        
        // 回復モーメントとの差を計算（ローカル座標系）
        Vector3 moment_diff = angular_moment - Ld_local;
        
        // zmp_localを計算（単脚支持時のみ）
        Vector3 zmp_local = Vector3(0.0, 0.0, 0.0);
        if ((robot->foot[0].contact_ref && !robot->foot[1].contact_ref) ||
            (!robot->foot[0].contact_ref && robot->foot[1].contact_ref)) {
            int sup = robot->foot[0].contact_ref ? 0 : 1;
            zmp_local = robot->foot[sup].ori_ref.conjugate() * (robot->centroid.zmp_ref - robot->foot[sup].pos_ref);
        } else if (robot->foot[0].contact_ref && robot->foot[1].contact_ref) {
            // 両脚支持時は、より接触力が大きい方の足のzmp_localを使用
            // 簡易的に左足のzmp_localを使用（foot[0].zmp_refは既にローカル座標系）
            zmp_local = robot->foot[0].zmp_ref;
        }
        
        // CSVに1行書き込む
        csv_file << time << ","
                 << robot->centroid.com_pos.x() << "," << robot->centroid.com_pos.y() << "," << robot->centroid.com_pos.z() << ","
                 << robot->centroid.com_pos_ref.x() << "," << robot->centroid.com_pos_ref.y() << "," << robot->centroid.com_pos_ref.z() << ","
                 << com_vel.x() << "," << com_vel.y() << "," << com_vel.z() << ","
                 << robot->centroid.zmp_ref.x() << "," << robot->centroid.zmp_ref.y() << ","
                 << dcm_actual.x() << "," << dcm_actual.y() << "," << dcm_actual.z() << ","
                 << robot->centroid.dcm_ref.x() << "," << robot->centroid.dcm_ref.y() << "," << robot->centroid.dcm_ref.z() << ","
                 << dcm_offset_actual_x << "," << dcm_offset_actual_y << ","
                 << dcm_offset_desired_x << "," << dcm_offset_desired_y << ","
                 << support_foot_actual_x << "," << support_foot_actual_y << ","
                 << next_step_dcm_x << "," << next_step_dcm_y << ","
                 << next_step_support_foot_x << "," << next_step_support_foot_y << ","
                 << robot->stepping_controller.land_dcm.x() << "," << robot->stepping_controller.land_dcm.y() << "," << robot->stepping_controller.land_dcm.z() << ","
                 << robot->base.angvel[0] << "," << robot->base.angvel[1] << "," << robot->base.angvel[2] << ","
                 << right_foot_sink << "," << left_foot_sink << ","
                 << (robot->foot[0].contact_ref ? 1.0 : 0.0) << "," << (robot->foot[1].contact_ref ? 1.0 : 0.0) << ","
                 << last_rl_action[0] << "," << last_rl_action[1] << ","
                 << robot->base.angle.x() << "," << robot->base.angle.y() << "," << robot->base.angle.z() << ","
                 << robot->base.angle_ref.x() << "," << robot->base.angle_ref.y() << "," << robot->base.angle_ref.z() << ","
                 << Ld_local.x() << "," << Ld_local.y() << "," << Ld_local.z() << ","
                 << angular_moment.x() << "," << angular_moment.y() << "," << angular_moment.z() << ","
                 << moment_diff.x() << "," << moment_diff.y() << "," << moment_diff.z() << ","
                 << zmp_local.x() << "," << zmp_local.y() << "," << zmp_local.z() << ","
                 << robot->base.ori.w() << "," << robot->base.ori.x() << "," << robot->base.ori.y() << "," << robot->base.ori.z() << ","
                 << robot->base.acc[0] << "," << robot->base.acc[1] << "," << robot->base.acc[2] << ","
                 << robot->foot[0].pos_ref[2] << "," << robot->foot[1].pos_ref[2]
                 << std::endl;
        
        // 定期的にflush（例：100回に1回）
        static int log_count = 0;
        if (++log_count % 100 == 0) {
            csv_file.flush();
        }
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

        // 最初のエピソードが終わったらロギングを停止してCSVファイルを閉じる
        // reset_count == 1 なら、これは2回目のreset（最初のエピソード終了後）
        reset_count++;
        if (!logging_completed && csv_opened && reset_count == 2) {
            logging_completed = true;
            csv_file.close();
            csv_opened = false;
            std::cout << "✅ 最初のエピソード完了、CSVロギング停止・ファイル保存完了: control_log.csv" << std::endl;
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
        prev_com_pos_for_reward = Vector3(0.0, 0.0, 0.0);
        com_vel_actual = Vector3(0.0, 0.0, 0.0);
        
        // ★ 角運動量の時間微分計算をリセット
        first_log = true;
        prev_angular_momentum_com = Vector3(0.0, 0.0, 0.0);
    
        return get_observation();
    }

    py::tuple step(py::array_t<double> action) {
        if (!initialized) {
            throw std::runtime_error("環境が初期化されていません。");
        }
        /**/
        auto buf = action.request();
        if (buf.ndim != 1 || buf.size < 2) {
            throw std::runtime_error("アクションの次元またはサイズが不正です。");
        }
        
        double* ptr = static_cast<double*>(buf.ptr);
        
        // RLアクションを保存
        last_rl_action[0] = ptr[0];
        last_rl_action[1] = ptr[1];
        
        RLParams rl_params;
         
        // actionから設定
        rl_params.foot_offset.x()= ptr[0];
        rl_params.foot_offset.y() = ptr[1];
        //rl_params.spacing_offset = ptr[2];
        //rl_params.climb_offset = ptr[3];
        //rl_params.duration_offset = ptr[4];

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

            step_counter++;

            if (step_counter  >= MAX_ITERATIONS) {
                terminated = true;
                timeout = true;  // タイムアウトフラグをセット
                std::cerr << "[WARNING] Step timeout" << std::endl;
                break;
            }
        }

        if (step_completed && !terminated) {
            //std::cout << "[INFO] Step completed after " << step_counter << " iterations" << std::endl;
            //24*1/60/0.001=400step/歩　これはdurationの0.4に等しい
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
        .def("should_close", &VnoidEnv::should_close)
        .def("get_control_log", &VnoidEnv::get_control_log)  // ★ 追加
        .def("clear_control_log", &VnoidEnv::clear_control_log);  // ★ 追加
}