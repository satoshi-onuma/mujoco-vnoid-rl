#include "vnoid_env.h"

VnoidEnv::VnoidEnv(const std::string& model_path, bool enable_rendering)
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
    initFootBodyIds();
    
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
                     << "base_pos_x,base_pos_y,base_pos_z,"
                     << "base_pos_ref_x,base_pos_ref_y,base_pos_ref_z,"
                     << "base_ori_w,base_ori_x,base_ori_y,base_ori_z,"
                     << "base_ori_ref_w,base_ori_ref_x,base_ori_ref_y,base_ori_ref_z,"
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
                     << "obs_foot_height_right,obs_foot_height_left,"
                     << "foot_pos_right_x,foot_pos_right_y,foot_pos_right_z,"
                     << "foot_pos_left_x,foot_pos_left_y,foot_pos_left_z,"
                     << "mj_foot_r_pos_x,mj_foot_r_pos_y,mj_foot_r_pos_z,"
                     << "mj_foot_r_linvel_x,mj_foot_r_linvel_y,mj_foot_r_linvel_z,"
                     << "mj_foot_r_angvel_x,mj_foot_r_angvel_y,mj_foot_r_angvel_z,"
                     << "mj_foot_l_pos_x,mj_foot_l_pos_y,mj_foot_l_pos_z,"
                     << "mj_foot_l_linvel_x,mj_foot_l_linvel_y,mj_foot_l_linvel_z,"
                     << "mj_foot_l_angvel_x,mj_foot_l_angvel_y,mj_foot_l_angvel_z,"
                     << "joint_R_UPPERLEG_Y_q,joint_R_UPPERLEG_R_q,joint_R_UPPERLEG_P_q,"
                     << "joint_R_LOWERLEG_P_q,joint_R_FOOT_P_q,joint_R_FOOT_R_q,"
                     << "joint_L_UPPERLEG_Y_q,joint_L_UPPERLEG_R_q,joint_L_UPPERLEG_P_q,"
                     << "joint_L_LOWERLEG_P_q,joint_L_FOOT_P_q,joint_L_FOOT_R_q,"
                     << "joint_R_UPPERLEG_Y_dq,joint_R_UPPERLEG_R_dq,joint_R_UPPERLEG_P_dq,"
                     << "joint_R_LOWERLEG_P_dq,joint_R_FOOT_P_dq,joint_R_FOOT_R_dq,"
                     << "joint_L_UPPERLEG_Y_dq,joint_L_UPPERLEG_R_dq,joint_L_UPPERLEG_P_dq,"
                     << "joint_L_LOWERLEG_P_dq,joint_L_FOOT_P_dq,joint_L_FOOT_R_dq"
                     << std::endl;
            
            csv_file.flush();  // 即座に書き込む
        }

#if VNOID_REWARD_LOG_DEBUG
        // reward_tracking 検証用（1歩1行）→ reward_log.csv
        reward_csv_file.open("reward_log.csv");
        if (!reward_csv_file.is_open()) {
            std::cerr << "⚠️ reward_log.csv を開けませんでした" << std::endl;
            reward_csv_opened = false;
        } else {
            reward_csv_opened = true;
            reward_step_index = 0;
            std::cout << "📝 reward_log.csv ロギング開始" << std::endl;
            reward_csv_file
                << "step,time,"
                << "dx,dy,ex_disp,ey_disp,ex,ey,"
                << "tracking,action_penalty,healthy,total,"
                << "cmd_stride,cmd_sway,cmd_turn,"
                << "step_start_yaw,base_yaw,"
                << "previous_x,previous_y,current_x,current_y"
                << std::endl;
            reward_csv_file.flush();
        }
#endif

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
    previous_y = d->qpos[1];
    step_start_yaw = robot->base.angle.z();

    prev_com_pos = Vector3(0.0, 0.0, 0.0);
    prev_com_pos_for_reward = Vector3(0.0, 0.0, 0.0);
    com_vel_actual = Vector3(0.0, 0.0, 0.0);
    first_step = true;
    
    std::cout << "✅ VnoidEnv初期化完了" << std::endl;
}

VnoidEnv::~VnoidEnv() {
    // CSVファイルを閉じる
    if (csv_opened) {
        csv_file.close();
        std::cout << "✅ CSVファイル保存完了: control_log.csv" << std::endl;
    }
#if VNOID_REWARD_LOG_DEBUG
    if (reward_csv_opened) {
        reward_csv_file.close();
        std::cout << "✅ reward_log.csv 保存完了 (" << reward_step_index << " steps)" << std::endl;
    }
#endif
    
    cleanup();
}

void VnoidEnv::initializeRobot() {
    robot = std::make_unique<MyRobot>();
    robot->Init(m, d);
    initFootBodyIds();
}

void VnoidEnv::initFootBodyIds() {
    bid_r_foot = mj_name2id(m, mjOBJ_BODY, "R_FOOT_R");
    bid_l_foot = mj_name2id(m, mjOBJ_BODY, "L_FOOT_R");
    if (bid_r_foot < 0 || bid_l_foot < 0) {
        std::cerr << "[WARNING] Foot body ID not found: R_FOOT_R=" << bid_r_foot
                  << " L_FOOT_R=" << bid_l_foot << std::endl;
    }
}

void VnoidEnv::cleanup() {
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
