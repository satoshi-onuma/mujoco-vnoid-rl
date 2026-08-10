#include "vnoid_env.h"

py::array_t<double> VnoidEnv::reset() {
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
    initFootBodyIds();

    mj_forward(m, d);
    previous_x = d->qpos[0];
    previous_y = d->qpos[1];
    step_start_yaw = robot->base.angle.z();
#if VNOID_REWARD_LOG_DEBUG
    reward_step_index = 0;
#endif
    //並列環境で変数が共有されてる？

    // ★ CoM速度計算をリセット
    first_step = true;
    prev_com_pos = Vector3(0.0, 0.0, 0.0);
    prev_com_pos_for_reward = Vector3(0.0, 0.0, 0.0);
    com_vel_actual = Vector3(0.0, 0.0, 0.0);
    
    // ★ 角運動量の時間微分計算をリセット
    first_log = true;
    prev_angular_momentum_com = Vector3(0.0, 0.0, 0.0);

    // 地盤切り替えリセット：硬地盤から開始し、ランダムなサイクルで軟地盤に切り替える
    apply_hard_terrain();
    std::uniform_int_distribution<int> dist(200,280);
    //One step takes 0.4s when duration = 0.4
    //Therefore one gait cycle takes 0.8s
    //So, The step range must cover 0.8/dt = 0.8/0.01 = 80 steps
    //start from 2s(2/dt=200)
    //200から280までの整数を生成する変換器をインスタンス化
    terrain_switch_at = dist(terrain_rng);
    std::cout << "地盤切り替えステップ数：" << terrain_switch_at << std::endl;
    //生成した乱数をシードとして入れる
    control_cycle_count = 0;

    return get_observation();
}

// Phase3: 外部から歩容コマンドを注入
void VnoidEnv::set_walk_command(double stride, double sway, double turn) {
    if (!robot) return;
    robot->SetWalkCommand(stride, sway, turn);
    robot->UpdateFootstepPlan();
}

py::tuple VnoidEnv::step(py::array_t<double> action) {
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
            control_cycle_count++;
            if (control_cycle_count == terrain_switch_at) {
                // apply_random_terrain();
                // apply_debug_terrain();
                // 切り替え先はPythonから設定可能（既定はsoft）
                apply_switch_terrain();
            }
            mj_step(m, d);
            
            if (control_cycle_count % robot->param.control_cycle == 0) {
                
                // 毎ステップ転倒チェック
                double hips_z = d->qpos[2];
                if (hips_z < MIN_HEIGHT) {
                    terminated = true;
                    break;
                }

                // ★ 毎制御サイクルでログ記録
                log_control_data();
            }
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
        
        //デバッグ用
        //step_counterが100の倍数のときはstep_completedをtrueにする
        //歩いてないけど擬似的に支持脚が変わったことにする
        // step_counter++;
        // if (step_counter % 100 == 0) {
        //     step_completed = true;
        // }else{
        //     step_completed = false;
        // }


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

    // 転倒時にペナルティを設定
    //truncatedはまだまだ行けるけど終了する場合なので報酬はブートストラップする
    //terminatedは終了する場合なので報酬はブートストラップしない
    //terminated使いたいけどtimeout使わないと負の報酬はできない
    if (timeout) {
        reward -= 10.0;
    }
    
    return py::make_tuple(obs, reward, terminated, py::dict(),frame_list);
}
