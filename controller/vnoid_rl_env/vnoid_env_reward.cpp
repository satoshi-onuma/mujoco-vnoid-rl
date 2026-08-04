#include "vnoid_env.h"

void VnoidEnv::set_reward_weights(double w_track_, double w_act_,
                                  double w_healthy_, double tracking_sigma_) {
    w_track = w_track_;
    w_act = w_act_;
    w_healthy = w_healthy_;
    tracking_sigma = tracking_sigma_;
    std::cout << "報酬重み変更: w_track=" << w_track
              << " w_act=" << w_act
              << " w_healthy=" << w_healthy
              << " tracking_sigma=" << tracking_sigma << std::endl;
}

double VnoidEnv::reward_tracking() {
    if (!robot) return 0.0;
    const double dx = d->qpos[0] - previous_x;
    const double dy = d->qpos[1] - previous_y;
    // 歩開始時点の胴体yaw（両足の中間角）で2D回転。roll/pitchは無視
    const double c = std::cos(-step_start_yaw);
    const double s = std::sin(-step_start_yaw);
    const double ex_disp = c*dx - s*dy;
    const double ey_disp = s*dx + c*dy;
    const double ex = robot->walk_cmd.stride - ex_disp;
    const double ey = robot->walk_cmd.sway   - ey_disp;
    return std::exp(-(ex*ex + ey*ey) / tracking_sigma);
}

double VnoidEnv::reward_action_penalty() {
    const double a0 = last_rl_action[0] / 0.15; //[-0.15 0.15]→[-1 1]
    const double a1 = last_rl_action[1] / 0.1; //[-0.1 0.1]→[-1 1]
    return a0*a0 + a1*a1;  //[0 2]
            //現状あまりにも介入に対するペナルティが低すぎる

}

double VnoidEnv::reward_healthy() {
    return 1.0;
}

#if VNOID_REWARD_LOG_DEBUG
void VnoidEnv::log_reward_step(double dx, double dy, double ex_disp, double ey_disp,
                     double ex, double ey, double tracking,
                     double action_penalty, double healthy, double total) {
    if (!reward_csv_opened) return;
    const double time = robot ? robot->timer.time : 0.0;
    reward_csv_file
        << reward_step_index << ","
        << time << ","
        << dx << "," << dy << ","
        << ex_disp << "," << ey_disp << ","
        << ex << "," << ey << ","
        << tracking << "," << action_penalty << "," << healthy << "," << total << ","
        << robot->walk_cmd.stride << "," << robot->walk_cmd.sway << "," << robot->walk_cmd.turn << ","
        << step_start_yaw << "," << robot->base.angle.z() << ","
        << previous_x << "," << previous_y << ","
        << d->qpos[0] << "," << d->qpos[1]
        << std::endl;
    reward_csv_file.flush();
    reward_step_index++;
}
#endif

py::array_t<double> VnoidEnv::get_observation() {
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
    


    // command（目標）を観測に含める（Phase3: ランダム指令のマルコフ性確保）
    obs.push_back(robot->walk_cmd.stride);
    obs.push_back(robot->walk_cmd.sway);
    obs.push_back(robot->walk_cmd.turn);

    return py::array_t<double>(obs.size(), obs.data());
}

double VnoidEnv::compute_reward() {
    if (!robot) {
        return 0.0;
    }

    double r;
#if VNOID_REWARD_LOG_DEBUG
    const double dx = d->qpos[0] - previous_x;
    const double dy = d->qpos[1] - previous_y;
    const double c = std::cos(-step_start_yaw);
    const double s = std::sin(-step_start_yaw);
    const double ex_disp = c*dx - s*dy;
    const double ey_disp = s*dx + c*dy;
    const double ex = robot->walk_cmd.stride - ex_disp;
    const double ey = robot->walk_cmd.sway   - ey_disp;
    const double tracking = std::exp(-(ex*ex + ey*ey) / tracking_sigma);
    const double action_penalty = reward_action_penalty();
    const double healthy = reward_healthy();
    r =
        // w_track   * tracking
       w_healthy * healthy
      - w_act     * action_penalty;
    log_reward_step(dx, dy, ex_disp, ey_disp, ex, ey, tracking,
                    action_penalty, healthy, r);
#else
    r =
        w_track   * reward_tracking()
      + w_healthy * reward_healthy()
      - w_act     * reward_action_penalty();
#endif

    // 次回のために現在の位置と次歩用の基準yawを保存
    previous_x = d->qpos[0];
    previous_y = d->qpos[1];
    step_start_yaw = robot->base.angle.z();
    return r;
}
