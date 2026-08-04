#include "vnoid_env.h"

// ★ ログ取得メソッドはCSVファイルから読み込む必要がある場合は削除
// CSVファイルは外部スクリプトで読み込むことを想定
py::dict VnoidEnv::get_control_log() {
    py::dict log;
    // CSVファイルから読み込む機能は実装しない（外部スクリプトで処理）
    // 後方互換性のため空の辞書を返す
    return log;
}

// ★ ログをクリアするメソッド（CSVファイルは削除しない）
void VnoidEnv::clear_control_log() {
    // CSVファイルは削除しない（外部で管理）
}

FootState VnoidEnv::get_foot_state(int bid) {
    FootState fs{};
    if (!m || !d || bid < 0) {
        return fs;
    }
    fs.pos[0] = d->xpos[3 * bid + 0];
    fs.pos[1] = d->xpos[3 * bid + 1];
    fs.pos[2] = d->xpos[3 * bid + 2];
    mj_objectVelocity(m, d, mjOBJ_BODY, bid, fs.vel, 0);
    return fs;
}

// ★ CoM速度を数値微分で計算
Vector3 VnoidEnv::calc_com_velocity() {
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

Vector3 VnoidEnv::calc_dcm_actual(const Vector3& com_vel) {
    // DCM = CoM_pos + T * CoM_vel
    // T = sqrt(h/g) はLIPMの時定数
    double T = robot->param.T;
    return robot->centroid.com_pos + T * com_vel;
    // Note: com_velの実測値がない場合はcom_vel_refを使う
    // より正確にはFK計算からcom_velを取得すべき
}

// ★ 重心周りの角運動量を計算（MuJoCoから取得）
Vector3 VnoidEnv::calc_angular_momentum_around_com() {
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

void VnoidEnv::log_control_data() {
    if (!rendering_enabled || !robot || !csv_opened || logging_completed) return;
    

    // DCM実測値を計算
    Vector3 com_vel = calc_com_velocity();
    Vector3 dcm_actual = calc_dcm_actual(com_vel);
    
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
    // desired moment (in local coordinate)
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
    Vector3 angular_momentum_com_local = robot->base.ori.conjugate() * angular_momentum_com;
    
    // 角運動量の時間微分を計算（モーメント、ローカル座標系）
    Vector3 angular_moment_local = Vector3(0.0, 0.0, 0.0);
    double dt = robot->timer.dt;
    if (!first_log && dt > 1e-6) {
        angular_moment_local = (angular_momentum_com_local - prev_angular_momentum_com) / dt;
    }
    prev_angular_momentum_com = angular_momentum_com_local;
    if (first_log) first_log = false;
    
    // 回復モーメントとの差を計算（ローカル座標系）
    Vector3 moment_diff = angular_moment_local - Ld_local;
    
    // MuJoCo から足位置・速度を直接取得
    FootState mj_foot_r = get_foot_state(bid_r_foot);
    FootState mj_foot_l = get_foot_state(bid_l_foot);

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
             << robot->base.pos.x() << "," << robot->base.pos.y() << "," << robot->base.pos.z() << ","
             << robot->base.pos_ref.x() << "," << robot->base.pos_ref.y() << "," << robot->base.pos_ref.z() << ","
             << robot->base.ori.w() << "," << robot->base.ori.x() << "," << robot->base.ori.y() << "," << robot->base.ori.z() << ","
             << robot->base.ori_ref.w() << "," << robot->base.ori_ref.x() << "," << robot->base.ori_ref.y() << "," << robot->base.ori_ref.z() << ","
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
             << angular_moment_local.x() << "," << angular_moment_local.y() << "," << angular_moment_local.z() << ","
             << moment_diff.x() << "," << moment_diff.y() << "," << moment_diff.z() << ","
             << zmp_local.x() << "," << zmp_local.y() << "," << zmp_local.z() << ","
             << robot->base.ori.w() << "," << robot->base.ori.x() << "," << robot->base.ori.y() << "," << robot->base.ori.z() << ","
             << robot->base.acc[0] << "," << robot->base.acc[1] << "," << robot->base.acc[2] << ","
             << robot->foot[0].pos_ref[2] << "," << robot->foot[1].pos_ref[2] << ","
             << robot->foot[0].pos[0] << "," << robot->foot[0].pos[1] << "," << robot->foot[0].pos[2] << ","
             << robot->foot[1].pos[0] << "," << robot->foot[1].pos[1] << "," << robot->foot[1].pos[2] << ","
             << mj_foot_r.pos[0] << "," << mj_foot_r.pos[1] << "," << mj_foot_r.pos[2] << ","
             << mj_foot_r.vel[0] << "," << mj_foot_r.vel[1] << "," << mj_foot_r.vel[2] << ","
             << mj_foot_r.vel[3] << "," << mj_foot_r.vel[4] << "," << mj_foot_r.vel[5] << ","
             << mj_foot_l.pos[0] << "," << mj_foot_l.pos[1] << "," << mj_foot_l.pos[2] << ","
             << mj_foot_l.vel[0] << "," << mj_foot_l.vel[1] << "," << mj_foot_l.vel[2] << ","
             << mj_foot_l.vel[3] << "," << mj_foot_l.vel[4] << "," << mj_foot_l.vel[5];
    // 脚部関節角度 (i=18..29)
    for (int i = 18; i < 30; i++) {
        csv_file << "," << robot->joint[i].q;
    }
    // 脚部関節角速度 (i=18..29)
    for (int i = 18; i < 30; i++) {
        csv_file << "," << robot->joint[i].dq;
    }
    csv_file << std::endl;
    
    // 定期的にflush（例：100回に1回）
    static int log_count = 0;
    if (++log_count % 100 == 0) {
        csv_file.flush();
    }
}
