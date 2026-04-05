/**
 * @file main.cpp
 * @brief Choreonoid SimpleController — 強化学習エージェント向け End-to-End 制御
 *
 * ## 役割
 *   Choreonoid のシミュレーションループ (control()) を Python 側の RL エージェントと
 *   POSIX 共有メモリ (ShmInterface) 経由で同期させる。
 *
 * ## 制御の流れ (1 ステップ)
 *   1. Python が行動 (29 次元, 正規化 [-1, 1]) を ActionBlock に書き込む
 *   2. Python が step_request = 1 をセット
 *   3. control() が step_request を検知
 *   4. 正規化行動 → 実際の関節角度に変換 (関節可動域でスケール)
 *   5. PD 制御でトルクを計算・適用
 *   6. 観測ベクトル (71 次元) を収集して StateBlock に書き込む
 *   7. 報酬・終了フラグを計算して ControlBlock に書き込む
 *   8. set_ready(true) で Python に完了を通知 (メモリバリア付き)
 *
 * ## PD 制御パラメータ
 *   全関節共通: Kp = 200.0, Kd = 20.0
 *   必要に応じて関節ごとに調整すること。
 *
 * ## 報酬関数 (compute_reward)
 *   + 前進速度 × 1.0
 *   + 直立度 (Z 方向 cos) × 0.5
 *   + 生存ボーナス 0.1
 *   − エネルギー消費 (|τ × dq| の総和) × 0.0001
 *
 * ## 終了条件 (check_termination)
 *   - ルートリンク高さ < 0.3 m  (転倒)
 *   - 直立度 < 0.5              (過度な傾き)
 */

#include <cnoid/SimpleController>
#include <cnoid/Body>
#include <vector>
#include <iostream>
#include <cmath>
#include "shm_interface.h"

using namespace cnoid;
using namespace vnoid_rl;

class VnoidRLController : public SimpleController {
private:
    BodyPtr ioBody;
    double  dt;
    int     num_joints;

    std::unique_ptr<ShmInterface> shm;

    std::vector<Link*>  joints;
    std::vector<double> initial_q;    ///< 初期関節角度 (reset 時に使用)
    std::vector<double> q_prev;       ///< 前ステップの関節角度 (速度計算用、現在未使用)
    std::vector<double> joint_lower;  ///< 関節可動域下限 [rad]
    std::vector<double> joint_upper;  ///< 関節可動域上限 [rad]

    std::vector<double> kp;  ///< PD 比例ゲイン (関節別)
    std::vector<double> kd;  ///< PD 微分ゲイン (関節別)

    Vector3 initial_base_pos;  ///< 初期ルートリンク位置
    Matrix3 initial_base_rot;  ///< 初期ルートリンク姿勢

    int     step_count;
    Vector3 prev_base_pos;  ///< 前ステップのルートリンク位置 (報酬計算用)

public:
    virtual bool initialize(SimpleControllerIO* io) override {
        ioBody     = io->body();
        dt         = io->timeStep();
        num_joints = ioBody->numJoints();

        std::cout << "[VnoidRLController] Initializing..." << std::endl;
        std::cout << "  Number of joints: " << num_joints << std::endl;
        std::cout << "  Time step: "        << dt         << std::endl;

        int env_id = ShmInterface::get_env_id();
        std::cout << "  Environment ID: " << env_id << std::endl;

        // Python 側が先に共有メモリを作成してから Choreonoid を起動する想定のため
        // create=false で既存の共有メモリを開く。
        try {
            shm = std::make_unique<ShmInterface>(env_id, false);
            std::cout << "  Shared memory opened successfully" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "  ERROR: Failed to open shared memory: " << e.what() << std::endl;
            return false;
        }

        // 関節情報を収集し、トルク制御モードを設定する
        joints.resize(num_joints);
        initial_q.resize(num_joints);
        q_prev.resize(num_joints);
        joint_lower.resize(num_joints);
        joint_upper.resize(num_joints);
        kp.resize(num_joints);
        kd.resize(num_joints);

        for (int i = 0; i < num_joints; ++i) {
            joints[i] = ioBody->joint(i);
            joints[i]->setActuationMode(Link::JointTorque);
            io->enableIO(joints[i]);

            initial_q[i]   = joints[i]->q();
            q_prev[i]      = initial_q[i];
            joint_lower[i] = joints[i]->q_lower();
            joint_upper[i] = joints[i]->q_upper();

            kp[i] = 200.0;
            kd[i] = 20.0;
        }

        initial_base_pos = ioBody->rootLink()->p();
        initial_base_rot = ioBody->rootLink()->R();
        prev_base_pos    = initial_base_pos;
        step_count       = 0;

        shm->set_ready(true);
        shm->set_done(false);

        std::cout << "[VnoidRLController] Initialization complete!" << std::endl;
        return true;
    }

    virtual bool control() override {
        // reset 要求を優先処理する
        if (shm->has_reset_request()) {
            reset_robot();
            shm->clear_reset_request();
            shm->set_ready(true);
            return true;
        }

        if (shm->has_step_request()) {
            apply_action();
            collect_observation();

            float reward = compute_reward();
            shm->set_reward(reward);

            bool done = check_termination();
            shm->set_done(done);

            // 全書き込みが完了してから ready=1 をセット (メモリバリア内蔵)
            shm->clear_step_request();
            shm->set_ready(true);

            step_count++;

            if (done)
                std::cout << "[VnoidRLController] Episode terminated at step " << step_count << std::endl;
        }

        return true;
    }

private:
    // -----------------------------------------------------------------------
    // リセット
    // -----------------------------------------------------------------------

    void reset_robot() {
        std::cout << "[VnoidRLController] Resetting robot..." << std::endl;

        ioBody->rootLink()->p() = initial_base_pos;
        ioBody->rootLink()->R() = initial_base_rot;
        ioBody->rootLink()->v().setZero();
        ioBody->rootLink()->w().setZero();

        for (int i = 0; i < num_joints; ++i) {
            joints[i]->q()  = initial_q[i];
            joints[i]->dq() = 0.0;
            q_prev[i]       = initial_q[i];
        }

        prev_base_pos = initial_base_pos;
        step_count    = 0;

        shm->set_done(false);
        shm->set_reward(0.0f);
    }

    // -----------------------------------------------------------------------
    // 行動の適用
    // -----------------------------------------------------------------------

    /**
     * @brief ActionBlock から正規化行動を読み取り、PD 制御でトルクを適用する。
     *
     * 正規化行動 a ∈ [-1, 1] を実際の関節角度に変換する式:
     *   q_target = q_lower + (a + 1) / 2 × (q_upper - q_lower)
     */
    void apply_action() {
        ActionBlock& action = shm->action();

        std::vector<double> target_q(num_joints);
        for (int i = 0; i < num_joints; ++i) {
            double a     = std::clamp(action.joint_targets[i], -1.0, 1.0);
            double range = joint_upper[i] - joint_lower[i];
            target_q[i]  = joint_lower[i] + (a + 1.0) * 0.5 * range;
        }

        for (int i = 0; i < num_joints; ++i) {
            double q     = joints[i]->q();
            double dq    = joints[i]->dq();
            double error = target_q[i] - q;
            joints[i]->u() = kp[i] * error - kd[i] * dq;
        }
    }

    // -----------------------------------------------------------------------
    // 観測収集
    // -----------------------------------------------------------------------

    /**
     * @brief ロボット状態を観測ベクトルとして StateBlock に書き込む。
     *
     * インデックス割り当て:
     *   [ 0- 2] base_pos    (x, y, z)
     *   [ 3- 6] ori_quat    (w, x, y, z)
     *   [ 7- 9] base_vel    (x, y, z)
     *   [10-12] base_angvel (x, y, z)
     *   [13-41] joint_q     (29 joints)
     *   [42-70] joint_dq    (29 joints)
     *   合計 71 次元
     */
    void collect_observation() {
        StateBlock& state = shm->state();
        int idx = 0;

        Vector3 base_pos = ioBody->rootLink()->p();
        state.observations[idx++] = base_pos.x();
        state.observations[idx++] = base_pos.y();
        state.observations[idx++] = base_pos.z();

        Matrix3    base_rot = ioBody->rootLink()->R();
        Quaternion q(base_rot);
        state.observations[idx++] = q.w();
        state.observations[idx++] = q.x();
        state.observations[idx++] = q.y();
        state.observations[idx++] = q.z();

        Vector3 base_vel = ioBody->rootLink()->v();
        state.observations[idx++] = base_vel.x();
        state.observations[idx++] = base_vel.y();
        state.observations[idx++] = base_vel.z();

        Vector3 base_angvel = ioBody->rootLink()->w();
        state.observations[idx++] = base_angvel.x();
        state.observations[idx++] = base_angvel.y();
        state.observations[idx++] = base_angvel.z();

        for (int i = 0; i < num_joints; ++i)
            state.observations[idx++] = joints[i]->q();

        for (int i = 0; i < num_joints; ++i)
            state.observations[idx++] = joints[i]->dq();

        // idx == NUM_OBSERVATIONS (71) のはず
    }

    // -----------------------------------------------------------------------
    // 報酬計算
    // -----------------------------------------------------------------------

    /**
     * @brief 簡易報酬関数。
     *
     *   + 前進速度報酬  : Δx / dt × 1.0
     *   + 直立維持報酬  : R.col(2).z() × 0.5  (Z 方向が上を向くほど大きい)
     *   + 生存ボーナス  : 0.1 (毎ステップ)
     *   − エネルギーペナルティ: Σ|τ_i × dq_i| × 0.0001
     */
    float compute_reward() {
        float reward = 0.0f;

        Vector3 base_pos    = ioBody->rootLink()->p();
        double  forward_vel = (base_pos.x() - prev_base_pos.x()) / dt;
        reward += static_cast<float>(forward_vel) * 1.0f;

        Matrix3 base_rot = ioBody->rootLink()->R();
        double  upright  = base_rot.col(2).z();
        reward += static_cast<float>(upright) * 0.5f;

        reward += 0.1f;

        double energy = 0.0;
        for (int i = 0; i < num_joints; ++i)
            energy += std::abs(joints[i]->u() * joints[i]->dq());
        reward -= static_cast<float>(energy) * 0.0001f;

        prev_base_pos = base_pos;

        return reward;
    }

    // -----------------------------------------------------------------------
    // 終了条件
    // -----------------------------------------------------------------------

    /**
     * @brief 転倒判定。
     * @return true  : エピソード終了 (転倒)
     * @return false : 継続
     *
     * 条件:
     *   - ルートリンク高さ < 0.3 m
     *   - ルートリンクの Z 軸 (直立方向) の Z 成分 < 0.5
     */
    bool check_termination() {
        Vector3 base_pos = ioBody->rootLink()->p();
        if (base_pos.z() < 0.3)
            return true;

        Matrix3 base_rot = ioBody->rootLink()->R();
        if (base_rot.col(2).z() < 0.5)
            return true;

        return false;
    }
};

CNOID_IMPLEMENT_SIMPLE_CONTROLLER_FACTORY(VnoidRLController)
