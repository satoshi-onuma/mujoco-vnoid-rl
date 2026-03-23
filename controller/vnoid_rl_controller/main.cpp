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
    double dt;
    int num_joints;
    
    // 共有メモリインターフェース
    std::unique_ptr<ShmInterface> shm;
    
    // 関節情報
    std::vector<Link*> joints;
    std::vector<double> initial_q;      // 初期関節角度
    std::vector<double> q_prev;         // 前回の関節角度（速度計算用）
    std::vector<double> joint_lower;    // 関節下限
    std::vector<double> joint_upper;    // 関節上限
    
    // PD制御ゲイン
    std::vector<double> kp;
    std::vector<double> kd;
    
    // 初期ベース位置
    Vector3 initial_base_pos;
    Matrix3 initial_base_rot;
    
    // ステップカウンタ
    int step_count;
    
    // 前回のベース位置（速度計算用）
    Vector3 prev_base_pos;
    double prev_time;

public:
    virtual bool initialize(SimpleControllerIO* io) override {
        ioBody = io->body();
        dt = io->timeStep();
        num_joints = ioBody->numJoints();
        
        std::cout << "[VnoidRLController] Initializing..." << std::endl;
        std::cout << "  Number of joints: " << num_joints << std::endl;
        std::cout << "  Time step: " << dt << std::endl;
        
        // 環境IDを取得
        int env_id = ShmInterface::get_env_id();
        std::cout << "  Environment ID: " << env_id << std::endl;
        
        // 共有メモリを開く（Python側が作成済み）
        try {
            shm = std::make_unique<ShmInterface>(env_id, false);
            std::cout << "  Shared memory opened successfully" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "  ERROR: Failed to open shared memory: " << e.what() << std::endl;
            return false;
        }
        
        // 関節の初期化
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
            
            initial_q[i] = joints[i]->q();
            q_prev[i] = initial_q[i];
            joint_lower[i] = joints[i]->q_lower();
            joint_upper[i] = joints[i]->q_upper();
            
            // PD制御ゲイン設定（簡易版）
            kp[i] = 200.0;  // 比例ゲイン
            kd[i] = 20.0;   // 微分ゲイン
        }
        
        // 初期ベース位置・姿勢を保存
        initial_base_pos = ioBody->rootLink()->p();
        initial_base_rot = ioBody->rootLink()->R();
        prev_base_pos = initial_base_pos;
        prev_time = 0.0;
        
        step_count = 0;
        
        // 準備完了を通知
        shm->set_ready(true);
        shm->set_done(false);
        
        std::cout << "[VnoidRLController] Initialization complete!" << std::endl;
        return true;
    }

    virtual bool control() override {
        // リセット要求の確認
        if (shm->has_reset_request()) {
            reset_robot();
            shm->clear_reset_request();
            shm->set_ready(true);
            return true;
        }
        
        // ステップ要求の確認
        if (shm->has_step_request()) {
            // 行動を取得
            ActionBlock& action = shm->action();
            
            // 行動を関節目標角度に変換（正規化 [-1, 1] → 実際の角度）
            std::vector<double> target_q(num_joints);
            for (int i = 0; i < num_joints; ++i) {
                double normalized = std::clamp(action.joint_targets[i], -1.0, 1.0);
                double range = joint_upper[i] - joint_lower[i];
                target_q[i] = joint_lower[i] + (normalized + 1.0) * 0.5 * range;
            }
            
            // PD制御でトルク計算
            for (int i = 0; i < num_joints; ++i) {
                double q = joints[i]->q();
                double dq = joints[i]->dq();
                double error = target_q[i] - q;
                double u = kp[i] * error - kd[i] * dq;
                joints[i]->u() = u;
            }
            
            // 観測データを収集
            collect_observation();
            
            // 報酬を計算
            float reward = compute_reward();
            shm->set_reward(reward);
            
            // 終了条件をチェック
            bool done = check_termination();
            shm->set_done(done);
            
            // 準備完了を通知
            shm->clear_step_request();
            shm->set_ready(true);
            
            step_count++;
            
            if (done) {
                std::cout << "[VnoidRLController] Episode terminated at step " << step_count << std::endl;
            }
        }
        
        return true;
    }

private:
    void reset_robot() {
        std::cout << "[VnoidRLController] Resetting robot..." << std::endl;
        
        // ベース位置・姿勢をリセット
        ioBody->rootLink()->p() = initial_base_pos;
        ioBody->rootLink()->R() = initial_base_rot;
        ioBody->rootLink()->v().setZero();
        ioBody->rootLink()->w().setZero();
        
        // 関節角度をリセット
        for (int i = 0; i < num_joints; ++i) {
            joints[i]->q() = initial_q[i];
            joints[i]->dq() = 0.0;
            q_prev[i] = initial_q[i];
        }
        
        prev_base_pos = initial_base_pos;
        step_count = 0;
        
        shm->set_done(false);
        shm->set_reward(0.0);
    }
    
    void collect_observation() {
        StateBlock& state = shm->state();
        int idx = 0;
        
        // ベース位置 (3)
        Vector3 base_pos = ioBody->rootLink()->p();
        state.observations[idx++] = base_pos.x();
        state.observations[idx++] = base_pos.y();
        state.observations[idx++] = base_pos.z();
        
        // ベース姿勢（クォータニオン） (4)
        Matrix3 base_rot = ioBody->rootLink()->R();
        Quaternion q(base_rot);
        state.observations[idx++] = q.w();
        state.observations[idx++] = q.x();
        state.observations[idx++] = q.y();
        state.observations[idx++] = q.z();
        
        // ベース線速度 (3)
        Vector3 base_vel = ioBody->rootLink()->v();
        state.observations[idx++] = base_vel.x();
        state.observations[idx++] = base_vel.y();
        state.observations[idx++] = base_vel.z();
        
        // ベース角速度 (3)
        Vector3 base_angvel = ioBody->rootLink()->w();
        state.observations[idx++] = base_angvel.x();
        state.observations[idx++] = base_angvel.y();
        state.observations[idx++] = base_angvel.z();
        
        // 関節角度 (29)
        for (int i = 0; i < num_joints; ++i) {
            state.observations[idx++] = joints[i]->q();
        }
        
        // 関節角速度 (29)
        for (int i = 0; i < num_joints; ++i) {
            state.observations[idx++] = joints[i]->dq();
        }
        
        // 足接地状態 (2) - 簡易版：高さで判定
        // TODO: より正確な接地判定（力センサー等）
        state.observations[idx++] = (base_pos.z() < 0.05) ? 1.0 : 0.0;  // 左足
        state.observations[idx++] = (base_pos.z() < 0.05) ? 1.0 : 0.0;  // 右足
        
        // 合計73次元
    }
    
    float compute_reward() {
        // 簡易的な報酬関数
        float reward = 0.0;
        
        // 1. 前進速度報酬
        Vector3 base_pos = ioBody->rootLink()->p();
        double forward_vel = (base_pos.x() - prev_base_pos.x()) / dt;
        reward += forward_vel * 1.0;  // 前進するほど報酬
        
        // 2. 姿勢維持報酬（直立）
        Matrix3 base_rot = ioBody->rootLink()->R();
        Vector3 up_vec = base_rot.col(2);  // Z軸方向
        double upright = up_vec.z();       // 1.0に近いほど直立
        reward += upright * 0.5;
        
        // 3. 生存報酬
        reward += 0.1;
        
        // 4. エネルギーペナルティ
        double energy = 0.0;
        for (int i = 0; i < num_joints; ++i) {
            energy += std::abs(joints[i]->u() * joints[i]->dq());
        }
        reward -= energy * 0.0001;
        
        prev_base_pos = base_pos;
        
        return reward;
    }
    
    bool check_termination() {
        // 転倒判定
        Vector3 base_pos = ioBody->rootLink()->p();
        Matrix3 base_rot = ioBody->rootLink()->R();
        
        // 高さチェック
        if (base_pos.z() < 0.3) {
            return true;  // 低すぎる
        }
        
        // 姿勢チェック
        Vector3 up_vec = base_rot.col(2);
        if (up_vec.z() < 0.5) {
            return true;  // 傾きすぎ
        }
        
        return false;
    }
};

CNOID_IMPLEMENT_SIMPLE_CONTROLLER_FACTORY(VnoidRLController)
