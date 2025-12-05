#include "myrobot.h"

using namespace std;

namespace cnoid{
namespace vnoid{
    

MyRobot::MyRobot(){
    base_actuation = false;
    
}

void MyRobot::Init(mjModel* _m, mjData* _d){
    // init params
    //  control parameters
    param.control_cycle = 1;

    //  dynamical parameters
	param.total_mass = 43.0;
    param.nominal_inertia = Vector3(2.5, 2.5, 0.2);
	param.com_height =  0.70;
	param.gravity    =  9.8;
    
    // kinematic parameters
    param.base_to_shoulder[0] = Vector3(0.0, -0.1,  0.3);
    param.base_to_shoulder[1] = Vector3(0.0,  0.1,  0.3);
    param.base_to_hip     [0] = Vector3(0.0, -0.1, -0.1);
    param.base_to_hip     [1] = Vector3(0.0,  0.1, -0.1);
    param.wrist_to_hand   [0] = Vector3(0.0,  0.0, -0.1);
    param.wrist_to_hand   [1] = Vector3(0.0,  0.0, -0.1);
    param.ankle_to_foot   [0] = Vector3(0.0,  0.0, -0.05);
    param.ankle_to_foot   [1] = Vector3(0.0,  0.0, -0.05);
    param.arm_joint_index [0] =  4;
    param.arm_joint_index [1] = 11;
    param.leg_joint_index [0] = 18;
    param.leg_joint_index [1] = 24;
    param.upper_arm_length = 0.2;
    param.lower_arm_length = 0.2;
    param.upper_leg_length = 0.3;
    param.lower_leg_length = 0.4;

    param.trunk_mass = 24.0;
    param.trunk_com = Vector3(0.0, 0.0, 0.166);

    param.arm_mass[0] = 0.5;
    param.arm_mass[1] = 0.5;
    param.arm_mass[2] = 1.0;
    param.arm_mass[3] = 0.5;
    param.arm_mass[4] = 1.0;
    param.arm_mass[5] = 0.5;
    param.arm_mass[6] = 0.5;
    param.arm_com[0] = Vector3(0.0, 0.0,  0.0);
    param.arm_com[1] = Vector3(0.0, 0.0,  0.0);
    param.arm_com[2] = Vector3(0.0, 0.0, -0.1);
    param.arm_com[3] = Vector3(0.0, 0.0,  0.0);
    param.arm_com[4] = Vector3(0.0, 0.0, -0.1);
    param.arm_com[5] = Vector3(0.0, 0.0,  0.0);
    param.arm_com[6] = Vector3(0.0, 0.0,  0.0);

    param.leg_mass[0] = 0.5;
    param.leg_mass[1] = 0.5;
    param.leg_mass[2] = 1.5;
    param.leg_mass[3] = 1.5;
    param.leg_mass[4] = 0.5;
    param.leg_mass[5] = 0.5;
    param.leg_com[0] = Vector3(0.0, 0.0,  0.0);
    param.leg_com[1] = Vector3(0.0, 0.0,  0.0);
    param.leg_com[2] = Vector3(0.0, 0.0, -0.15);
    param.leg_com[3] = Vector3(0.0, 0.0, -0.20);
    param.leg_com[4] = Vector3(0.0, 0.0,  0.0);
    param.leg_com[5] = Vector3(0.0, 0.0,  0.0);

    // stabilizer uses z-movement of zmp to stabilize com height
    // so certain admissible range in z-direction is needed
    param.zmp_min = Vector3(-0.1, -0.05, -0.1);
    param.zmp_max = Vector3( 0.1,  0.05,  0.1);

    param.Init();

    // two hands and two feet
     foot.resize(2);
    hand.resize(2);

    // 30 joints
    // assign small PD gains to ankle joints so that torque-based ZMP control works well
    joint_pos_filter_cutoff = 10.0;
    joint.resize(30);
    joint[ 0].Set(500.0, 100.0, 100.0);
    joint[ 1].Set(500.0, 100.0, 100.0);
    joint[ 2].Set(500.0, 100.0, 100.0);
    joint[ 3].Set(500.0, 100.0, 100.0);
    joint[ 4].Set(500.0, 100.0, 100.0);
    joint[ 5].Set(500.0, 100.0, 100.0);
    joint[ 6].Set(500.0, 100.0, 100.0);
    joint[ 7].Set(500.0, 100.0, 100.0);
    joint[ 8].Set(500.0, 100.0, 100.0);
    joint[ 9].Set(500.0, 100.0, 100.0);
    joint[10].Set(500.0, 100.0, 100.0);
    joint[11].Set(500.0, 100.0, 100.0);
    joint[12].Set(500.0, 100.0, 100.0);
    joint[13].Set(500.0, 100.0, 100.0);
    joint[14].Set(500.0, 100.0, 100.0);
    joint[15].Set(500.0, 100.0, 100.0);
    joint[16].Set(500.0, 100.0, 100.0);
    joint[17].Set(500.0, 100.0, 100.0);
    joint[18].Set(2000.0, 400.0, 200.0);
    joint[19].Set(2000.0, 400.0, 200.0);
    joint[20].Set(2000.0, 400.0, 200.0);
    joint[21].Set(2000.0, 400.0, 200.0);
    joint[22].Set(100.0, 20.0, 100.0);
    joint[23].Set(100.0, 20.0, 100.0);
    joint[24].Set(2000.0, 400.0, 200.0);
    joint[25].Set(2000.0, 400.0, 200.0);
    joint[26].Set(2000.0, 400.0, 200.0);
    joint[27].Set(2000.0, 400.0, 200.0);
    joint[28].Set(100.0, 20.0, 100.0);
    joint[29].Set(100.0, 20.0, 100.0);
    
    // init hardware (simulator interface)
    gyro_filter_cutoff = 20.0;
    
    // init hardware (simulator interface)
	RobotMujoco::Init(_m, _d, param, timer, joint);

    // set initial state
    base.ori_ref   = Quaternion(1.0, 0.0, 0.0, 0.0);
    base.angle_ref = Vector3(0.0, 0.0, 0.0);
    centroid.com_pos_ref = Vector3(0.0, 0.0, param.com_height);
    centroid.com_vel_ref = Vector3(0.0, 0.0, 0.0);
    centroid.com_acc_ref = Vector3(0.0, 0.0, 0.0);
    centroid.zmp_ref     = Vector3(0.0, 0.0, 0.0);
    centroid.zmp_target  = Vector3(0.0, 0.0, 0.0);
    centroid.dcm_ref     = Vector3(0.0, 0.0, param.com_height);
    centroid.dcm_target  = Vector3(0.0, 0.0, param.com_height);
    foot[0].pos_ref = Vector3(0.02, -0.10, 0.0);
    foot[1].pos_ref = Vector3(0.02,  0.10, 0.0);
    foot[0].contact_ref = true;
    foot[1].contact_ref = true;


    // ★追加: ReactiveWalkingControllerの初期化
    rea_con.swing_height     = 0.10;
    rea_con.swing_tilt       = 0.0;
    rea_con.nominal_duration = 0.45;
    rea_con.max_dcm_distance    = 0.5;
    rea_con.min_duration     = 0.3;
    rea_con.dsp_rate         = 0.05;
    rea_con.descend_rate        = 0.0;
    
    // ★追加: 初期gaitパラメータ設定
    rea_con.stride  = 0.0;
    rea_con.sway    = 0.0;
    rea_con.spacing = 0.15;
    rea_con.turn    = 0.0;
    rea_con.land_height = 0.0;
    
    // ★追加: 歩行開始フラグ
    rea_con.stepping = false;

    rea_con.min_contact_force = 100.0;
    rea_con.orientation_ctrl_gain_p = Vector3(400.0, 400.0, 10.0); //yaw 10.0
    rea_con.orientation_ctrl_gain_d = Vector3( 40.0,  40.0, 10.0); //yaw 10.0
    rea_con.orientation_ctrl_gain_i = Vector3(120.0, 120.0,  0.0);
    rea_con.orientation_ctrl_deadband = Vector3(0.0, 0.0, 0.0);
    rea_con.dcm_ctrl_gain       = 2.0;
    rea_con.base_tilt_rate      = 0.0;
    rea_con.base_tilt_damping_p = 1000.0;
    rea_con.base_tilt_damping_d = 100.0;
    rea_con.Ldmax = Vector3(30.0, 30.0, 0.0);

    contact_established = false;
    
    
    
}
void MyRobot::Control(const RLParams& rl_params){ // ★引数を追加
    if(timer.count % param.control_cycle == 0){    
        RobotMujoco::Sense(timer, base, foot, joint);

        // calc FK
        fk_solver.Comp(param, joint, base, centroid, hand, foot);

        

        /*

        if(!contact_established){
            // ★ 接地力で判定（より安定）
            double min_contact_force = -1.0;  // 閾値
            bool both_feet_contact = (foot[0].force.z() < min_contact_force) && 
                         (foot[1].force.z() < min_contact_force);
                            if(timer.count % 100 ==0)
                         printf("foot.contact[0],[1] = %f,%f/n",foot[0].force.z(),foot[1].force.z() );
            if(both_feet_contact){
                contact_established = true;
            }
        }


        // ★ 接地待機中の処理
        if(!contact_established && timer.time > 0.7){
            // 接地するまでは目標位置を現在位置に設定（待機姿勢）
            centroid.com_pos_ref = centroid.com_pos;
            centroid.com_vel_ref = Vector3(0.0, 0.0, 0.0);
            centroid.zmp_ref     = Vector3(0.0, 0.0, 0.0);
            
            foot[0].pos_ref = foot[0].pos;
            foot[1].pos_ref = foot[1].pos;
            foot[0].angle_ref = foot[0].angle;
            foot[1].angle_ref = foot[1].angle;
            
            base.ori_ref = base.ori;
            base.angle_ref = base.angle;
        }
        */
        
        if(!rea_con.stepping && timer.time > 0.0){
            centroid.zmp_target   = foot[rea_con.sup].pos_ref;
            rea_con.lift_pos   = foot[rea_con.swg].pos_ref;
            rea_con.lift_angle = foot[rea_con.swg].angle_ref;

            rea_con.CalcDcmOffset(param);
            rea_con.duration = rea_con.nominal_duration;
            rea_con.t_land   = rea_con.nominal_duration + param.T*log(rea_con.dcm_offset[rea_con.sup].norm());
            //controller.dcm_scale = controller.dcm_offset[controller.sup].norm();
            rea_con.time_switch = timer.time - param.T*log(
                Vector2(centroid.dcm_ref.x() - foot[rea_con.sup].pos_ref.x(), centroid.dcm_ref.y() - foot[rea_con.sup].pos_ref.y()).norm()/rea_con.dcm_offset[rea_con.sup].norm()
            );
            rea_con.stepping    = true;
        }
        if(timer.time > 0.0){
            rea_con.Ldmax = Vector3(60.0, 60.0, 0);
        }

            // ★ 初期化処理（ReactiveWalkingController用） ★
        // ★追加: RLからのgaitパラメータ設定
        
        if (timer.time>0.2)
        {
        rea_con.stride  = 0.2 + rl_params.stride_offset;  
        rea_con.sway    = 0.0;
        rea_con.spacing = 0.15 + rl_params.spacing_offset;
        rea_con.turn    = 0.0 + rl_params.turn_offset;
        rea_con.nominal_duration  = 0.45 + rl_params.duration_offset;
        rea_con.land_height = rl_params.climb_offset;  
        }
        

        
        
        rea_con.Update(timer, param, centroid, base, foot);
            
        
        
    
        hand[0].pos_ref = centroid.com_pos_ref + base.ori_ref*Vector3(0.0, -0.25, -0.1);
        hand[0].ori_ref = base.ori_ref;
        hand[1].pos_ref = centroid.com_pos_ref + base.ori_ref*Vector3(0.0,  0.25, -0.1);
        hand[1].ori_ref = base.ori_ref;

        // calc CoM IK
        ik_solver.Comp(&fk_solver, param, centroid, base, hand, foot, joint);

        RobotMujoco::Actuate(timer, base, joint);

        timer.control_count++;
    }
	timer.Countup();
}


}
}
