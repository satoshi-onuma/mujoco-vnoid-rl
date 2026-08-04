#pragma once

#include "robot_mujoco.h"
#include "iksolver.h"
#include "fksolver.h"
#include "footstep.h"
#include "footstep_planner.h"
#include "stepping_controller.h"
#include "stabilizer.h"
#include "types.h"

namespace cnoid{
namespace vnoid{

class MyRobot : public RobotMujoco{
public:
    double    standby_period;      ///< period of initial standby mode
	double    standby_com_height;  ///< com height in standby mode

    int       plan_cycle;
    bool      use_joystick;
    double    max_stride;
    double    max_turn;
    
	Timer            timer;
    Param            param;
    Centroid         centroid;
    Base             base;
    vector<Hand>     hand;
    vector<Foot>     foot;
    vector<Joint>    joint;
    Footstep         footstep;    
    Footstep         footstep_buffer;

    FootstepPlanner     footstep_planner;
    SteppingController  stepping_controller;
    Stabilizer          stabilizer;
    FkSolver            fk_solver;
    IkSolver            ik_solver;

    // 歩容コマンド（唯一の真実源）
    WalkCommand walk_cmd;

public:
	
	virtual void Init(mjModel* _m, mjData* _d);
	virtual void Control(const RLParams& rl_params); // ★引数を追加
    void ResetState(); 

    // コマンド設定（デバッグしやすいよう用途別に分離）
    void SetFixedWalkCommand();
    void SetWalkCommand(double stride, double sway, double turn);

    // 現在の walk_cmd で footstep を再計画
    void UpdateFootstepPlan();
	
	MyRobot();

};

}
}
