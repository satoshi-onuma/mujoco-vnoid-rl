#pragma once

#include "robot_mujoco.h"
#include "iksolver.h"
#include "fksolver.h"
#include "stabilizer.h"
#include "types.h"
#include "reactive_walking_controller.h"

namespace cnoid{
namespace vnoid{

class MyRobot : public RobotMujoco{
public:
    double    standby_period;      ///< period of initial standby mode
	double    standby_com_height;  ///< com height in standby mode
    
	Timer            timer;
    Param            param;
    Centroid         centroid;
    Base             base;
    vector<Hand>     hand;
    vector<Foot>     foot;
    vector<Joint>    joint;
    ReactiveWalkingController rea_con;  // ★追加
    Stabilizer          stabilizer;
    FkSolver            fk_solver;
    IkSolver            ik_solver;

public:
	
    bool contact_established; 
    double rl_start_delay;    
	virtual void Init(mjModel* _m, mjData* _d);
	virtual void Control(const RLParams& rl_params); // ★引数を追加
    void ResetState(); 
	
	MyRobot();

};

}
}
