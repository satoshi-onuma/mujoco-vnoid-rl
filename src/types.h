#pragma once

#ifdef VNOID_BUILD_CNOID
# include <cnoid/EigenTypes>
#else
# include <Eigen/Core>
# include <Eigen/Geometry>
 typedef Eigen::Matrix2d Matrix2;
 typedef Eigen::Vector2d Vector2;
 typedef Eigen::Matrix3d Matrix3;
 typedef Eigen::Vector3d Vector3;
 typedef Eigen::Vector3f Vector3f;
 typedef Eigen::Matrix<double, 6, 1> Vector6;
 typedef Eigen::AngleAxisd AngleAxis;
 typedef Eigen::Quaterniond Quaternion;
#endif
// RLからの指令を受け取るための構造体
 struct RLParams {
    Vector3 foot_offset = Vector3(0.0, 0.0, 0.0);
 };

// 歩容（コマンド）: 1歩あたりの変位/旋回量を基本単位として持つ
// - stride/sway/turn は joystick の指令と同型
// - spacing/climb/duration は footstep 生成でそのまま使用
struct WalkCommand {
   double stride   = 0.0;
   double sway     = 0.0;
   double turn     = 0.0;
   double spacing  = 0.2;
   double climb    = 0.0;
   double duration = 0.4;
};
