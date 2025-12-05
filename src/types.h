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
