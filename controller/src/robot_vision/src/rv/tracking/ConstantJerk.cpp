// SPDX-FileCopyrightText: (C) 2019 - 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "rv/tracking/ConstantJerk.hpp"

namespace rv {
namespace tracking {

constexpr double alpha = 0.001;

void ConstantJerk::stateConversionFunction(const cv::Mat &x_k, const cv::Mat &u_k, const cv::Mat &v_k, cv::Mat &x_kplus1)
{
  cv::Mat vk = v_k.clone();

  /*
   * The time is considered the control input
   */
  const double &deltaT = u_k.at<double>(0, 0);

  const double & x = x_k.at<double>(0, 0);
  const double & y = x_k.at<double>(1, 0);
  const double & z = x_k.at<double>(2, 0);
  const double & vx = x_k.at<double>(3, 0);
  const double & vy = x_k.at<double>(4, 0);
  const double & vz = x_k.at<double>(5, 0);
  const double & ax = x_k.at<double>(6, 0);
  const double & ay = x_k.at<double>(7, 0);
  const double & az = x_k.at<double>(8, 0);
  const double & jx = x_k.at<double>(9, 0);
  const double & jy = x_k.at<double>(10, 0);
  const double & jz = x_k.at<double>(11, 0);

  const double & length = x_k.at<double>(12, 0);
  const double & width = x_k.at<double>(13, 0);
  const double & height = x_k.at<double>(14, 0);
  const double & yaw = x_k.at<double>(15, 0);
  const double & yawRate = x_k.at<double>(16, 0);

  const double deltaT2 = deltaT * deltaT;
  const double deltaT3 = deltaT * deltaT2;

  /*
   * The equations for the constant jerk model are:
   */
  double px;
  double qx;
  double rx;
  double sx;

  double py;
  double qy;
  double ry;
  double sy;

  double pz;
  double qz;
  double rz;
  double sz;

  px = jx * deltaT3 / 6;
  py = jy * deltaT3 / 6;
  pz = jz * deltaT3 / 6;

  qx = 0.5 * jx * deltaT2;
  qy = 0.5 * jy * deltaT2;
  qz = 0.5 * jz * deltaT2;

  rx = jx * deltaT;
  ry = jy * deltaT;
  rz = jz * deltaT;

  const double expAlphaT = std::exp(-alpha * deltaT);

  // Simulate jerk decay
  sx = expAlphaT * jx;
  sy = expAlphaT * jy;
  sz = expAlphaT * jz;

  x_kplus1.at<double>(0, 0) = x + vx * deltaT + 0.5 * ax * deltaT2 + px; // Position in X
  x_kplus1.at<double>(1, 0) = y + vy * deltaT + 0.5 * ay * deltaT2 + py; // Position in Y
  x_kplus1.at<double>(2, 0) = z + vz * deltaT + 0.5 * az * deltaT2 + pz; // Position in Z
  x_kplus1.at<double>(3, 0) = vx + ax * deltaT + qx;                     // Velocity in X
  x_kplus1.at<double>(4, 0) = vy + ay * deltaT + qy;                     // Velocity in Y
  x_kplus1.at<double>(5, 0) = vz + az * deltaT + qz;                     // Velocity in Z
  x_kplus1.at<double>(6, 0) = ax + rx;                                   // Acceleration in X
  x_kplus1.at<double>(7, 0) = ay + ry;                                   // Acceleration in Y
  x_kplus1.at<double>(8, 0) = az + rz;                                   // Acceleration in Z
  x_kplus1.at<double>(9, 0) = sx;                                   // Jerk in X
  x_kplus1.at<double>(10, 0) = sy;                                  // Jerk in Y
  x_kplus1.at<double>(11, 0) = sz;                                  // Jerk in Z
  x_kplus1.at<double>(12, 0) = length;                              // Length
  x_kplus1.at<double>(13, 0) = width;                               // Width
  x_kplus1.at<double>(14, 0) = height;                              // Height
  x_kplus1.at<double>(15, 0) = yaw;                                 // Yaw
  x_kplus1.at<double>(16, 0) = yawRate;                             // Yaw Rate

  x_kplus1 += vk; // additive process noise
}

void ConstantJerk::measurementFunction(const cv::Mat &x_k, const cv::Mat &n_k, cv::Mat &z_k)
{
  z_k.at<double>(0, 0) = x_k.at<double>(0, 0);  // Position in X
  z_k.at<double>(1, 0) = x_k.at<double>(1, 0);  // Position in Y
  z_k.at<double>(2, 0) = x_k.at<double>(2, 0);  // Position in Z
  z_k.at<double>(3, 0) = x_k.at<double>(12, 0);  // Length
  z_k.at<double>(4, 0) = x_k.at<double>(13, 0);  // Width
  z_k.at<double>(5, 0) = x_k.at<double>(14, 0);  // Height
  z_k.at<double>(6, 0) = x_k.at<double>(15, 0); // Yaw
  z_k += n_k;                                   // additive measurement noise
}

  cv::Mat ConstantJerk::processNoiseCovariance(double processNoise, double deltaT, int type)
  {
    const double T = deltaT;
    const double T2 = T * T;
    const double T3 = T2 * T;
    const double T4 = T3 * T;
    const double T5 = T4 * T;
    const double T6 = T5 * T;
    const double T7 = T6 * T;

    cv::Mat M = (cv::Mat_<double>(12,12) <<  \
      T7/252.,0,0,  T6/72.,0,0,  T5/30.,0,0,  T4/24.,0,0, \
      0,T7/252.,0,  0,T6/72.,0,  0,T5/30.,0,  0,T4/24.,0, \
      0,0,T7/252.,  0,0,T6/72.,  0,0,T5/30.,  0,0,T4/24., \

      T6/72.,0,0,  T5/20.,0,0,  T4/8.0,0,0,  T3/6.0,0,0, \
      0,T6/72.,0,  0,T5/20.,0,  0,T4/8.0,0,  0,T3/6.0,0, \
      0,0,T6/72.,  0,0,T5/20.,  0,0,T4/8.0,  0,0,T3/6.0, \

      T5/30.,0,0,  T4/8.0,0,0,  T3/3.,0,0,  T2/2.0,0,0, \
      0,T5/30.,0,  0,T4/8.0,0,  0,T3/3.,0,  0,T2/2.0,0, \
      0,0,T5/30.,  0,0,T4/8.0,  0,0,T3/3.,  0,0,T2/2.0, \

      T4/24.,0,0,  T3/6.0,0,0,  T2/2.0,0,0,  T,0,0, \
      0,T4/24.,0,  0,T3/6.0,0,  0,T2/2.0,0,  0,T,0, \
      0,0,T4/24.,  0,0,T3/6.0,  0,0,T2/2.0,  0,0,T  );

    auto covariance = cv::Mat::eye(17, 17, type) * processNoise;
    cv::Mat subMatrix = covariance(cv::Rect_<int>(0,0,12,12));
    M.copyTo(subMatrix);
    return covariance;
  }

  cv::Mat ConstantJerk::measurementNoiseCovariance(double measurementNoise, double deltaT, int type)
  {
    return cv::Mat::eye(7, 7, type) * measurementNoise;
  }

} // namespace tracking
} // namespace rv