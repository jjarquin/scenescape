// SPDX-FileCopyrightText: (C) 2019 - 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <rv/tracking/CPModel.hpp>

namespace rv {
namespace tracking {

void CPModel::stateConversionFunction(const cv::Mat &x_k, const cv::Mat &u_k, const cv::Mat &v_k, cv::Mat &x_kplus1)
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

  x_kplus1.at<double>(0, 0) = x; // Position in X
  x_kplus1.at<double>(1, 0) = y; // Position in Y
  x_kplus1.at<double>(2, 0) = z; // Position in Z
  x_kplus1.at<double>(3, 0) = 0.;                             // Velocity in X
  x_kplus1.at<double>(4, 0) = 0.;                             // Velocity in Y
  x_kplus1.at<double>(5, 0) = 0.;                             // Velocity in Z
  x_kplus1.at<double>(6, 0) = 0.;                                           // Acceleration in X
  x_kplus1.at<double>(7, 0) = 0.;                                           // Acceleration in Y
  x_kplus1.at<double>(8, 0) = 0.;                                           // Acceleration in Z
  x_kplus1.at<double>(9, 0) = 0.;                                   // Jerk in X
  x_kplus1.at<double>(10, 0) = 0.;                                  // Jerk in Y
  x_kplus1.at<double>(11, 0) = 0.;                                  // Jerk in Z
  x_kplus1.at<double>(12, 0) = length;                              // Length
  x_kplus1.at<double>(13, 0) = width;                               // Width
  x_kplus1.at<double>(14, 0) = height;                              // Height
  x_kplus1.at<double>(15, 0) = yaw;                                 // Yaw
  x_kplus1.at<double>(16, 0) = yawRate;                             // Yaw Rate

  x_kplus1 += vk; // additive process noise
}

void CPModel::measurementFunction(const cv::Mat &x_k, const cv::Mat &n_k, cv::Mat &z_k)
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

cv::Mat CPModel::processNoiseCovariance(double processNoise, double deltaT, int type)
{
  return cv::Mat::eye(17, 17, type) * processNoise;
}

cv::Mat CPModel::measurementNoiseCovariance(double measurementNoise, double deltaT, int type)
{
  return cv::Mat::eye(7, 7, type) * measurementNoise;
}

} // namespace tracking
} // namespace rv