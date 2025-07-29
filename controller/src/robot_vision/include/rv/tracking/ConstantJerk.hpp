// SPDX-FileCopyrightText: (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "rv/tracking/MotionModel.hpp"
#include <opencv2/imgproc/imgproc.hpp>

namespace rv {
namespace tracking {

/**
 * @brief ConstantJerk: Implements a cv::detail::tracking::UkfSystemModel
 *
 * The ConstantJerk is a UkfSystemModel which overrides the state conversion and measurement functions
 * The ConstantJerk refers to the Constant jerk model.
 *
 * See "Comparison and evaluation of advanced motion models for vehicle tracking".
 */
class ConstantJerk : public MotionModel
{
public:
  /**
   * @brief State transition function for the Constant Jerk Model
   */

  void stateConversionFunction(const cv::Mat &x_k, const cv::Mat &u_k, const cv::Mat &v_k, cv::Mat &x_kplus1) override;

  /**
    * @brief State measurement function for the Constant Jerk Model
    */
  void measurementFunction(const cv::Mat &x_k, const cv::Mat &n_k, cv::Mat &z_k) override;

  cv::Mat processNoiseCovariance(double processNoise, double deltaT, int type) override;

  cv::Mat measurementNoiseCovariance(double measurementNoise, double deltaT, int type) override;

  double timeVarying () override {return false;}
};
} // namespace tracking
} // namespace rv