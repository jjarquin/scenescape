// SPDX-FileCopyrightText: 2019 - 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
 
#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <string>
#include <vector>
#include <cmath>
#include <tuple>
#include <stdexcept>
#include <rv/Utils.hpp>

namespace rv {
namespace tracking {
namespace visual {

constexpr size_t MaxVisualFeaturesSetSize = 1024;

double inline similarity(const Eigen::VectorXd & visualFeaturesA, const Eigen::VectorXd & visualFeaturesB)
{
  return std::clamp(visualFeaturesA.dot(visualFeaturesB) / (visualFeaturesA.norm() * visualFeaturesB.norm()), 0., 1.0);
}

Eigen::VectorXd inline similarity(const Eigen::VectorXd & visualFeatures, const Eigen::MatrixXd & visualFeaturesMatrix)
{
    Eigen::VectorXd dot = visualFeaturesMatrix.transpose() * visualFeatures;
    Eigen::VectorXd norm = (visualFeaturesMatrix.colwise().norm() * visualFeatures.norm());

    return (dot.array() / norm.array()).min(1.0).max(0.0);
}

double inline distance(const Eigen::VectorXd & visualFeaturesA, const Eigen::VectorXd & visualFeaturesB)
{
  return std::clamp(1.0 - visualFeaturesA.dot(visualFeaturesB) / (visualFeaturesA.norm() * visualFeaturesB.norm()), 0., 1.0);
}

Eigen::VectorXd inline distance(const Eigen::VectorXd & visualFeatures, const  Eigen::MatrixXd & visualFeaturesMatrix)
{
    Eigen::VectorXd dot = visualFeaturesMatrix.transpose() * visualFeatures;
    Eigen::VectorXd norm = (visualFeaturesMatrix.colwise().norm() * visualFeatures.norm());

    return (1.0 - dot.array() / norm.array()).min(1.0).max(0.0);
}

} // namespace visual
} // namespace tracking
} // namespace rv