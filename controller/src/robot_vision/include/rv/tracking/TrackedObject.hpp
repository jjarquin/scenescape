// SPDX-FileCopyrightText: (C) 2019 - 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <string>
#include <unordered_map>
#include <memory>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/nil_generator.hpp>
#include <boost/uuid/uuid_hash.hpp>
#include <deque>

#include "rv/tracking/Classification.hpp"
#include "rv/tracking/VisualFeatures.hpp"

namespace rv {
namespace tracking {

class TrackedObject
{
public:
  TrackedObject();

  static const int StateSize;
  static const int MeasurementSize;

  boost::uuids::uuid uuid{boost::uuids::nil_uuid()};

  // Position
  double x{0.};
  double y{0.};
  double z{0.};

  // Linear Velocity
  double vx{0.};
  double vy{0.};
  double vz{0.};

  // Linear Acceleration
  double ax{0.};
  double ay{0.};
  double az{0.};

  // Jerk
  double jx{0.};
  double jy{0.};
  double jz{0.};

  // Orientation
  double yaw{0.};
  double previousYaw{0.};

  // Angular velocity
  double yawRate{0.}; // Turn rate

  // Size
  double length{0.}; // along x
  double width{0.};  // along y
  double height{0.}; // along z

  bool corrected{false};

  std::string toString() const;

  // tracked object parameters
  cv::Mat predictedMeasurementMean;
  cv::Mat predictedMeasurementCov;
  cv::Mat predictedMeasurementCovInv;
  cv::Mat errorCovariance;

  Classification classification;

  double trackedTime{0.}; // time in seconds the object has been tracked consecutively
  double predictedTime{0.}; // time in seconds the object has been predicted consecutively
  double age{0.}; // total time in seconds present in the database

  double visualCertainty{0.}; // mean value of the similarity index

  double associationProbability{0.}; // Computed using current state and assigned measurement

  std::unordered_map<std::string, std::string> attributes;

  void updateOnPrediction(double deltaT);
  void updateOnCorrection(const TrackedObject &measurement);

  TrackedObject cloneState() const;

  std::string uuidString() const;

  void setUuidString(const std::string &newUuidString);

  bool isUuidNil();

  bool isDynamic() const;

  Eigen::VectorXf getVectorXf() const;

  void setVectorXf(const Eigen::VectorXf &vector);

  const Eigen::VectorXd getVisualFeatures() const
  {
    if (visualFeaturesSet.size() > 0)
    {
      return visualFeaturesSet.back();
    }
    else
    {
      return Eigen::VectorXd();
    }
  }

  void addVisualFeatures(Eigen::VectorXd visualFeatures);

  const Eigen::MatrixXd &getVisualFeaturesMatrix() const { return visualFeaturesMatrix;}

  const std::deque<Eigen::VectorXd> &getVisualFeaturesSet() const {return visualFeaturesSet;}

  void setVisualFeaturesSet(std::deque<Eigen::VectorXd> & _visualFeaturesSet)
  {
    if (_visualFeaturesSet.size() >= visual::MaxVisualFeaturesSetSize)
    {
      visualFeaturesSet = std::deque<Eigen::VectorXd>(_visualFeaturesSet.end() - visual::MaxVisualFeaturesSetSize, _visualFeaturesSet.end());
    }
    else
    {
      visualFeaturesSet = _visualFeaturesSet;
    }

    visualFeaturesMatrix = createVisualFeaturesMatrix();
  }
  /**
   * @brief Convert to a cv::Mat vector.
   */
  cv::Mat stateVector() const;

  /**
   * @brief Fill data from a cv::Mat vector.
   */
  void setStateVector(const cv::Mat &vector);

  /**
   * @brief Convert to a cv::Mat vector.
   */
  cv::Mat measurementVector() const;

  private:
    std::deque<Eigen::VectorXd> visualFeaturesSet; // historic visual features, for test purposes
    Eigen::MatrixXd visualFeaturesMatrix; // historic visual features in matrix form

    Eigen::MatrixXd createVisualFeaturesMatrix();
};

double calculateVisualScalingFactor(const TrackedObject &track, double alpha = 0.5);

double calculateMahalanobisDistance(const TrackedObject &track, const TrackedObject &measurement);

double calculateAssociationProbability(const TrackedObject &track, const TrackedObject &measurement);

} // namespace tracking
} // namespace rv
