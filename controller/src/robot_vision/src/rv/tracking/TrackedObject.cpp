// SPDX-FileCopyrightText: (C) 2019 - 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "rv/tracking/TrackedObject.hpp"
#include <boost/uuid/uuid_io.hpp>
#include <boost/uuid/string_generator.hpp>

namespace rv {
namespace tracking {

const int TrackedObject::StateSize = 17;
const int TrackedObject::MeasurementSize = 7;

TrackedObject::TrackedObject()
{
  classification = Classification::Constant(1,1.0);

  predictedMeasurementMean = cv::Mat::zeros(TrackedObject::MeasurementSize, 1, CV_64F);
  predictedMeasurementCov = 1e-4 * cv::Mat::eye(TrackedObject::MeasurementSize, TrackedObject::MeasurementSize, CV_64F);
  predictedMeasurementCovInv = 1e4 * cv::Mat::eye(TrackedObject::MeasurementSize, TrackedObject::MeasurementSize, CV_64F);
  errorCovariance = 1e-4 * cv::Mat::eye(TrackedObject::StateSize, TrackedObject::StateSize, CV_64F);
}

void TrackedObject::updateOnPrediction(double deltaT)
{
  this->age += deltaT;
  this->predictedTime += deltaT;
  this->corrected = false;
}

void TrackedObject::updateOnCorrection(const TrackedObject &measurement)
{
  associationProbability = calculateAssociationProbability(*this, measurement);

  if (measurement.getVisualFeaturesSet().size() > 0)
  {
    if ((measurement.getVisualFeaturesSet().size() > 0) && (this->getVisualFeaturesSet().size() > 0))
    {
      visualCertainty = visual::similarity(measurement.getVisualFeatures(), this->getVisualFeaturesMatrix()).mean();
    }

    this->addVisualFeatures(measurement.getVisualFeatures());
  }

  this->classification = rv::tracking::classification::combine(this->classification , measurement.classification);
  this->attributes = measurement.attributes;

  this->trackedTime += this->predictedTime;
  this->predictedTime = 0.;
  this->corrected = true;
}

TrackedObject TrackedObject::cloneState() const
{
  TrackedObject trackedObject;

  trackedObject.x = x;
  trackedObject.y = y;
  trackedObject.z = z;

  trackedObject.vx = vx;
  trackedObject.vy = vy;
  trackedObject.vz = vz;

  trackedObject.ax = ax;
  trackedObject.ay = ay;
  trackedObject.az = az;

  trackedObject.jx = jx;
  trackedObject.jy = jy;
  trackedObject.jz = jz;

  // Orientation
  trackedObject.yaw = yaw;
  trackedObject.previousYaw = previousYaw;

  // Angular velocity
  trackedObject.yawRate = yawRate; // Turn rate

  // Size
  trackedObject.length = length; // along x
  trackedObject.width = width;  // along y
  trackedObject.height = height; // along z

  trackedObject.corrected = corrected;

  // tracked object parameters
  trackedObject.predictedMeasurementMean = predictedMeasurementMean.clone();
  trackedObject.predictedMeasurementCov = predictedMeasurementCov.clone();
  trackedObject.predictedMeasurementCovInv = predictedMeasurementCovInv.clone();
  trackedObject.errorCovariance = errorCovariance.clone();

  trackedObject.classification = classification;

  trackedObject.trackedTime = trackedTime; // time in seconds the object has been tracked consecutively
  trackedObject.predictedTime = predictedTime; // time in seconds the object has been predicted consecutively
  trackedObject.age = age; // total time in seconds present in the database

  trackedObject.visualCertainty = visualCertainty;

  trackedObject.associationProbability = associationProbability; // Probability of this object being an actual object

  return trackedObject;
}

std::string TrackedObject::uuidString() const
{
  return boost::uuids::to_string(uuid);
}

void TrackedObject::setUuidString(const std::string &newUuidString)
{
  boost::uuids::string_generator stringGenerator;

  uuid = stringGenerator(newUuidString);
}

bool TrackedObject::isUuidNil()
{
  return uuid.is_nil();
}

std::string TrackedObject::toString() const
{
  return "TrackedObject( uuid:" + boost::uuids::to_string(uuid)
    + ", x:" + std::to_string(x) + ", y:" + std::to_string(y) + ", z:" + std::to_string(z)
    + ", vx:" + std::to_string(vx) + ", vy:" + std::to_string(vy) + ", vz:" + std::to_string(vz)
    + ", ax:" + std::to_string(ax) + ", ay:" + std::to_string(ay) + ", az:" + std::to_string(az)
    + ", length:" + std::to_string(length) + ", width:" + std::to_string(width) + ", height:" + std::to_string(height)
    + ", yaw:" + std::to_string(yaw) + ", yaw_rate:" + std::to_string(yawRate) + ")";
}

bool TrackedObject::isDynamic() const
{
  return (vx * vx + vy * vy) > 1.0;
}

Eigen::VectorXf TrackedObject::getVectorXf() const
{
  Eigen::VectorXf vector(StateSize);
  vector(0) = x;
  vector(1) = y;
  vector(2) = z;

  vector(3) = vx;
  vector(4) = vy;
  vector(5) = vz;

  vector(6) = ax;
  vector(7) = ay;
  vector(8) = az;

  vector(9) = jx;
  vector(10) = jy;
  vector(11) = jz;

  vector(12) = length;
  vector(13) = width;
  vector(14) = height;

  vector(15) = yaw;
  vector(16) = yawRate; //turn rate

  return vector;
}

void TrackedObject::addVisualFeatures(Eigen::VectorXd visualFeatures)
{
  visualFeaturesSet.push_back(visualFeatures);

  if (visualFeaturesSet.size() >= visual::MaxVisualFeaturesSetSize)
  {
    visualFeaturesSet.pop_front();
  }

  visualFeaturesMatrix = createVisualFeaturesMatrix();
}

Eigen::MatrixXd TrackedObject::createVisualFeaturesMatrix()
{
  if (visualFeaturesSet.size() > 0)
  {
    Eigen::MatrixXd visualFeaturesMatrix(visualFeaturesSet.back().size(), visualFeaturesSet.size());
    for (int k = 0; k < visualFeaturesSet.size(); ++k)
    {
      // new (&visualFeaturesMatrix.col(k)) Eigen::Map<Eigen::VectorXd>(visualFeaturesSet[k].data(), visualFeaturesSet.back().size());
      visualFeaturesMatrix.col(k) = visualFeaturesSet[k];
    }
    return visualFeaturesMatrix;
  }
  else
  {
    return Eigen::MatrixXd();
  }
}

void TrackedObject::setVectorXf(const Eigen::VectorXf &vector)
{
  x = vector(0);
  y = vector(1);
  z = vector(2);

  vx = vector(3);
  vy = vector(4);
  vz = vector(5);

  ax = vector(6);
  ay = vector(7);
  az = vector(8);

  jx = vector(9);
  jy = vector(10);
  jz = vector(11);

  length = vector(12);
  width = vector(13);
  height = vector(14);

  yaw = vector(15);
  yawRate = vector(16);
}

/**
 * @brief Convert to a cv::Mat vector.
 */
cv::Mat TrackedObject::stateVector() const
{
  cv::Mat vector(StateSize, 1, CV_64F);
  vector.at<double>(0, 0) = x;
  vector.at<double>(1, 0) = y;
  vector.at<double>(2, 0) = z;

  vector.at<double>(3, 0) = vx;
  vector.at<double>(4, 0) = vy;
  vector.at<double>(5, 0) = vz;

  vector.at<double>(6, 0) = ax;
  vector.at<double>(7, 0) = ay;
  vector.at<double>(8, 0) = az;

  vector.at<double>(9, 0) = jx;
  vector.at<double>(10, 0) = jy;
  vector.at<double>(11, 0) = jz;

  vector.at<double>(12, 0) = length;
  vector.at<double>(13, 0) = width;
  vector.at<double>(14, 0) = height;

  vector.at<double>(15, 0) = yaw;
  vector.at<double>(16, 0) = yawRate;

  return vector;
}

/**
 * @brief Fill data from a cv::Mat vector.
 */
void TrackedObject::setStateVector(const cv::Mat &vector)
{
  x = vector.at<double>(0, 0);
  y = vector.at<double>(1, 0);
  z = vector.at<double>(2, 0);

  vx = vector.at<double>(3, 0);
  vy = vector.at<double>(4, 0);
  vz = vector.at<double>(5, 0);

  ax = vector.at<double>(6, 0);
  ay = vector.at<double>(7, 0);
  az = vector.at<double>(8, 0);

  jx = vector.at<double>(9, 0);
  jy = vector.at<double>(10, 0);
  jz = vector.at<double>(11, 0);

  length = vector.at<double>(12, 0);
  width = vector.at<double>(13, 0);
  height = vector.at<double>(14, 0);

  yaw = vector.at<double>(15, 0);
  yawRate = vector.at<double>(16, 0);
}

/**
 * @brief Convert to a cv::Mat vector.
 */
cv::Mat TrackedObject::measurementVector() const
{
  cv::Mat vector(MeasurementSize, 1, CV_64F);
  vector.at<double>(0, 0) = x;
  vector.at<double>(1, 0) = y;
  vector.at<double>(2, 0) = z;
  vector.at<double>(3, 0) = length;
  vector.at<double>(4, 0) = width;
  vector.at<double>(5, 0) = height;
  vector.at<double>(6, 0) = yaw;

  return vector;
}

double calculateMahalanobisDistance(const TrackedObject &track, const TrackedObject &measurement)
{
  cv::Mat innovation = measurement.measurementVector() - (track.predictedMeasurementMean);

  // ignore yaw, 2D detectors cannot detect orientation
  innovation.at<double>(6, 0) = 0.;

  cv::Mat distance = innovation.t() * (track.predictedMeasurementCovInv) * innovation;

  return 0.5 * std::sqrt(distance.at<double>(0,0));
}

double calculateAssociationProbability(const TrackedObject &track, const TrackedObject &measurement)
{
  double mahalanobisDistance = calculateMahalanobisDistance(track, measurement);

  double det = determinant(2.0 * M_PI * track.predictedMeasurementCov);

  auto likelihood = std::exp(- mahalanobisDistance) / std::sqrt(det);
  return likelihood / (1.0 + likelihood);
}

double calculateVisualScalingFactor(const TrackedObject &track, double alpha)
{
  double X = alpha * static_cast<double>(track.getVisualFeaturesSet().size());

  return X / (2.0 + X);
}

// TODO: ?, generate a vector of scaling factors, that prioritizes newer samples

} // namespace tracking
} // namespace rv