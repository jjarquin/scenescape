// SPDX-FileCopyrightText: 2019 - 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <functional>
#include <numeric>
#include <opencv2/core.hpp>

#include "rv/tracking/ObjectMatching.hpp"
#include "rv/apollo/multi_hm_bipartite_graph_matcher.hpp"
#include "rv/apollo/secure_matrix.hpp"
#include "rv/tracking/Classification.hpp"

#include <iostream>

namespace rv {
namespace tracking {

constexpr double kDefaultClassBoundValue = 1000.;

double calculateVisualDistance(const TrackedObject &measurement, const TrackedObject &track)
{
  if ((track.getVisualFeaturesSet().size() > 0) && (measurement.getVisualFeaturesSet().size() > 0))
  {
    auto visualDistance = rv::tracking::visual::distance(measurement.getVisualFeatures(), track.getVisualFeaturesMatrix());

    return static_cast<double>(visualDistance.minCoeff());
  }
  else
  {
    return 1.0;
  }
}


double calculateSpatialDistance(const TrackedObject &measurement, const TrackedObject &track)
{
  return sqrt(pow(measurement.x - track.x, 2) + pow(measurement.y - track.y, 2) + pow(measurement.z - track.z, 2));
}

double calculateVisualSpatialDistance(const TrackedObject &measurement, const TrackedObject &track)
{
  auto visualDistance = calculateVisualDistance(measurement, track);
  auto visualScalingFactor = calculateVisualScalingFactor(track);
  auto spatialDistance = calculateSpatialDistance(measurement, track);

  return (0.5 + 2.0 * (0.5 + 0.5 * visualScalingFactor) * visualDistance) * spatialDistance;
}


double calculateVisualMultiClassDistance(const TrackedObject &measurement, const TrackedObject &track)
{
  auto visualDistance = calculateVisualDistance(measurement, track);
  auto visualScalingFactor = calculateVisualScalingFactor(track);
  auto multiClassDistance = rv::tracking::classification::distance(measurement.classification, track.classification);

  return visualScalingFactor * visualDistance + multiClassDistance;
}

double calculateSpatialMultiClassDistance(const TrackedObject &measurement, const TrackedObject &track)
{
  auto multiClassDistance = rv::tracking::classification::distance(measurement.classification, track.classification);

  double spatialDistance = calculateSpatialDistance(measurement, track);

  return (1.0 + multiClassDistance) * spatialDistance;
}

double calculateVisualSpatialMultiClassDistance(const TrackedObject &measurement, const TrackedObject &track)
{
  auto visualDistance = calculateVisualDistance(measurement, track);
  auto visualScalingFactor = calculateVisualScalingFactor(track);
  auto spatialDistance = calculateSpatialDistance(measurement, track);
  auto multiClassDistance = rv::tracking::classification::distance(measurement.classification, track.classification);

  return (0.25 + 2.0 * visualScalingFactor * visualDistance + 2.0 * multiClassDistance) * spatialDistance;
}

double calculateMatchingMahalanobisDistance(const TrackedObject &track, const TrackedObject &measurement)
{
  cv::Mat innovation = measurement.measurementVector() - (track.predictedMeasurementMean);

  // ignore yaw, 2D detectors cannot detect orientation
  innovation.at<double>(6, 0) = 0.;

  cv::Mat distance = innovation.t() * (track.predictedMeasurementCovInv) * innovation;

  return 0.5 * std::sqrt(distance.at<double>(0,0));
}

void match(const std::vector<TrackedObject> &tracks,
                          const std::vector<TrackedObject> &measurements,
                          std::vector<std::pair<size_t, size_t>> &assignments,
                          std::vector<size_t> &unassignedTracks,
                          std::vector<size_t> &unassignedMeasurements,
                          const DistanceType &distanceType, double threshold)
{
  apollo::perception::lidar::MultiHmBipartiteGraphMatcher matcher;

  matcher.cost_matrix()->Reserve(tracks.size(), measurements.size());

  assignments.clear();
  unassignedTracks.clear();
  unassignedMeasurements.clear();
  if (measurements.empty() || tracks.empty())
  {
    unassignedMeasurements.resize(measurements.size());
    unassignedTracks.resize(tracks.size());

    std::iota(unassignedMeasurements.begin(), unassignedMeasurements.end(), 0);
    std::iota(unassignedTracks.begin(), unassignedTracks.end(), 0);
    return;
  }

  apollo::perception::lidar::BipartiteGraphMatcherOptions matcherOptions;
  std::function<double(const TrackedObject &, const TrackedObject &)> distanceFunction;
  switch (distanceType)
  {
    case DistanceType::Visual:
      distanceFunction = std::bind(&calculateVisualDistance, std::placeholders::_1, std::placeholders::_2);
      matcherOptions.cost_thresh = threshold;
      matcherOptions.bound_value = kDefaultClassBoundValue;
      break;
    case DistanceType::Spatial:
      distanceFunction = std::bind(&calculateSpatialDistance, std::placeholders::_1, std::placeholders::_2);
      matcherOptions.cost_thresh = threshold;
      matcherOptions.bound_value = kDefaultClassBoundValue;
      break;
    case DistanceType::VisualSpatial:
      distanceFunction = std::bind(&calculateVisualSpatialDistance, std::placeholders::_1, std::placeholders::_2);
      matcherOptions.cost_thresh = threshold;
      matcherOptions.bound_value = kDefaultClassBoundValue;
      break;
    case DistanceType::VisualMultiClass:
      distanceFunction = std::bind(&calculateVisualMultiClassDistance, std::placeholders::_1, std::placeholders::_2);
      matcherOptions.cost_thresh = threshold;
      matcherOptions.bound_value = kDefaultClassBoundValue;
      break;
    case DistanceType::SpatialMultiClass:
      distanceFunction = std::bind(&calculateSpatialMultiClassDistance, std::placeholders::_1, std::placeholders::_2);
      matcherOptions.cost_thresh = threshold;
      matcherOptions.bound_value = kDefaultClassBoundValue;
      break;
    case DistanceType::VisualSpatialMultiClass:
      distanceFunction = std::bind(&calculateVisualSpatialMultiClassDistance, std::placeholders::_1, std::placeholders::_2);
      matcherOptions.cost_thresh = threshold;
      matcherOptions.bound_value = kDefaultClassBoundValue;
      break;
    case DistanceType::Mahalanobis:
      distanceFunction = std::bind(&calculateMahalanobisDistance, std::placeholders::_1, std::placeholders::_2);
      matcherOptions.cost_thresh = threshold;
      matcherOptions.bound_value = kDefaultClassBoundValue;
      break;
    default:
      distanceFunction = std::bind(&calculateSpatialDistance, std::placeholders::_1, std::placeholders::_2);
      matcherOptions.cost_thresh = threshold;
      matcherOptions.bound_value = kDefaultClassBoundValue;
      break;
  }

  apollo::perception::common::SecureMat<double> *costMatrix = matcher.cost_matrix();
  costMatrix->Resize(tracks.size(), measurements.size());

  for (size_t i = 0; i < tracks.size(); ++i)
  {
    for (size_t j = 0; j < measurements.size(); ++j)
    {
      (*costMatrix)(i, j) = distanceFunction(measurements[j], tracks[i]);
    }
  }

  matcher.Match(matcherOptions, &assignments, &unassignedTracks, &unassignedMeasurements);
}

} // namespace tracking
} // namespace rv