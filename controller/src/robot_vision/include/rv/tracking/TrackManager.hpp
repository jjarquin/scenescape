// SPDX-FileCopyrightText: 2017 - 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <stdint.h>
#include <string>
#include <unordered_map>
#include <chrono>
#include <vector>

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/nil_generator.hpp>

#include "rv/tracking/MultiModelKalmanEstimator.hpp"
#include "rv/tracking/TrackedObject.hpp"

namespace rv {
namespace tracking {

struct TrackManagerConfig
{
  uint32_t mNonMeasurementFramesDynamic{15};
  uint32_t mNonMeasurementFramesStatic{30};
  uint32_t mMaxNumberOfUnreliableFrames{2};
  uint32_t mReactivationFrames{1};

  double mDefaultProcessNoise{1e-3};
  double mDefaultMeasurementNoise{1e-2};
  double mInitStateCovariance{1.};

  std::vector<MotionModelType> mMotionModels{MotionModelType::CV, MotionModelType::CA};

  std::string toString() const
  {
    std::string motionModelsText = " motion_models:";
    for (auto const &motionModel: mMotionModels)
    {
      motionModelsText += " ";

      switch (motionModel)
      {
        case MotionModelType::CV:
          motionModelsText += "CV";
          break;
        case MotionModelType::CA:
          motionModelsText += "CA";
          break;
        case MotionModelType::CJ:
          motionModelsText += "CJ";
          break;
        case MotionModelType::CTRV:
          motionModelsText += "CTRV";
          break;
        default:
          motionModelsText += "Unknown";
      }
    }

    return "TrackManagerConfig( non_measurement_frames_dynamic:" + std::to_string(mNonMeasurementFramesDynamic)
      + ", non_measurement_frames_static:" + std::to_string(mNonMeasurementFramesStatic) + ", max_number_of_unreliable_frames:"
      + std::to_string(mMaxNumberOfUnreliableFrames) + ", reactivation_frames:" + std::to_string(mReactivationFrames)
      + ", default_process_noise:" + std::to_string(mDefaultProcessNoise) + ", default_measurement_noise:"
      + std::to_string(mDefaultMeasurementNoise) + ", init_state_covariance:"
      + std::to_string(mInitStateCovariance) + motionModelsText + ")";
  }
};

/**
 * @brief TrackManager: Provides interfaces to create new tracks and assign measurements to existing tracks
 *
 * The TrackManager module maintains tracks as a map of <boost::uuids::uuid, KalmanEstimator>
 * It also provides the functionality of Reliable/unreliable track, this reduces the number of false positives
 * and allows the user to work only with the reliable objects. An object becomes reliable when at least
 * mMaxNumberOfUnreliableFrames frames have been measured.
 */
class TrackManager
{
public:
  TrackManager()
  {
  }

  TrackManager(TrackManagerConfig const &trackManagerConfig)
    : mConfig(trackManagerConfig)
  {
  }

  TrackManager(bool autoIdGeneration)
    : mAutoIdGeneration(autoIdGeneration)
  {
  }

  TrackManager(TrackManagerConfig const &trackManagerConfig, bool autoIdGeneration)
    : mConfig(trackManagerConfig)
    , mAutoIdGeneration(autoIdGeneration)
  {
  }

  /**
   * @brief Create a new track with the object information
   *
   */
  boost::uuids::uuid createTrack(TrackedObject object, const std::chrono::system_clock::time_point &timestamp);

  /**
   * @brief Trigger state estimation update
   *
   */
  void predict(const std::chrono::system_clock::time_point &timestamp);

  /**
   * @brief Trigger state estimation update
   *
   */
  void predict(double deltaT);

  /**
   * @brief Assign a measurement to an KalmanEstimator.
   *
   * The measurement won't be applied inmediately, it will be applied during the next correct measurement step
   */
  void setMeasurement(const boost::uuids::uuid &uuid, const TrackedObject &measurement);

  /**
   * @brief Triggers the correct measurements step
   *
   */
  void correct();

  /**
   * @brief Access a specific track
   *
   */
  TrackedObject getTrack(const boost::uuids::uuid &uuid);

  /**
   * @brief Access a specific kalman estimator
   *
   */
  MultiModelKalmanEstimator getKalmanEstimator(const boost::uuids::uuid &uuid);

  /**
   * @brief Returns a list of tracked objects states
   *
   */
  std::vector<TrackedObject> getTracks();
  std::vector<TrackedObject> getReliableTracks();
  std::vector<TrackedObject> getUnreliableTracks();
  std::vector<TrackedObject> getSuspendedTracks();
  std::vector<TrackedObject> getDriftingTracks();

  /**
   * @brief Check wether the given boost::uuids::uuid is registered in the track manager
   *
   * @param uuid
   * @return true
   * @return false
   */
  bool hasUuid(const boost::uuids::uuid &uuid);

  /**
   * @brief Delete an existing track
   */
  void deleteTrack(const boost::uuids::uuid &uuid);

  /**
   * @brief Sets a track into suspended mode
   */
  void suspendTrack(const boost::uuids::uuid &uuid);

  /**
   * @brief Moves a track from suspended mode into non reliable tracks
   */
  void reactivateTrack(const boost::uuids::uuid &uuid);

  /**
   * @brief Track has been measured for at least mMaxNumberOfUnreliableFrames
   */
  bool isReliable(const boost::uuids::uuid &uuid);

  /**
   * @brief Track is in the mSuspendedKalmanEstimators map
   */
  bool isSuspended(const boost::uuids::uuid &uuid);

  inline TrackManagerConfig getConfig()
  {
    return mConfig;
  }

private:
  std::unordered_map<boost::uuids::uuid, MultiModelKalmanEstimator> mKalmanEstimators;
  std::unordered_map<boost::uuids::uuid, MultiModelKalmanEstimator> mSuspendedKalmanEstimators;
  std::unordered_map<boost::uuids::uuid, TrackedObject> mMeasurementMap;
  std::unordered_map<boost::uuids::uuid, uint32_t> mNonMeasurementFrames;
  std::unordered_map<boost::uuids::uuid, uint32_t> mNumberOfTrackedFrames;

  boost::uuids::random_generator mUuidGenerator;

  bool mAutoIdGeneration{true};

  TrackManagerConfig mConfig;
};

} // namespace tracking
} // namespace rv