// SPDX-FileCopyrightText: (C) 2017 - 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "rv/Utils.hpp"
#include "rv/tracking/TrackManager.hpp"

namespace rv {
namespace tracking {

boost::uuids::uuid TrackManager::createTrack(TrackedObject object, const std::chrono::system_clock::time_point &timestamp)
{
  if (mAutoIdGeneration)
  {
    object.uuid = mUuidGenerator();
  }

  mKalmanEstimators[object.uuid].initialize(object, timestamp, mConfig.mDefaultProcessNoise, mConfig.mDefaultMeasurementNoise, mConfig.mInitStateCovariance, mConfig.mMotionModels);

  // Initialize non measurement and tracked frames counters
  mNonMeasurementFrames[object.uuid] = 0;
  mNumberOfTrackedFrames[object.uuid] = 0;

  return object.uuid;
}

void TrackManager::deleteTrack(const boost::uuids::uuid &uuid)
{
  if (isSuspended(uuid))
  {
    reactivateTrack(uuid);
  }

  mKalmanEstimators.erase(uuid);
  mNonMeasurementFrames.erase(uuid);
  mNumberOfTrackedFrames.erase(uuid);
}

void TrackManager::suspendTrack(const boost::uuids::uuid &uuid)
{
  mSuspendedKalmanEstimators[uuid] = std::move(mKalmanEstimators.at(uuid));
  mKalmanEstimators.erase(uuid);
  mNonMeasurementFrames.erase(uuid);
}

void TrackManager::reactivateTrack(const boost::uuids::uuid &uuid)
{
  mKalmanEstimators[uuid] = std::move(mSuspendedKalmanEstimators.at(uuid));

  // Initialize non measurement and tracked frames counters
  mNonMeasurementFrames[uuid] = 0;
  mNumberOfTrackedFrames[uuid] = mConfig.mMaxNumberOfUnreliableFrames - mConfig.mReactivationFrames;

  mSuspendedKalmanEstimators.erase(uuid);
}

void TrackManager::predict(const std::chrono::system_clock::time_point &timestamp)
{
  for (auto &element : mKalmanEstimators)
  {
    auto &estimator = element.second;
    estimator.predict(timestamp);
  }

  mMeasurementMap.clear();
}


void TrackManager::predict(double deltaT)
{
  for (auto &element : mKalmanEstimators)
  {
    auto &estimator = element.second;
    estimator.predict(deltaT);
  }

  mMeasurementMap.clear();
}

void TrackManager::correct()
{
  for (auto &element : mKalmanEstimators)
  {
    auto const &uuid = element.first;
    if (mMeasurementMap.count(uuid))
    {
      auto &estimator = element.second;
      auto const measurement = mMeasurementMap.find(uuid);
      estimator.correct(measurement->second);

      // Reset non measurement frames counter, increment tracked frames
      mNonMeasurementFrames[uuid] = 0;
      mNumberOfTrackedFrames[uuid]++;
    }
    else
    {
      mNonMeasurementFrames[uuid]++;
    }
  }

  std::vector<boost::uuids::uuid> reactivationList;
  for (auto &element : mSuspendedKalmanEstimators)
  {
    if (mMeasurementMap.count(element.first) > 0)
    {
      reactivationList.push_back(element.first);
    }
  }
  for (const auto &uuid : reactivationList)
  {
    reactivateTrack(uuid);
    mKalmanEstimators[uuid].correct(mMeasurementMap[uuid]);
  }

  std::vector<boost::uuids::uuid> deletionList;
  std::vector<boost::uuids::uuid> suspendList;

  // Check no longer valid states and delete accordingly
  for (const auto &element : mNonMeasurementFrames)
  {
    auto const &uuid = element.first;
    auto const &nonmeasurementFrames = element.second;

    if (isReliable(uuid))
    {
      uint32_t maxNonMeasurementFrames = 0;
      // let static objects stay longer
      if (mKalmanEstimators[uuid].currentState().isDynamic())
      {
        if (nonmeasurementFrames > mConfig.mNonMeasurementFramesDynamic)
        {
          deletionList.push_back(uuid);
        }
      }
      else
      {
        if (nonmeasurementFrames > mConfig.mNonMeasurementFramesStatic)
        {
          suspendList.push_back(uuid);
        }
      }
    }
    else
    {
      if (nonmeasurementFrames > mConfig.mNonMeasurementFramesDynamic)
      {
        deletionList.push_back(uuid);
      }
    }
  }
  for (const auto &uuid : deletionList)
  {
    deleteTrack(uuid);
  }
  for (const auto &uuid : suspendList)
  {
    suspendTrack(uuid);
  }
}

std::vector<TrackedObject> TrackManager::getTracks()
{
  std::vector<TrackedObject> tracks;

  for (const auto &element : mKalmanEstimators)
  {
    tracks.push_back(element.second.currentState());
  }
  for (const auto &element : mSuspendedKalmanEstimators)
  {
    tracks.push_back(element.second.currentState());
  }

  return tracks;
}

std::vector<TrackedObject> TrackManager::getReliableTracks()
{
  std::vector<TrackedObject> tracks;

  for (const auto &element : mKalmanEstimators)
  {
    if (isReliable(element.first))
    {
      tracks.push_back(element.second.currentState());
    }
  }

  return tracks;
}


std::vector<TrackedObject> TrackManager::getUnreliableTracks()
{
  std::vector<TrackedObject> tracks;

  for (const auto &element : mKalmanEstimators)
  {
    if (!isReliable(element.first))
    {
      tracks.push_back(element.second.currentState());
    }
  }

  return tracks;
}

std::vector<TrackedObject> TrackManager::getSuspendedTracks()
{
  std::vector<TrackedObject> tracks;

  for (const auto &element : mSuspendedKalmanEstimators)
  {
    tracks.push_back(element.second.currentState());
  }

  return tracks;
}

std::vector<TrackedObject> TrackManager::getDriftingTracks()
{
  std::vector<TrackedObject> tracks;

  for (const auto &element : mKalmanEstimators)
  {
    if (isReliable(element.first) && (mNonMeasurementFrames[element.first] > mConfig.mNonMeasurementFramesDynamic / 2))
    {
      tracks.push_back(element.second.currentState());
    }
  }

  return tracks;
}

void TrackManager::setMeasurement(const boost::uuids::uuid &uuid, const TrackedObject &measurement)
{
  auto previousMeasurement = mMeasurementMap.find(uuid);
  if (previousMeasurement != mMeasurementMap.end())
  {
    mMeasurementMap[uuid] = measurement;
  }
  else
  {
    mMeasurementMap.insert(std::make_pair(uuid, measurement));
  }
}

TrackedObject TrackManager::getTrack(const boost::uuids::uuid &uuid)
{
  return getKalmanEstimator(uuid).currentState();
}

MultiModelKalmanEstimator TrackManager::getKalmanEstimator(const boost::uuids::uuid &uuid)
{
  if (mKalmanEstimators.count(uuid) > 0)
  {
    return mKalmanEstimators[uuid];
  }
  else if(mSuspendedKalmanEstimators.count(uuid) > 0)
  {
    return mSuspendedKalmanEstimators[uuid];
  }
  else
  {
    throw std::runtime_error("The given uuid is not registered in this TrackManager.");
  }
}


bool TrackManager::hasUuid(const boost::uuids::uuid &uuid)
{
  return (mKalmanEstimators.count(uuid) > 0) || (mSuspendedKalmanEstimators.count(uuid) > 0);
}

bool TrackManager::isReliable(const boost::uuids::uuid &uuid)
{
  return mNumberOfTrackedFrames[uuid] >= mConfig.mMaxNumberOfUnreliableFrames;
}

bool TrackManager::isSuspended(const boost::uuids::uuid &uuid)
{
  return mSuspendedKalmanEstimators.count(uuid) > 0;
}

} // namespace tracking
} // namespace rv