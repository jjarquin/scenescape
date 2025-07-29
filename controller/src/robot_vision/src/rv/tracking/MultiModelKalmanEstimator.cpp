// SPDX-FileCopyrightText: (C) 2019 - 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "rv/tracking/MultiModelKalmanEstimator.hpp"
#include "rv/tracking/CAModel.hpp"
#include "rv/tracking/CTRVModel.hpp"
#include "rv/tracking/CVModel.hpp"
#include "rv/tracking/CPModel.hpp"
#include "rv/tracking/ConstantJerk.hpp"
#include "rv/tracking/Classification.hpp"

namespace rv {
namespace tracking {

MultiModelKalmanEstimator::MultiModelKalmanEstimator(double alpha, double beta)
  : mAlpha(alpha)
  , mBeta(beta)
  , mInitialized(false)
{
  mDP = TrackedObject::StateSize;
  mMP = TrackedObject::MeasurementSize;
  mCP = 1;            // Control vector is time
  mKappa = 3.0 - mDP; // 3 - state_size
}

void MultiModelKalmanEstimator::initialize(TrackedObject track, const std::chrono::system_clock::time_point &timestamp, double processNoise, double measurementNoise, double initStateCovariance, const std::vector<MotionModelType> &motionModels)
{
  mLastTimestamp = timestamp;

  mSystemModels.clear();

  if (motionModels.empty())
  {
    mSystemModels.push_back(cv::makePtr<tracking::CAModel>());
    mSystemModels.push_back(cv::makePtr<tracking::CVModel>());
  }
  else
  {
    for (auto const &motionModel : motionModels)
    {
      switch(motionModel)
      {
        case MotionModelType::CV:
          mSystemModels.push_back(cv::makePtr<tracking::CVModel>());
          break;
        case MotionModelType::CA:
          mSystemModels.push_back(cv::makePtr<tracking::CAModel>());
          break;
        case MotionModelType::CJ:
          mSystemModels.push_back(cv::makePtr<tracking::ConstantJerk>());
          break;
        case MotionModelType::CP:
          mSystemModels.push_back(cv::makePtr<tracking::CPModel>());
          break;
        case MotionModelType::CTRV:
          mSystemModels.push_back(cv::makePtr<tracking::CTRVModel>());
          break;
        default:
          break;
      }
    }
  }
  mProcessNoise = processNoise;
  mMeasurementNoise = measurementNoise;
  mNumberOfModels = mSystemModels.size();

  double pxModel = 1.0 / static_cast<double>(mNumberOfModels); // Initial model probability is uniform

  mModelProbability = cv::Mat(mNumberOfModels, 1, CV_64F, cv::Scalar(pxModel));

  double pxOtherModels = 0.10;                                // Probability of transition to other models
  double pxSameModel = 1.0 - mNumberOfModels * pxOtherModels; // Probability of transition to the same model

  mTransitionProbability = cv::Mat(mNumberOfModels, mNumberOfModels, CV_64F, cv::Scalar(pxOtherModels));
  mTransitionProbability += cv::Mat::eye(mNumberOfModels, mNumberOfModels, CV_64F) * pxSameModel;

  for (auto &model : mSystemModels)
  {
    cv::detail::tracking::UnscentedKalmanFilterParams modelParams;
    modelParams = cv::detail::tracking::UnscentedKalmanFilterParams(mDP, mMP, mCP, 0, 0, model);
    modelParams.stateInit = track.stateVector();
    modelParams.errorCovInit = cv::Mat::eye(mDP, mDP, CV_64F) * initStateCovariance;
    modelParams.measurementNoiseCov = model->measurementNoiseCovariance(mMeasurementNoise, 0.1, CV_64F);
    modelParams.processNoiseCov = model->processNoiseCovariance(mProcessNoise, 0.1, CV_64F);
    modelParams.alpha = mAlpha;
    modelParams.beta = mBeta;
    modelParams.k = mKappa;
    mKalmanFilters.push_back(createUnscentedKalmanFilterMod(modelParams));
    mSystemModelStates.push_back(track.cloneState());
  }

  mCurrentState = std::move(track);
  mCurrentState.trackedTime = 0.;
  mCurrentState.predictedTime = 0.;
  mCurrentState.age = 0.;

  mInitialized = true;
}


void MultiModelKalmanEstimator::singleModelPredict(double deltaT)
{
  cv::Mat deltaTVector = cv::Mat(mCP, 1, CV_64F, cv::Scalar(deltaT));
  cv::Mat noiseVector = cv::Mat::zeros(mMP, 1, CV_64F);

  auto predictedState = mKalmanFilters[0]->predict(deltaTVector);

  mCurrentState.previousYaw = mCurrentState.yaw;
  mCurrentState.setStateVector(predictedState); // combined current state
  mCurrentState.errorCovariance = mKalmanFilters[0]->getErrorCov().clone();
  mCurrentState.predictedMeasurementMean = cv::Mat::zeros(mMP, 1, CV_64F);

  mSystemModels[0]->measurementFunction(predictedState, noiseVector, mCurrentState.predictedMeasurementMean);

  if (mKalmanFilters[0]->getMeasurementCov().empty())
  {
    mCurrentState.predictedMeasurementCov = mKalmanFilters[0]->getMeasurementNoiseCov().clone();
  }
  else
  {
    mCurrentState.predictedMeasurementCov = mKalmanFilters[0]->getMeasurementCov().clone();
  }

  mCurrentState.predictedMeasurementCovInv = mCurrentState.predictedMeasurementCov.inv(cv::DECOMP_SVD);

  // To ensure the correct predict/correct flow
  mPredicted = true;
}

void MultiModelKalmanEstimator::predict(const std::chrono::system_clock::time_point &timestamp)
{
  predictState(rv::toSeconds(timestamp - mLastTimestamp));

  mLastTimestamp = timestamp;
}

void MultiModelKalmanEstimator::predict(const double deltaT)
{
  predictState(deltaT);

  mLastTimestamp = addSecondsToTimestamp(mLastTimestamp, std::chrono::duration<double>(deltaT));
}

void MultiModelKalmanEstimator::predictState(const double deltaT)
{
  if (!mInitialized)
  {
    throw std::runtime_error("You must call the KalmanEstimator::initialize() method before the prediction step.");
  }

  mCurrentState.updateOnPrediction(deltaT);

  for (size_t index = 0 ; index < mNumberOfModels ; index++)
  {
    if (mSystemModels[index]->timeVarying())
    {
      mKalmanFilters[index]->setProcessNoiseCov(mSystemModels[index]->processNoiseCovariance(mProcessNoise, deltaT, CV_64F));
    }
  }

  if (mNumberOfModels == 1)
  {
    return singleModelPredict(deltaT);
  }

  cv::Mat deltaTVector = cv::Mat(mCP, 1, CV_64F, cv::Scalar(deltaT));
  cv::Mat noiseVector = cv::Mat::zeros(mMP, 1, CV_64F);
  cv::Mat conditionalProbability = cv::Mat::zeros(mNumberOfModels, mNumberOfModels, CV_64F);

  combiningProbability(mTransitionProbability, mModelProbability, conditionalProbability);

  std::vector<cv::Mat> states;
  std::vector<cv::Mat> covariances;

  for (auto const &state : mSystemModelStates)
  {
    states.push_back(state.stateVector());
  }
  for (auto const &kalmanFilter : mKalmanFilters)
  {
    covariances.push_back(kalmanFilter->getErrorCov());
  }

  std::vector<cv::Mat> covarianceEstimate;
  std::vector<cv::Mat> stateEstimate;

  interaction(states, covariances, conditionalProbability, covarianceEstimate, stateEstimate);

  std::vector<cv::Mat> predictedStates;
  std::vector<cv::Mat> predictedStateCovariances;

  for (std::size_t i = 0; i < mNumberOfModels; ++i)
  {
    mKalmanFilters[i]->setStateAndCovariance(stateEstimate[i], covarianceEstimate[i]);
    mSystemModelStates[i].predictedMeasurementMean = cv::Mat::zeros(mMP, 1, CV_64F);
    auto predictedState = mKalmanFilters[i]->predict(deltaTVector);
    predictedStates.push_back(predictedState);
    predictedStateCovariances.push_back(mKalmanFilters[i]->getErrorCov());
    mSystemModelStates[i].setStateVector(predictedState);
    mSystemModels[i]->measurementFunction(predictedState, noiseVector, mSystemModelStates[i].predictedMeasurementMean);
  }

  cv::Mat combinedState;
  cv::Mat combinedCovariance;
  combineStatesAndCovariances(
    predictedStates, predictedStateCovariances, mModelProbability, combinedState, combinedCovariance);

  // save yaw before it is replaced by the predicted one
  mCurrentState.previousYaw = mCurrentState.yaw;
  mCurrentState.setStateVector(combinedState); // combined current state
  mCurrentState.errorCovariance = combinedCovariance.clone();

  // calculate combined measurement mean and covariance necessary for association
  std::vector<cv::Mat> measurements;
  std::vector<cv::Mat> measurementCovariances;

  for (auto const &state : mSystemModelStates)
  {
    measurements.push_back(state.predictedMeasurementMean);
  }

  for (auto const &kalmanFilter : mKalmanFilters)
  {
    if (kalmanFilter->getMeasurementCov().empty())
    {
      measurementCovariances.push_back(kalmanFilter->getMeasurementNoiseCov());
    }
    else
    {
      measurementCovariances.push_back(kalmanFilter->getMeasurementCov());
    }
  }

  cv::Mat combinedMeasurement;
  cv::Mat combinedMeasurementCovariance;
  combineStatesAndCovariances(
    measurements, measurementCovariances, mModelProbability, combinedMeasurement, combinedMeasurementCovariance);

  mCurrentState.predictedMeasurementMean = combinedMeasurement;
  mCurrentState.predictedMeasurementCov = combinedMeasurementCovariance.clone();
  mCurrentState.predictedMeasurementCovInv = combinedMeasurementCovariance.inv(cv::DECOMP_SVD);

  // To ensure the correct predict/correct flow
  mPredicted = true;
}


void MultiModelKalmanEstimator::singleModelCorrect(const TrackedObject &measurement)
{
  auto newMeasurement = measurement;
  newMeasurement.yaw = mCurrentState.previousYaw - rv::deltaTheta(measurement.yaw, mCurrentState.previousYaw);
  auto correctedState = mKalmanFilters[0]->correct(newMeasurement.measurementVector());

  mCurrentState.errorCovariance = mKalmanFilters[0]->getErrorCov().clone();
  mCurrentState.setStateVector(correctedState);

  // To ensure the correct predict/correct flow
  mPredicted = false;
}

void MultiModelKalmanEstimator::correct(const TrackedObject &measurement)
{
  if (!mPredicted)
  {
    throw std::runtime_error("You must call the KalmanEstimator::predict() method before the correction step.");
  }

  mCurrentState.updateOnCorrection(measurement);

  // motion related updates

  if (mNumberOfModels == 1)
  {
    return singleModelCorrect(measurement);
  }

  auto newMeasurement = measurement;
  newMeasurement.yaw = mCurrentState.previousYaw - rv::deltaTheta(measurement.yaw, mCurrentState.previousYaw);

  std::vector<cv::Mat> states;
  std::vector<cv::Mat> covariances;
  std::vector<cv::Mat> predictedMeasurements;
  std::vector<cv::Mat> measurementCovariances;

  for (std::size_t i = 0; i < mNumberOfModels; ++i)
  {
    auto correctedState = mKalmanFilters[i]->correct(newMeasurement.measurementVector());
    mSystemModelStates[i].setStateVector(correctedState);

    states.push_back(correctedState);
    covariances.push_back(mKalmanFilters[i]->getErrorCov());
    predictedMeasurements.push_back(mSystemModelStates[i].predictedMeasurementMean);
    measurementCovariances.push_back(mKalmanFilters[i]->getMeasurementCov());
  }

  cv::Mat combinedState;
  cv::Mat combinedCovariance;

  updateModelProbability(newMeasurement.measurementVector(),
                         predictedMeasurements,
                         measurementCovariances,
                         mModelProbability);
  combineStatesAndCovariances(states, covariances, mModelProbability, combinedState, combinedCovariance);

  mCurrentState.errorCovariance = combinedCovariance.clone();
  mCurrentState.setStateVector(combinedState);

  // To ensure the correct predict/correct flow
  mPredicted = false;
}

void MultiModelKalmanEstimator::combiningProbability(cv::Mat const &transitionProbability,
                                                     cv::Mat const &modelProbability,
                                                     cv::Mat &conditionalProbablity)
{
  auto nModels = modelProbability.size[0];

  for (std::size_t j = 0; j < nModels; ++j)
  {
    double sumProbability(0.0);

    for (std::size_t i = 0; i < nModels; ++i)
    {
      sumProbability += transitionProbability.at<double>(i, j) * modelProbability.at<double>(i, 0);
    }

    for (std::size_t i = 0; i < nModels; ++i)
    {
      auto const & pij = transitionProbability.at<double>(i, j);

      conditionalProbablity.at<double>(i, j) = pij * modelProbability.at<double>(i, 0) / sumProbability;
    }
  }
}

void MultiModelKalmanEstimator::interaction(std::vector<cv::Mat> const &states,
                                            std::vector<cv::Mat> const &statesErrorCovariance,
                                            cv::Mat const &conditionalProbablity,
                                            std::vector<cv::Mat> &covarianceEstimate,
                                            std::vector<cv::Mat> &stateEstimates)
{
  auto nModels = conditionalProbablity.size[0];
  auto stateSize = states[0].size[0];

  covarianceEstimate.clear();
  stateEstimates.clear();

  for (int j = 0; j < nModels; j++)
  {
    stateEstimates.push_back(cv::Mat::zeros(stateSize, 1, CV_64F));

    covarianceEstimate.push_back(cv::Mat::zeros(stateSize, stateSize, CV_64F));
  }

  for (std::size_t j = 0; j < nModels; ++j)
  {
    for (std::size_t i = 0; i < nModels; ++i)
    {
      stateEstimates[j] += states[i] * conditionalProbablity.at<double>(i, j);
    }
  }

  for (std::size_t j = 0; j < nModels; ++j)
  {
    for (std::size_t i = 0; i < nModels; ++i)
    {
      covarianceEstimate[j] += conditionalProbablity.at<double>(i, j)
        * (statesErrorCovariance[i] + ((states[i] - stateEstimates[j]) * ((states[i] - stateEstimates[j]).t())));
    }
  }
}

void inline expNormalize(const std::vector<double> &values, std::vector<double> &normalizedValues)
{
  normalizedValues.clear();

  double maxValue = *std::max_element(values.begin(), values.end());

  double sum = 0.0;
  for (auto const &value : values)
  {
    auto normalizedValue = std::exp(value - maxValue);
    sum += normalizedValue;

    normalizedValues.push_back(normalizedValue);
  }
  for (auto &value : normalizedValues)
  {
    value = value / sum;
  }
}


void MultiModelKalmanEstimator::updateModelProbability(cv::Mat const &measurement,
                                                       std::vector<cv::Mat> const &predictedMeasurements,
                                                       std::vector<cv::Mat> const &measurementNoiseCovariance,
                                                       cv::Mat &modelProbability)
{
  const auto &nModels = modelProbability.size[0];
  const auto &measurementSize = measurement.size[0];
  std::vector<double> lambda;
  std::vector<double> likelihood;
  std::vector<cv::Mat> measurementDifference;
  std::vector<cv::Mat> measurementDistance;

  auto previousModelProbability = modelProbability.clone();

  for (int j = 0; j < nModels; j++)
  {
    measurementDistance.push_back(cv::Mat::zeros(measurementSize, measurementSize, CV_64F));
    measurementDifference.push_back(cv::Mat::zeros(measurementSize, 1, CV_64F));
  }

  for (std::size_t j = 0; j < nModels; ++j)
  {
    measurementDifference[j] = measurement - predictedMeasurements[j];
  }

  for (std::size_t j = 0; j < nModels; ++j)
  {
    measurementDistance[j]
      = measurementDifference[j].t() * measurementNoiseCovariance[j].inv(cv::DECOMP_SVD) * measurementDifference[j];
  }
  // Calculate log likelihood per model
  for (std::size_t j = 0; j < nModels; ++j)
  {
    double det = determinant(2.0 * M_PI * measurementNoiseCovariance[j]);
    double logLikelihood = -0.5 * std::log(det) - 0.5 * measurementDistance[j].at<double>(0, 0);

    likelihood.push_back(logLikelihood);
  }

  // Normalize using exponential normalization and store it as lambda
  expNormalize(likelihood, lambda);

  // Calculate normalizing value for lambda
  double lambdaSum = 0.;
  for (std::size_t j = 0; j < nModels; ++j)
  {
    lambdaSum += lambda[j] * previousModelProbability.at<double>(j, 0);
  }

  // Update new model probability
  for (std::size_t j = 0; j < nModels; ++j)
  {
    // Constrain probability to be within a [min,max] bound
    auto probability = std::clamp(previousModelProbability.at<double>(j, 0) * lambda[j] / (lambdaSum), 1e-12, 1.0 - 1e-12);

    modelProbability.at<double>(j, 0) = probability;
  }
}

void MultiModelKalmanEstimator::combineStatesAndCovariances(std::vector<cv::Mat> const &states,
                                                            std::vector<cv::Mat> const &covariances,
                                                            cv::Mat const &modelProbability,
                                                            cv::Mat &combinedState,
                                                            cv::Mat &combinedCovariance)
{
  auto nModels = modelProbability.size[0];
  auto stateSize = states[0].size[0];

  combinedState = cv::Mat::zeros(stateSize, 1, CV_64F);
  combinedCovariance = cv::Mat::zeros(stateSize, stateSize, CV_64F);

  // First calculate the mean (combined state)
  for (std::size_t i = 0; i < nModels; ++i)
  {
    combinedState += states[i] * modelProbability.at<double>(i, 0);
  }

  // Use the combined state to calculate the combined covariance
  for (std::size_t i = 0; i < nModels; ++i)
  {
    combinedCovariance += modelProbability.at<double>(i, 0) * (covariances[i] + ((states[i] - combinedState) * ((states[i] - combinedState).t())));
  }
}

cv::Mat MultiModelKalmanEstimator::getModelProbability() const
{
  return mModelProbability;
}

cv::Mat MultiModelKalmanEstimator::getTransitionProbability() const
{
  return mTransitionProbability;
}

cv::Mat MultiModelKalmanEstimator::getConditionalProbability() const
{
  cv::Mat conditionalProbability = cv::Mat::zeros(mNumberOfModels, mNumberOfModels, CV_64F);
  combiningProbability(mTransitionProbability, mModelProbability, conditionalProbability);

  return conditionalProbability;
}

cv::Mat MultiModelKalmanEstimator::getKalmanFilterMeasurementCovariance(std::size_t j) const
{
  return mKalmanFilters[j]->getErrorCov().clone();
}
cv::Mat MultiModelKalmanEstimator::getKalmanFilterErrorCovariance(std::size_t j) const
{
  return mKalmanFilters[j]->getMeasurementCov().clone();
}

} // namespace tracking
} // namespace rv