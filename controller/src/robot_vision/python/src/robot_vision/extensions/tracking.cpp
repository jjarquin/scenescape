// SPDX-FileCopyrightText: (C) 2019 - 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <opencv2/core.hpp>
#include <pybind11/chrono.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <rv/tracking/MultiModelKalmanEstimator.hpp>
#include <rv/tracking/MultipleObjectTracker.hpp>
#include <rv/tracking/TrackManager.hpp>
#include <rv/tracking/TrackTracker.hpp>
#include <rv/tracking/TrackedObject.hpp>
#include <rv/tracking/Classification.hpp>
#include <rv/tracking/VisualFeatures.hpp>
#include <chrono>
#include <vector>
#include <Eigen/Dense>

#include "uuid.hpp"

namespace py = pybind11;

PYBIND11_MODULE(tracking, tracking)
{
  tracking.doc() = R"pbdoc(
    Algorithms for tracking 3D objects
    -----------------------
    )pbdoc";

  py::class_<rv::tracking::ClassificationData>(tracking, "ClassificationData", "Helper class to initialize and get data from a class probability vector (numpy.array).")
     .def(py::init<>(), "Default constructor. The classes vector will default to ['Unknown'].")
     .def(py::init<std::vector<std::string> &>(),
          "Initialize ClassificationData with the given class list",
          py::arg("classes"))
     .def("classification", &rv::tracking::ClassificationData::classification,
          "Create a classification vector with the given class set to the corresponding probability.",
          py::arg("class"),
          py::arg("probability") = 1.0)
     .def("get_class", &rv::tracking::ClassificationData::getClass,
          "Returns the name of the class with the maximum probability.",
          py::arg("classification"))
     .def("get_class_index", &rv::tracking::ClassificationData::classIndex,
          "Returns the index for the given class name.",
          py::arg("class_name"))
     .def("unform_prior", &rv::tracking::ClassificationData::uniformPrior,
          "Generate a prior vector using the given probability for each class.",
          py::arg("probability"))
     .def("prior", &rv::tracking::ClassificationData::prior,
          "Generate a prior classification assigning the same probability for all clases.")
     .def_property("classes",
                  &rv::tracking::ClassificationData::getClasses,
                  &rv::tracking::ClassificationData::setClasses,
                  py::return_value_policy::take_ownership, "List of classes defined in this ClassificationData.");

  py::class_<rv::tracking::TrackedObject>(tracking, "TrackedObject",
     "A TrackedObject holds the object's state (position, orientation, velocity, acceleration, size) and properties. It provides interfaces to facilitate its use in state filtering and tracking.")
    .def(py::init<>(), "Default constructor. The classification probability is initialized as numpy.array([1.0])")
    .def_readwrite("x", &rv::tracking::TrackedObject::x, "Position 'x' in meters.")
    .def_readwrite("y", &rv::tracking::TrackedObject::y, "Position 'y' in meters.")
    .def_readwrite("z", &rv::tracking::TrackedObject::z, "Position 'z' in meters.")
    .def_readwrite("length", &rv::tracking::TrackedObject::length, "Object's length in meters.")
    .def_readwrite("width", &rv::tracking::TrackedObject::width, "Object's width in meters.")
    .def_readwrite("height", &rv::tracking::TrackedObject::height, "Object's height in meters.")
    .def_readwrite("yaw", &rv::tracking::TrackedObject::yaw, "Orientation about Z axis in radians.")
    .def_readwrite("yaw_rate", &rv::tracking::TrackedObject::yawRate, "Turn rate about Z axis in radians/s.")
    .def_readwrite("vx", &rv::tracking::TrackedObject::vx, "Velocity component 'x' (forward) in m/s.")
    .def_readwrite("vy", &rv::tracking::TrackedObject::vy, "Velocity component 'y' (left) in m/s.")
    .def_readwrite("vz", &rv::tracking::TrackedObject::vz, "Velocity component 'z' (up) in m/s.")
    .def_readwrite("ax", &rv::tracking::TrackedObject::ax, "Acceleration component 'x' (forward) in m/s^2.")
    .def_readwrite("ay", &rv::tracking::TrackedObject::ay, "Acceleration component 'y' (left) in m/s^2.")
    .def_readwrite("az", &rv::tracking::TrackedObject::az, "Acceleration component 'z' (up) in m/s^2.")
    .def_readwrite("jx", &rv::tracking::TrackedObject::jx, "Jerk component 'x' (forward) in m/s^3.")
    .def_readwrite("jy", &rv::tracking::TrackedObject::jy, "Jerk component 'y' (left) in m/s^3.")
    .def_readwrite("jz", &rv::tracking::TrackedObject::jz, "Jerk component 'z' (up) in m/s^3.")
    .def_readwrite("age", &rv::tracking::TrackedObject::age, "Time in seconds the object has been updated by a Kalman Estimator.")
    .def_readwrite("predicted_time", &rv::tracking::TrackedObject::predictedTime, "Time in seconds the object has been consecutively predicted by a Kalman Estimator.")
    .def_readwrite("tracked_time", &rv::tracking::TrackedObject::trackedTime, "Time in seconds the object has been consecutively tracked by a Kalman Estimator.")
    .def_readwrite("corrected", &rv::tracking::TrackedObject::corrected, "Returns True if the TrackedObject was the result of a correction step.")
    .def_readwrite("association_probability", &rv::tracking::TrackedObject::associationProbability, "Returns the last association probability computed during the correction step.")
    .def_readwrite("visual_certainty", &rv::tracking::TrackedObject::visualCertainty, "Returns the mean of the visual similarity calculation.")
    .def_property("uuid", &rv::tracking::TrackedObject::uuidString, &rv::tracking::TrackedObject::setUuidString, "Returns the uuid string.")
    .def("is_uuid_nil", &rv::tracking::TrackedObject::isUuidNil, "Whether the current uuid is nil/empty.")
    .def("isDynamic", &rv::tracking::TrackedObject::isDynamic, "Returns True if the TrackedObject is considered to be moving.")
    .def_readwrite("classification", &rv::tracking::TrackedObject::classification, "Returns a numpy array with classification probabilities.")
    .def("add_visual_features", &rv::tracking::TrackedObject::addVisualFeatures, "Set the current visual feature embeddding vector")
    .def("get_visual_features", &rv::tracking::TrackedObject::getVisualFeatures, "Get the current visual feature embeddding vector")
    .def("get_visual_features_matrix", &rv::tracking::TrackedObject::getVisualFeaturesMatrix, py::return_value_policy::reference_internal, "Get the current visual feature embeddding vector")
    .def_property("visual_features_set", &rv::tracking::TrackedObject::getVisualFeaturesSet,
                                          &rv::tracking::TrackedObject::setVisualFeaturesSet,
                                               "Read/write access to visual feature set")
    .def_readwrite("attributes", &rv::tracking::TrackedObject::attributes, "Dictionary of attributes. Note: only string types are supported.")
    .def_property("vector",
                  &rv::tracking::TrackedObject::getVectorXf,
                  &rv::tracking::TrackedObject::setVectorXf,
                  py::return_value_policy::take_ownership, "Returns this object's state vector as numpy array.")
    .def("state_vector",  &rv::tracking::TrackedObject::stateVector, "State vector 14x1, convert to numpy using np.array(tracked_object.state_vector).")
    .def("measurement_vector",  &rv::tracking::TrackedObject::measurementVector, "Measurement vector 7x1, convert to numpy using np.array(tracked_object.measurement_vector).")
    .def_readwrite("measurement_mean", &rv::tracking::TrackedObject::predictedMeasurementMean, "Returns this object's measurement vector as numpy array.")
    .def_readwrite("measurement_covariance", &rv::tracking::TrackedObject::predictedMeasurementCov, "Measurement covariance matrix, convert to numpy using np.array(tracked_object.measurement_covariance).")
    .def_readwrite("measurement_covariance_inv", &rv::tracking::TrackedObject::predictedMeasurementCovInv, "Measurement covariance matrix inverse, convert to numpy using np.array(tracked_object.measurement_covariance_inv).")
    .def_readwrite("error_covariance", &rv::tracking::TrackedObject::errorCovariance, "Error covariance matrix, convert to numpy using np.array(tracked_object.error_covariance).")
    .def("__repr__", &rv::tracking::TrackedObject::toString, "String representation.");


  py::class_<rv::tracking::MultiModelKalmanEstimator>(tracking, "MultiModelKalmanEstimator",
    "Implements the Interacting Multiple Model Kalman Estimator. The models can be selected during initialization. The method initialize() must be called before using the KalmanEstimator.")
    .def(py::init<>(), "Default constructor. The method initialize() must be called before using the KalmanEstimator.")
    .def(py::init<double, double>(), "Initialize with given parameters. The method initialize() must be called before using the KalmanEstimator.",
    py::arg("alpha") = 1.0,
    py::arg("beta") = 1.0)
    .def("initialize",
        &rv::tracking::MultiModelKalmanEstimator::initialize,
         "Initialize the MultiModelKalmanEstimator with the given tracked object.",
         py::arg("tracked_object"),
         py::arg("timestamp"),
         py::arg("process_noise") = 1e-6,
         py::arg("measurement_noise") = 1e-4,
         py::arg("init_state_covariance") = 1.,
         py::arg("motion_models") = std::vector<rv::tracking::MotionModelType>())
    .def("predict",
         py::overload_cast<double>(&rv::tracking::MultiModelKalmanEstimator::predict),
         "Predict the position at T+deltaT time.",
         py::arg("deltaT"))
    .def("predict",
         py::overload_cast<const std::chrono::system_clock::time_point &>(&rv::tracking::MultiModelKalmanEstimator::predict),
         "Predict the position at given timestamp.",
         py::arg("timestamp"))
    .def("correct",
         &rv::tracking::MultiModelKalmanEstimator::correct,
         "Update estimator with current measurement.",
         py::arg("measurement"))
    .def("timestamp", &rv::tracking::MultiModelKalmanEstimator::getTimestamp, "Read current timestamp.")
    .def("track",
         &rv::tracking::MultiModelKalmanEstimator::track,
         "Trigger the track step for the next timestamp.",
         py::arg("measurement"),
         py::arg("timestamp"))
    .def("current_state",
         &rv::tracking::MultiModelKalmanEstimator::currentState,
         "Returns the current filtered state.")
    .def("current_states",
         &rv::tracking::MultiModelKalmanEstimator::currentStates,
         "Returns the list of internal states.")
     .def("kalman_filter_error_covariance",
          &rv::tracking::MultiModelKalmanEstimator::getKalmanFilterErrorCovariance,
          "Get error covariance of the Nth kalman filter.", py::arg("n"))
     .def("kalman_filter_measurement_covariance",
          &rv::tracking::MultiModelKalmanEstimator::getKalmanFilterMeasurementCovariance,
          "Get measurement covariance of the Nth kalman filter.", py::arg("n"))
     .def_property_readonly("model_probability",
          &rv::tracking::MultiModelKalmanEstimator::getModelProbability,
          "Probability of following certain motion model.")
     .def_property_readonly("transition_probability",
          &rv::tracking::MultiModelKalmanEstimator::getTransitionProbability,
          "Transition probability from model a to model b.")
     .def_property_readonly("conditional_probability",
          &rv::tracking::MultiModelKalmanEstimator::getConditionalProbability,
          "Current conditional probability from model a to model b.");

  py::enum_<rv::tracking::MotionModelType>(tracking, "MotionModel", "MotionModel enum class.")
    .value("CV", rv::tracking::MotionModelType::CV, "Constant Velocity.")
    .value("CA", rv::tracking::MotionModelType::CA, "Constant Acceleration.")
    .value("CP", rv::tracking::MotionModelType::CP, "Constant Position.")
    .value("CJ", rv::tracking::MotionModelType::CJ, "Constant Jerk.")
    .value("CTRV", rv::tracking::MotionModelType::CTRV, "Constant Turn-Rate and Velocity.")
    .export_values();

  py::enum_<rv::tracking::DistanceType>(tracking, "DistanceType", "DistanceType enum class.")
    .value("Visual", rv::tracking::DistanceType::Visual,
     "Use the cosine distance on the visual features.")
    .value("Spatial", rv::tracking::DistanceType::Spatial,
     "Standard euclidean distance that considers the object position (x,y,z).")
    .value("VisualSpatial", rv::tracking::DistanceType::VisualSpatial,
     "Scaled euclidean metric distance. It is scaled by visual distance between objects.")
    .value("VisualMultiClass", rv::tracking::DistanceType::VisualMultiClass,
     "The sum of the multiclass probability conflict and the visual distance")
    .value("SpatialMultiClass", rv::tracking::DistanceType::SpatialMultiClass,
     "Scaled euclidean metric distance. It is scaled by the multiclass probability difference.")
    .value("VisualSpatialMultiClass", rv::tracking::DistanceType::VisualSpatialMultiClass,
     "Scaled euclidean metric distance. It is scaled by the sum of the multiclass probability conflict and the visual distance.")
    .value("Mahalanobis", rv::tracking::DistanceType::Mahalanobis,
     "Mahalanobis distance that considers the objects measurement vector.")
    .export_values();

  py::class_<rv::tracking::TrackManagerConfig>(tracking, "TrackManagerConfig", "Holds all the configuration parameters used by the TrackManager.")
    .def(py::init<>(), "Initialize TrackManagerConfig with default parameters.")
    .def_readwrite("non_measurement_frames_dynamic", &rv::tracking::TrackManagerConfig::mNonMeasurementFramesDynamic,
     "Sets the maximum number of non measurement frames for a dynamic object. The track will be removed if it is not seen for given number of frames.")
    .def_readwrite("non_measurement_frames_static", &rv::tracking::TrackManagerConfig::mNonMeasurementFramesStatic,
     "Sets the maximum number of non measurement frames for a static object. The track will be suspended if it is not seen for given number of frames.")
    .def_readwrite("max_number_of_unreliable_frames", &rv::tracking::TrackManagerConfig::mMaxNumberOfUnreliableFrames,
     "Number of frames to measure an object before considering it a reliable object.")
    .def_readwrite("reactivation_frames", &rv::tracking::TrackManagerConfig::mReactivationFrames,
     "Number of frames to measure a suspended object before reactivating the track.")
    .def_readwrite("default_process_noise", &rv::tracking::TrackManagerConfig::mDefaultProcessNoise,
     "Default process noise passed to the KalmanEstimator init function.")
    .def_readwrite("default_measurement_noise", &rv::tracking::TrackManagerConfig::mDefaultMeasurementNoise,
     "Default measurement noise passed to the KalmanEstimator init function.")
    .def_readwrite("init_state_covariance", &rv::tracking::TrackManagerConfig::mInitStateCovariance,
     "Default init state covariance passed to the KalmanEstimator init function.")
    .def_readwrite("motion_models", &rv::tracking::TrackManagerConfig::mMotionModels,
     "List of motion models to use. It defaults to [CV, CA, CTRV]")
    .def("__repr__", &rv::tracking::TrackManagerConfig::toString, "String representation");


     py::class_<rv::tracking::TrackManager>(tracking, "TrackManager",
      "Track management system for multiple objects, it holds databases of the current objects on the scene and facilitates updates of multiple objects via uuid queries.")
    .def(py::init<>(), "Construct with default config")
    .def(py::init<const rv::tracking::TrackManagerConfig &>(), "Construct with given config", py::arg("track_manager_config"))
    .def(py::init<bool>(),
     "Construct with default config. Set auto_id_generation to False to use the already assigned track_id instead.",
     py::arg("auto_id_generation"))
    .def(py::init<const rv::tracking::TrackManagerConfig &, bool>(),
     "Construct with default config. Set auto_id_generation to False to use the already assigned track_id instead.",
     py::arg("track_manager_config"), py::arg("auto_id_generation"))
    .def("create_track",
         &rv::tracking::TrackManager::createTrack,
         "Create a new track, returns object uuid of new track.",
         py::arg("object"),
         py::arg("timestamp"))
    .def("predict",
         py::overload_cast<const double>(&rv::tracking::TrackManager::predict),
         "Predict at T+deltaT time.",
         py::arg("deltaT"))
    .def("predict",
         py::overload_cast<const std::chrono::system_clock::time_point &>(&rv::tracking::TrackManager::predict),
         "Predict at the given timestamp.",
         py::arg("timestamp"))
    .def("set_measurement",
         &rv::tracking::TrackManager::setMeasurement,
         "Create a new track, returns object uuid of new track.",
         py::arg("uuid"),
         py::arg("measurement"))
     .def("correct", &rv::tracking::TrackManager::correct, "Trigger state correction for all tracks.")
     .def("get_tracks", &rv::tracking::TrackManager::getTracks, "returns a list of all active tracks.")
     .def("get_reliable_tracks",
          &rv::tracking::TrackManager::getReliableTracks,
          "Returns a list of all tracks classified as reliable.")
     .def("get_unreliable_tracks",
          &rv::tracking::TrackManager::getUnreliableTracks,
          "Returns a list of all tracks classified as unreliable.")
     .def("get_suspended_tracks",
          &rv::tracking::TrackManager::getSuspendedTracks,
          "Returns a list of suspended tracks. Static objects that have not been visible for config.non_measurement_frames_static frames.")
     .def("get_drifting_tracks",
          &rv::tracking::TrackManager::getDriftingTracks,
          "Returns a list of tracks in risk of being deleted. Objects that have not been visible for config.non_measurement_frames_dynamic / 2.")
     .def("get_track",
          &rv::tracking::TrackManager::getTrack,
          "Returns the TrackedObject stored for the given uuid.",
          py::arg("uuid"))
     .def("get_kalman_estimator",
          &rv::tracking::TrackManager::getKalmanEstimator,
          "Returns the MultiModelKalmanEstimator  stored for the given uuid.",
          py::arg("uuid"))
     .def("has_uuid",
         &rv::tracking::TrackManager::hasUuid,
         "Check wether the given Id is registered in the track manager.",
         py::arg("uuid"))
     .def("delete_track",
         &rv::tracking::TrackManager::deleteTrack,
         "Delete the given track uuid from the track manager.",
         py::arg("uuid"))
     .def("suspend_track",
         &rv::tracking::TrackManager::suspendTrack,
          "Set the given track uuid as suspended.",
         py::arg("uuid"))
     .def("reactivate_track",
         &rv::tracking::TrackManager::reactivateTrack,
         "Move a suspended track uuid to active tracks.",
         py::arg("uuid"))
     .def("is_reliable",
         &rv::tracking::TrackManager::isReliable,
         "Check whether the given track uuid is reliable.",
         py::arg("uuid"))
     .def("is_suspended",
         &rv::tracking::TrackManager::isSuspended,
         "Check whether the given track uuid is suspended.",
         py::arg("uuid"))
     .def_property_readonly("config", &rv::tracking::TrackManager::getConfig, "Current track manager configuration");

  py::class_<rv::tracking::MultipleObjectTracker>(tracking, "MultipleObjectTracker",
     "Multiple Object Tracking algorithm using the TrackManager in the background. It performs an association step using the Gated Hungarian matcher.")
    .def(py::init<>(), "Default constructor, use default config parameters.")
    .def(py::init<const rv::tracking::TrackManagerConfig &>(),
      "Use the given config parameters for the track manager.",
      py::arg("track_manager_config"))
    .def(py::init<const rv::tracking::TrackManagerConfig &, const rv::tracking::DistanceType &, double>(),
     "Use the given config parameters for the track manager. Initialize with the given distance type and threshold.",
      py::arg("track_manager_config"),
      py::arg("distance_type"),
      py::arg("distance_threshold"))
    .def("track",
         py::overload_cast<std::vector<rv::tracking::TrackedObject>, const std::chrono::system_clock::time_point &, double>(&rv::tracking::MultipleObjectTracker::track),
         "Trigger the track step for the next timestamp. Use the default distance type and threshold.",
         py::arg("objects"),
         py::arg("timestamp"),
         py::arg("probability_threshold") = 0.5)
    .def("track",
         py::overload_cast<std::vector<rv::tracking::TrackedObject>, const std::chrono::system_clock::time_point &, const rv::tracking::DistanceType &, double, double>(&rv::tracking::MultipleObjectTracker::track),
         "Trigger the track step for the next timestamp. Run match() with the given distance type and threshold.",
         py::arg("objects"),
         py::arg("timestamp"),
         py::arg("distance_type"),
         py::arg("distance_threshold"),
         py::arg("probability_threshold") = 0.5)
    .def("timestamp", &rv::tracking::MultipleObjectTracker::getTimestamp, "Read current timestamp.")
    .def("get_tracks", &rv::tracking::MultipleObjectTracker::getTracks, "Returns a list of all active tracks")
    .def("get_reliable_tracks",
         &rv::tracking::MultipleObjectTracker::getReliableTracks,
         "Returns a list of all active reliable tracks.");

  py::class_<rv::tracking::TrackTracker>(tracking,
  "TrackTracker", "Multiple Object Tracking algorithm using the TrackManager in the background. This tracker does not perform any association step, instead it relies on the object's uuid for association.")
    .def(py::init<>(), "Default constructor, use default config parameters.")
    .def(py::init<const rv::tracking::TrackManagerConfig &>(),
      "Use the given config parameters for the track manager.",
      py::arg("track_manager_config"))
    .def("track",
         &rv::tracking::TrackTracker::track,
         "Trigger the track step for the next timestamp. Note: The objects must have an uuid already assigned.",
         py::arg("tracked_objects"),
         py::arg("timestamp"))
    .def("timestamp", &rv::tracking::TrackTracker::getTimestamp, "Read current timestamp.")
    .def("get_tracks", &rv::tracking::TrackTracker::getTracks, "Returns a list of all active tracks.")
    .def("get_reliable_tracks",
         &rv::tracking::TrackTracker::getReliableTracks,
         "Returns a list of all active reliable tracks.");

     tracking.def("match", [](const std::vector<rv::tracking::TrackedObject> &measurements, const std::vector<rv::tracking::TrackedObject> &tracks, const rv::tracking::DistanceType &distanceType, double threshold) {
          std::vector<std::pair<size_t, size_t>> assignments;
          std::vector<size_t> unassignedTracks;
          std::vector<size_t> unassignedObjects;
          rv::tracking::match(measurements, tracks, assignments,  unassignedTracks, unassignedObjects, distanceType, threshold);

          return std::tuple<std::vector<std::pair<size_t, size_t>>,std::vector<size_t>,  std::vector<size_t>> (assignments, unassignedTracks, unassignedObjects);
          },
          "Match measurements to tracks. Returns a tuple containing (track and object index, unassigned tracks, unassigned objects).",
          py::arg("tracks"),
          py::arg("measurements"),
          py::arg("distance_type") = rv::tracking::DistanceType::Spatial,
          py::arg("threshold") = 1.0);

     tracking.def("mahalanobis_distance",
        &rv::tracking::calculateMahalanobisDistance,
        "Calculates the mahalanobis distance between the predicted measurement of the track and the new measurement",
        py::arg("track"), py::arg("measurement"));

     tracking.def("association_probability",
        &rv::tracking::calculateAssociationProbability,
        "Calculates the association probability between the predicted measurement of the track and the new measurement",
        py::arg("track"), py::arg("measurement"));

     tracking.def("delta_theta",
        &rv::deltaTheta,
        "Calculate the difference between two angles, considering possible jumps of pi.");

     py::module visual = tracking.def_submodule("visual", "Operations applied on class visual vector embeddings.");
    visual.def("similarity", py::overload_cast<const Eigen::VectorXd&, const Eigen::VectorXd&>(&rv::tracking::visual::similarity),
    "Calculate the visual similarity (1 - cosine distance) between two visual vectors.", py::arg("visual_features_a"), py::arg("visual_features_b"));
    visual.def("similarity", py::overload_cast<const Eigen::VectorXd&, const Eigen::MatrixXd&>(&rv::tracking::visual::similarity),
    "Calculate the visual similarity (1 - cosine distance) between a visual vector and a matrix of visual vectors.", py::arg("visual_features"), py::arg("visual_features_matrix"));
    visual.def("distance", py::overload_cast<const Eigen::VectorXd&, const Eigen::VectorXd&>(&rv::tracking::visual::distance),
    "Calculate the visual distance (cosine distance) between two visual vectors.", py::arg("visual_features_a"), py::arg("visual_features_b"));
    visual.def("distance", py::overload_cast<const Eigen::VectorXd&, const Eigen::MatrixXd&>(&rv::tracking::visual::distance),
    "Calculate the visual distance (cosine distance) between two visual vectors.", py::arg("visual_features"), py::arg("visual_features_matrix"));

     py::module classification = tracking.def_submodule("classification", "Operations applied on class probability vectors.");

     classification.def("distance", &rv::tracking::classification::distance,
          "Calculate the distance between two classification probability vectors.",
          py::arg("classification_a"),
          py::arg("classification_b"))
     .def("similarity", &rv::tracking::classification::similarity,
          "Calculate how similar two given classifications are.",
          py::arg("classification_a"),
          py::arg("classification_b"))
     .def("combine", &rv::tracking::classification::combine,
          "Combine probability vectors using multiclass bayes update.",
          py::arg("classification_a"),
          py::arg("classification_b"));

}
