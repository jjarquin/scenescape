# robot vision 1.2.0, changes from 1.1.0

### Added
- New fields in the TrackedObject which are updated after each correction step of the MultiModelKalmanEstimator
  - trackedTime
  - predictedTime
  - age
  - visualCertainty
  - associationProbability
- Extended the member variables of the TrackedObject type to include the Z component of each physical quantity (position, velocity, acceleration).
- (Experimental). Included Jerk variables (jx, jy, jz) in the object representation.

- functions related to the visual features management.
  - addVisualFeatures
  - getVisualFeatures
  - getVisualFeaturesMatrix
  - getVisualFeaturesSet

- Functions to calculate visual similarity and visual distance between visual feature vectors and a visualfeature vector and a set of visual feature vectors in the form of matrix
  - similarity(const Eigen::VectorXd & visualFeaturesA, const Eigen::VectorXd & visualFeaturesB)
  - similarity(const Eigen::VectorXd & visualFeatures, const Eigen::MatrixXd & visualFeaturesMatrix)
  - distance(const Eigen::VectorXd & visualFeaturesA, const Eigen::VectorXd & visualFeaturesB)
  - distance(const Eigen::VectorXd & visualFeatures, const  Eigen::MatrixXd & visualFeaturesMatrix)

- (Experimental) Added ConstantJerk motion model useful to model sudden break/accelerate motion profiles

- mahalanobis_distance function
- association_probability function

- New distance types, adjusted formulas and created new combined distance types.
  - Visual
  - VisualSpatial
  - VisualMultiClass
  - VisualSpatialMultiClass

- Added example implementation of our MultipleObjectTracker with python code (multiple_object_tracker_python_impl.py)

### Changed
- Changed C++ version from C++14 to C++17
- Changed Id type from int_32t to boost::uuids::uuid.
New Ids are created with the boost::uuids::random_generator instead of incrementally, the UUID boost::uuids::nil_uuid() is used now to represent an invalid or uninitiallized UUID.
- Moved all updates pertinent to thet TrackedObject to the following functions:
  - updateOnPrediction(double deltaT);
  - updateOnCorrection(const TrackedObject &measurement);
- Renamed turn rate from w to yawRate.


- Renamed MotionModel enum class to MotionModelType (only applicable to C++)
- Created base MotionModel class inherited by each custom MotionModel
- Changed default MultiModelKalmanEstimator models to MotionModelType::CV, MotionModelType::CA

- removed angle_difference function
- removed custom clamp function which is available in C++17
- Renamed distance types, adjusted formulas and created new combined distance types.
  - Euclidean -> Spatial
  - MultiClassEuclidean -> SpatialMultiClass
- Updated example for custom tracker in python


### Fixed
- Fixed a problem with the model probability normalization which affected the multimodel state estimation.

### Notes
- For changes to the Python API please check the API document.
