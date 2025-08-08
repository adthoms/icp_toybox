#pragma once

#include <memory>

#include <Eigen/Dense>
#include <open3d/Open3D.h>
#include <open3d/geometry/PointCloud.h>
#include <open3d/geometry/KDTreeFlann.h>

#include "ICP/ceres_optimizer.hpp"

class ICP_BASE {
public:
  using PointCloud = open3d::geometry::PointCloud;
  using KDTree = open3d::geometry::KDTreeFlann;
  using PointCloudPtr = typename std::shared_ptr<PointCloud>;
  using KDTreePtr = typename std::shared_ptr<KDTree>;

  enum class SolverType { LeastSquares, LeastSquaresUsingCeres };

  ICP_BASE() {}

  void align(PointCloud& source_cloud, PointCloud& target_cloud);

  void setIteration(int iteration) {
    max_iteration_ = iteration;
  }
  void setMaxCorrespondenceDist(double dist) {
    max_corres_dist_ = dist;
  }

  void setEigenvalueRotationThreshold(double threshold) {
    eigenvalue_rotation_threshold_ = threshold;
  }

  void setEigenvalueTranslationThreshold(double threshold) {
    eigenvalue_translation_threshold_ = threshold;
  }

  // not converged if abs(current rmse of corresponded points - prev) > threshold
  void setRelativeMatchingRmseThreshold(double threshold) {
    relative_matching_rmse_threshold_ = threshold;
  }

  // not converged if squared norm of translation > threshold
  void setTranslationThreshold(double threshold) {
    translation_threshold_ = threshold;
  }

  // not converged if cos(theta) < threshold
  void setRotationThreshold(double threshold) {
    cos_theta_threshold_ = threshold;
  }

  Eigen::Matrix4d getResultTransform() const {
    return total_transform_;
  }
  bool hasConverged() const {
    return converged_;
  }

protected:
  virtual bool checkValidity(PointCloud& source_cloud, PointCloud& target_cloud) = 0;
  virtual std::pair<Eigen::Matrix<double, 6, 6>, Eigen::Matrix<double, 6, 1>>
  compute_JTJ_and_JTr(const PointCloud& source_cloud, const PointCloud& target_cloud, int i) = 0;
  virtual Eigen::Matrix4d computeTransformLeastSquaresUsingCeres(const PointCloud& source_cloud,
                                                                 const PointCloud& target_cloud) = 0;

  void correspondenceMatching(const PointCloud& tmp_cloud);
  Eigen::Matrix4d computeTransform(const PointCloud& source_cloud, const PointCloud& target_cloud);
  Eigen::Matrix4d computeTransformLeastSquares(const PointCloud& source_cloud, const PointCloud& target_cloud);
  void computeAugmentedHessianAndGradient(const Eigen::Matrix<double, 6, 6>& H,
                                          const Eigen::Matrix<double, 6, 1>& g,
                                          Eigen::MatrixXd& H_aug,
                                          Eigen::VectorXd& g_aug);
  bool convergenceCheck(const Eigen::Matrix4d& transform_iter) const;

  SolverType solver_type_;
  std::unique_ptr<CeresOptimizer> optimizer_;
  KDTreePtr tree_ = nullptr;
  Eigen::Matrix4d total_transform_ = Eigen::Matrix4d::Identity();
  std::vector<std::pair<int, int>> correspondence_set_;
  int max_iteration_ = 30;
  bool converged_ = false;
  double max_corres_dist_ = 10.0;
  double relative_matching_rmse_threshold_ = 1e-6;
  double translation_threshold_ = 1e-6;
  double cos_theta_threshold_ = 1.0 - 1e-5;
  double matching_rmse_ = std::numeric_limits<double>::max();
  double matching_rmse_prev_ = std::numeric_limits<double>::max();
  double eigenvalue_rotation_threshold_ = 1e-6;
  double eigenvalue_translation_threshold_ = 1e-6;
};
