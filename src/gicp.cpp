#include <algorithm>

#include <Eigen/Dense>

#include <omp.h>

#include "ICP/gicp.hpp"
#include "ICP/utils.hpp"

bool GICP::checkValidity(PointCloud& source_cloud, PointCloud& target_cloud) {
  if (source_cloud.IsEmpty() || target_cloud.IsEmpty()) {
    LOG(WARNING) << "source cloud or target cloud are empty!";
    return false;
  }

  if (!source_cloud.HasCovariances()) {
    if (source_cloud.HasNormals()) {
      LOG(INFO) << "compute source cloud covariances from normals";
      computeCovariancesFromNormals(source_cloud);
    } else {
      LOG(WARNING) << "source cloud needs normals or covariances";
      return false;
    }
  }

  if (!target_cloud.HasCovariances()) {
    if (target_cloud.HasNormals()) {
      LOG(INFO) << "compute target cloud covariances from normals";
      computeCovariancesFromNormals(target_cloud);
    } else {
      LOG(WARNING) << "target cloud needs normals or covariances";
      return false;
    }
  }

  return true;
}

std::pair<Eigen::Matrix6d, Eigen::Vector6d>
GICP::compute_JTJ_and_JTr(const PointCloud& source_cloud, const PointCloud& target_cloud, int i) {
  const auto& p = source_cloud.points_[correspondence_set_[i].first];
  const auto& q = target_cloud.points_[correspondence_set_[i].second];
  const auto& p_cov = source_cloud.covariances_[correspondence_set_[i].first];
  const auto& q_cov = target_cloud.covariances_[correspondence_set_[i].second];

  Eigen::Matrix<double, 3, 6> J;
  J.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
  J.block<3, 3>(0, 3) = -skewSymmetric(p);

  const Eigen::Matrix3d C_inv = (p_cov + q_cov).inverse();
  const Eigen::Matrix6d JTJ = J.transpose() * C_inv * J;
  const Eigen::Vector6d JTr = J.transpose() * C_inv * (p - q);
  return std::make_pair(JTJ, JTr);
}

Eigen::Matrix4d GICP::computeTransformLeastSquaresUsingCeres(const PointCloud& source_cloud,
                                                             const PointCloud& target_cloud) {
  optimizer_->clear();
  Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity();
  Eigen::Vector3d translation = Eigen::Vector3d::Zero();
  const auto& source_points = source_cloud.points_;
  const auto& source_covs = source_cloud.covariances_;
  const auto& target_points = target_cloud.points_;
  const auto& target_covs = target_cloud.covariances_;
  for (std::size_t i = 0; i < correspondence_set_.size(); ++i) {
    auto [source_idx, target_idx] = correspondence_set_[i];
    optimizer_->addGICPResidual(source_points[source_idx],
                                source_covs[source_idx],
                                target_points[target_idx],
                                target_covs[target_idx],
                                rotation,
                                translation);
  }
  optimizer_->solve();

  Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
  transform.block<3, 3>(0, 0) = rotation.toRotationMatrix();
  transform.block<3, 1>(0, 3) = translation;

  return transform;
}

void GICP::computeCovariancesFromNormals(PointCloud& cloud) {
  int num_points = cloud.normals_.size();
  const Eigen::Matrix3d Cov = Eigen::Vector3d(cov_epsilon_, 1.0, 1.0).asDiagonal();
  cloud.covariances_.resize(num_points);
#pragma omp parallel for
  for (int i = 0; i < num_points; ++i) {
    Eigen::Matrix3d R = getRotationFromNormal(cloud.normals_[i]);
    cloud.covariances_[i] = R * Cov * R.transpose();
  }
}

Eigen::Matrix3d GICP::getRotationFromNormal(const Eigen::Vector3d& normal) {
  Eigen::Vector3d e1{1.0, 0.0, 0.0};

  Eigen::Vector3d v = e1.cross(normal);
  double cos = e1.dot(normal);
  if (1.0 + cos < 1e-3)
    return Eigen::Matrix3d::Identity();

  Eigen::Matrix3d v_skew;
  v_skew << 0.0, -v(2), v(1), v(2), 0.0, -v(0), -v(1), v(0), 0.0;
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity() + v_skew + (1.0 / (1.0 + cos)) * (v_skew * v_skew);

  return R;
}
