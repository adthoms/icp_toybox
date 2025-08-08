#include <algorithm>

#include <Eigen/Dense>

#include <omp.h>

#include "ICP/icp_plane.hpp"
#include "ICP/utils.hpp"

bool ICP_PLANE::checkValidity(PointCloud& source_cloud, PointCloud& target_cloud) {
  if (source_cloud.IsEmpty() || target_cloud.IsEmpty()) {
    LOG(WARNING) << "source cloud or target cloud are empty!";
    return false;
  }

  if (!target_cloud.HasNormals()) {
    LOG(WARNING) << "point to plane ICP needs normal points in the target pointcloud.";
    return false;
  }

  return true;
}

std::pair<Eigen::Matrix<double, 6, 6>, Eigen::Matrix<double, 6, 1>>
ICP_PLANE::compute_JTJ_and_JTr(const PointCloud& source_cloud, const PointCloud& target_cloud, int i) {
  const auto& p = source_cloud.points_[correspondence_set_[i].first];
  const auto& q = target_cloud.points_[correspondence_set_[i].second];
  const auto& q_norm = target_cloud.normals_[correspondence_set_[i].second];

  Eigen::Matrix<double, 6, 1> JT;
  JT.block<3, 1>(0, 0) = q_norm;
  JT.block<3, 1>(3, 0) = p.cross(q_norm);
  double r = (p - q).transpose() * q_norm;
  return std::make_pair(JT * JT.transpose(), JT * r);
}

Eigen::Matrix4d ICP_PLANE::computeTransformLeastSquaresUsingCeres(const PointCloud& source_cloud,
                                                                  const PointCloud& target_cloud) {
  optimizer_->clear();
  Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity();
  Eigen::Vector3d translation = Eigen::Vector3d::Zero();
  const auto& source_points = source_cloud.points_;
  const auto& target_points = target_cloud.points_;
  const auto& target_norms = target_cloud.normals_;
  for (std::size_t i = 0; i < correspondence_set_.size(); ++i) {
    auto [source_idx, target_idx] = correspondence_set_[i];
    optimizer_->addPointToPlaneResidual(
        source_points[source_idx], target_points[target_idx], target_norms[target_idx], rotation, translation);
  }
  optimizer_->solve();

  Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
  transform.block<3, 3>(0, 0) = rotation.toRotationMatrix();
  transform.block<3, 1>(0, 3) = translation;

  return transform;
}
