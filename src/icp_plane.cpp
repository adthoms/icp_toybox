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

Eigen::Matrix4d ICP_PLANE::computeTransform(const PointCloud& source_cloud, const PointCloud& target_cloud) {
  switch (solver_type_) {
  case SolverType::LeastSquares:
    return computeTransformLeastSquares(source_cloud, target_cloud);
  case SolverType::LeastSquaresUsingCeres:
    return computeTransformLeastSquaresUsingCeres(source_cloud, target_cloud);
  default:
    return computeTransformLeastSquares(source_cloud, target_cloud);
  }
}

Eigen::Matrix4d ICP_PLANE::computeTransformLeastSquares(const PointCloud& source_cloud,
                                                        const PointCloud& target_cloud) {
  Eigen::Matrix<double, 6, 6> JTJ;
  Eigen::Matrix<double, 6, 1> JTr;
  JTJ.setZero();
  JTr.setZero();

  int num_corr = correspondence_set_.size();

#pragma omp parallel
  {
    Eigen::Matrix<double, 6, 6> JTJ_private;
    Eigen::Matrix<double, 6, 1> JTr_private;
    JTJ_private.setZero();
    JTr_private.setZero();
#pragma omp for nowait
    for (int i = 0; i < num_corr; ++i) {
      const auto& p = source_cloud.points_[correspondence_set_[i].first];
      const auto& q = target_cloud.points_[correspondence_set_[i].second];
      const auto& q_norm = target_cloud.normals_[correspondence_set_[i].second];
      auto [JTJi, JTri] = compute_JTJ_and_JTr(p, q, q_norm);
      double wi = 1.0;
      JTJ_private += wi * JTJi;
      JTr_private += wi * JTri;
    }
#pragma omp critical
    {
      JTJ += JTJ_private;
      JTr += JTr_private;
    }
  }

  // lambda expression for solving {V, Σ} of H = V Σ V^T where H is PD
  auto compute_svd = [](const Eigen::Matrix3d& H) -> std::pair<Eigen::Matrix3d, Eigen::Vector3d> {
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeThinU | Eigen::ComputeThinV);
    return std::make_pair(svd.matrixU(), svd.singularValues());
  };

  // compute eigenvectors and eigenvalues of the hessian from P2P-ICP
  const Eigen::Matrix3d& H_tt = JTJ.block<3, 3>(0, 0);
  const Eigen::Matrix3d& H_rr = JTJ.block<3, 3>(3, 3);
  const auto [V_t, Sigma_t] = compute_svd(H_tt);
  const auto [V_r, Sigma_r] = compute_svd(H_rr);

  // count number of constraints
  const int num_trans_constraints = std::count_if(Sigma_t.data(), Sigma_t.data() + 3, [this](double v) {
    // std::cout << "Sigma_t " << v << std::endl;
    return v < eigenvalue_translation_threshold_;
  });
  const int num_rot_constraints = std::count_if(Sigma_r.data(), Sigma_r.data() + 3, [this](double v) {
    // std::cout << "Sigma_r " << v << std::endl;
    return v < eigenvalue_rotation_threshold_;
  });
  const int c = num_trans_constraints + num_rot_constraints;

  // construct constraint matrix
  Eigen::MatrixXd C = Eigen::MatrixXd::Zero(c, 6);
  if (c > 0) {
    // check if v_j ∈ {V_t, V_r} is in the null space of V_t and V_r by thresholding the eigenvalues
    int idx = 0;
    for (int j = 0; j < 3; ++j) {
      if (Sigma_t(j) < eigenvalue_translation_threshold_) {
        C.block<1, 3>(idx, 0) = V_t.col(j).transpose();
        idx++;
      }
    }
    for (int j = 0; j < 3; ++j) {
      if (Sigma_r(j) < eigenvalue_rotation_threshold_) {
        C.block<1, 3>(idx, 3) = V_r.col(j).transpose();
        idx++;
      }
    }
  }

  // init augmented hessian and gradient
  Eigen::MatrixXd H_aug;
  Eigen::VectorXd g_aug;

  // resize hessian and gradient to accommodate constraints
  H_aug.resize(6 + c, 6 + c);
  g_aug.resize(6 + c);
  H_aug.setZero();
  g_aug.setZero();

  // augment hessian to satisfy KKT equations of equality-constrained P2P-ICP optimization
  H_aug.block<6, 6>(0, 0) = JTJ;
  H_aug.block(0, 6, 6, c) = C.transpose();
  H_aug.block(6, 0, c, 6) = C;
  g_aug.head<6>() = JTr;

  // solve for x* = {t*, r*}
  Eigen::VectorXd x_opt = H_aug.ldlt().solve(-g_aug);

  // construct the transformation matrix from x*
  Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
  transform.block<3, 3>(0, 0) = createRotationMatrix(x_opt.segment<3>(3));
  transform.block<3, 1>(0, 3) = x_opt.head(3);
  return transform;
}

std::pair<Eigen::Matrix<double, 6, 6>, Eigen::Matrix<double, 6, 1>>
ICP_PLANE::compute_JTJ_and_JTr(const Eigen::Vector3d& p, const Eigen::Vector3d& q, const Eigen::Vector3d& q_norm) {
  Eigen::Matrix<double, 6, 1> JT;
  double r;
  JT.block<3, 1>(0, 0) = q_norm;
  JT.block<3, 1>(3, 0) = p.cross(q_norm);
  r = (p - q).transpose() * q_norm;
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
