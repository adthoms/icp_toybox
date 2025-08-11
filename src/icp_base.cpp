#include <algorithm>

#include <omp.h>

#include "ICP/icp_base.hpp"
#include "ICP/utils.h"

void ICP_BASE::align(PointCloud& source_cloud, PointCloud& target_cloud) {
  if (!checkValidity(source_cloud, target_cloud)) {
    return;
  }

  total_transform_ = Eigen::Matrix4d::Identity();
  PointCloud tmp_cloud = source_cloud;
  tree_ = std::make_shared<KDTree>(target_cloud);
  correspondence_set_.reserve(source_cloud.points_.size());

  int64_t t_corr = 0, t_comp = 0, t_trans = 0;

  for (int i = 0; i < max_iteration_; ++i) {
    auto t_0 = std::chrono::high_resolution_clock::now();
    correspondenceMatching(tmp_cloud);
    auto t_1 = std::chrono::high_resolution_clock::now();
    Eigen::Matrix4d transform = computeTransform(tmp_cloud, target_cloud);
    auto t_2 = std::chrono::high_resolution_clock::now();
    total_transform_ = transform * total_transform_;
    if (convergenceCheck(transform)) {
      t_corr += std::chrono::duration_cast<std::chrono::microseconds>(t_1 - t_0).count();
      t_comp += std::chrono::duration_cast<std::chrono::microseconds>(t_2 - t_1).count();
      LOG(INFO) << "ICP converged! iter = " << (i + 1);
      converged_ = true;
      break;
    }
    tmp_cloud.Transform(transform);
    auto t_3 = std::chrono::high_resolution_clock::now();

    t_corr += std::chrono::duration_cast<std::chrono::microseconds>(t_1 - t_0).count();
    t_comp += std::chrono::duration_cast<std::chrono::microseconds>(t_2 - t_1).count();
    t_trans += std::chrono::duration_cast<std::chrono::microseconds>(t_3 - t_2).count();
  }

  LOG(INFO) << "correspondence elapsed time : " << t_corr << " micro seconds";
  LOG(INFO) << "compute transform elapsed time : " << t_comp << " micro seconds";
  LOG(INFO) << "transform/check elapsed time : " << t_trans << " micro seconds";
}

void ICP_BASE::correspondenceMatching(const PointCloud& tmp_cloud) {
  correspondence_set_.clear();
  matching_rmse_prev_ = matching_rmse_;
  matching_rmse_ = 0.0;
  auto& points = tmp_cloud.points_;

#pragma omp parallel
  {
    double euclidean_error_private = 0.0;
    std::vector<std::pair<int, int>> correspondence_set_private;
    std::vector<int> indices(1);
    std::vector<double> distances2(1);
#pragma omp for nowait
    for (int i = 0; i < static_cast<int>(points.size()); ++i) {
      if (tree_->SearchHybrid(points[i], max_corres_dist_, 1, indices, distances2) > 0) {
        correspondence_set_private.emplace_back(i, indices[0]);
        euclidean_error_private += distances2[0];
      }
    }

#pragma omp critical
    {
      matching_rmse_ += euclidean_error_private;
      for (std::size_t i = 0; i < correspondence_set_private.size(); ++i)
        correspondence_set_.push_back(correspondence_set_private[i]);
    }
  }

  matching_rmse_ = std::sqrt(matching_rmse_ / static_cast<double>(correspondence_set_.size()));
}

Eigen::Matrix4d ICP_BASE::computeTransform(const PointCloud& source_cloud, const PointCloud& target_cloud) {
  switch (solver_type_) {
  case SolverType::LeastSquares:
    return computeTransformLeastSquares(source_cloud, target_cloud);
  case SolverType::LeastSquaresUsingCeres:
    return computeTransformLeastSquaresUsingCeres(source_cloud, target_cloud);
  default:
    return computeTransformLeastSquares(source_cloud, target_cloud);
  }
}

Eigen::Matrix4d ICP_BASE::computeTransformLeastSquares(const PointCloud& source_cloud, const PointCloud& target_cloud) {
  const int num_corr = correspondence_set_.size();
  Eigen::Matrix6d H = Eigen::Matrix6d::Zero();
  Eigen::Vector6d g = Eigen::Vector6d::Zero();

  // construct Hessian and gradient
#pragma omp parallel
  {
    Eigen::Matrix6d H_private = Eigen::Matrix6d::Zero();
    Eigen::Vector6d g_private = Eigen::Vector6d::Zero();
#pragma omp for nowait
    for (int i = 0; i < num_corr; ++i) {
      Eigen::Matrix6d Hi;
      Eigen::Vector6d gi;
      computeHessianAndGradient(source_cloud, target_cloud, i, Hi, gi);
      H_private += Hi;
      g_private += gi;
    }
#pragma omp critical
    {
      H += H_private;
      g += g_private;
    }
  }

  // apply active geometric degeneracy mitigation step
  Eigen::MatrixXd H_aug;
  Eigen::VectorXd g_aug;
  computeAugmentedHessianAndGradient(H, g, H_aug, g_aug);

  // solve for x* = {t*, r*}
  const Eigen::VectorXd x_opt = H_aug.ldlt().solve(-g_aug);

  // construct the transformation matrix from x*
  Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
  transform.block<3, 3>(0, 0) = createRotationMatrix(x_opt.segment<3>(3));
  transform.block<3, 1>(0, 3) = x_opt.head(3);
  return transform;
}

void ICP_BASE::computeAugmentedHessianAndGradient(const Eigen::Matrix6d& H,
                                                  const Eigen::Vector6d& g,
                                                  Eigen::MatrixXd& H_aug,
                                                  Eigen::VectorXd& g_aug) {
  // lambda expression for solving {V, Σ} of H = V Σ V^T where H is PD
  auto compute_svd = [](const Eigen::Matrix3d& H) -> std::pair<Eigen::Matrix3d, Eigen::Vector3d> {
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeThinU | Eigen::ComputeThinV);
    return std::make_pair(svd.matrixU(), svd.singularValues());
  };

  // compute eigenvectors and eigenvalues of the hessian from P2P-ICP
  const Eigen::Matrix3d& H_tt = H.block<3, 3>(0, 0);
  const Eigen::Matrix3d& H_rr = H.block<3, 3>(3, 3);
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

  // resize hessian and gradient to accommodate constraints
  H_aug.resize(6 + c, 6 + c);
  g_aug.resize(6 + c);
  H_aug.setZero();
  g_aug.setZero();

  // augment hessian to satisfy KKT equations of equality-constrained optimization
  H_aug.block<6, 6>(0, 0) = H;
  H_aug.block(0, 6, 6, c) = C.transpose();
  H_aug.block(6, 0, c, 6) = C;
  g_aug.head<6>() = g;
}

bool ICP_BASE::convergenceCheck(const Eigen::Matrix4d& transform_iter) const {
  double relative_matching_rmse = std::abs(matching_rmse_ - matching_rmse_prev_);
  if (relative_matching_rmse > relative_matching_rmse_threshold_) {
    return false;
  }

  double cos_theta = 0.5 * (transform_iter.trace() - 2.0);
  if (cos_theta < cos_theta_threshold_) {
    return false;
  }

  double trans_sq = transform_iter.block<3, 1>(0, 3).squaredNorm();
  if (trans_sq > translation_threshold_) {
    return false;
  }

  return true;
}
