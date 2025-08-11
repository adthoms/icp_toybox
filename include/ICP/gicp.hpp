#pragma once

#include "ICP/icp_base.hpp"
#include "ICP/ceres_optimizer.hpp"

class GICP : public ICP_BASE {
public:
  GICP(SolverType solver_type = SolverType::LeastSquares) {
    solver_type_ = solver_type;
    if (solver_type == SolverType::LeastSquaresUsingCeres)
      optimizer_ = std::make_unique<CeresOptimizer>(CeresOptimizer::Type::GICP);
  }

private:
  bool checkValidity(PointCloud& source_cloud, PointCloud& target_cloud) override;
  std::pair<Eigen::Matrix6d, Eigen::Vector6d>
  compute_JTJ_and_JTr(const PointCloud& source_cloud, const PointCloud& target_cloud, int i) override;
  Eigen::Matrix4d computeTransformLeastSquaresUsingCeres(const PointCloud& source_cloud,
                                                         const PointCloud& target_cloud) override;

  void computeCovariancesFromNormals(PointCloud& cloud);
  Eigen::Matrix3d getRotationFromNormal(const Eigen::Vector3d& normal);

  double cov_epsilon_ = 5e-3;
};
