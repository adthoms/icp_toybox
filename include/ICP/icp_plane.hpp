#pragma once

#include "ICP/icp_base.hpp"
#include "ICP/ceres_optimizer.hpp"

class ICP_PLANE : public ICP_BASE {
public:
  ICP_PLANE(SolverType solver_type = SolverType::LeastSquares) {
    solver_type_ = solver_type;
    if (solver_type == SolverType::LeastSquaresUsingCeres)
      optimizer_ = std::make_unique<CeresOptimizer>(CeresOptimizer::Type::PointToPlane);
  }

private:
  bool checkValidity(PointCloud& source_cloud, PointCloud& target_cloud) override;
  void computeHessianAndGradient(const PointCloud& source_cloud,
                                 const PointCloud& target_cloud,
                                 int i,
                                 Eigen::Matrix6d& H,
                                 Eigen::Vector6d& g) override;
  Eigen::Matrix4d computeTransformLeastSquaresUsingCeres(const PointCloud& source_cloud,
                                                         const PointCloud& target_cloud) override;
};
