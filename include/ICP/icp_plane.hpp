#ifndef _ICP_ICP_PLANE_HPP_
#define _ICP_ICP_PLANE_HPP_

#include "ICP/icp_base.hpp"
#include "ICP/ceres_optimizer.hpp"

class ICP_PLANE : public ICP_BASE {
public:
  ICP_PLANE(SolverType solver_type = SolverType::LeastSquares) {
    solver_type_ = solver_type;

    if (solver_type == SolverType::SVD) {
      LOG(WARNING) << "ICP_PLANE has no SVD solver. use LeastSquares solver";
      solver_type = SolverType::LeastSquares;
    } else if (solver_type == SolverType::LeastSquaresUsingCeres)
      optimizer_ = std::make_unique<CeresOptimizer>(CeresOptimizer::Type::PointToPlane);
  }

private:
  bool checkValidity(PointCloud& source_cloud, PointCloud& target_cloud) override;
  std::pair<Eigen::Matrix<double, 6, 6>, Eigen::Matrix<double, 6, 1>>
  compute_JTJ_and_JTr(const PointCloud& source_cloud, const PointCloud& target_cloud, int i) override;
  Eigen::Matrix4d computeTransformLeastSquaresUsingCeres(const PointCloud& source_cloud,
                                                         const PointCloud& target_cloud) override;
};

#endif // _ICP_ICP_PLANE_HPP_
