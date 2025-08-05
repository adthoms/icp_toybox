#include <memory>
#include <thread>
#include <vector>
#include <chrono>
#include <fstream>

#include <open3d/Open3D.h>
#include <open3d/geometry/PointCloud.h>
#include <open3d/geometry/KDTreeFlann.h>

#include <CLI/CLI.hpp>
#include "ICP/gicp.hpp"
#include "ICP/icp_plane.hpp"

enum class ICPMethod { GICP_Open3D, GICP_direct, GICP_iterative, P2P_ICP_Open3D, P2P_ICP_direct, P2P_ICP_iterative };

std::string ICPMethodToString(ICPMethod method) {
  switch (method) {
  case ICPMethod::GICP_Open3D: {
    return "GICP_Open3D";
  }
  case ICPMethod::GICP_direct: {
    return "GICP_direct";
  }
  case ICPMethod::GICP_iterative: {
    return "GICP_iterative";
  }
  case ICPMethod::P2P_ICP_Open3D: {
    return "P2P_ICP_Open3D";
  }
  case ICPMethod::P2P_ICP_direct: {
    return "P2P_ICP_direct";
  }
  case ICPMethod::P2P_ICP_iterative: {
    return "P2P_ICP_iterative";
  }
  }
  return "";
}

void visualizeRegistration(const open3d::geometry::PointCloud& source,
                           const open3d::geometry::PointCloud& target,
                           const Eigen::Matrix4d& T_source_target) {
  // copy
  std::shared_ptr<open3d::geometry::PointCloud> source_copy = std::make_shared<open3d::geometry::PointCloud>();
  std::shared_ptr<open3d::geometry::PointCloud> target_copy = std::make_shared<open3d::geometry::PointCloud>();
  *source_copy = source;
  *target_copy = target;

  // color
  source_copy->PaintUniformColor({1, 0, 0}); // red
  target_copy->PaintUniformColor({0, 1, 0}); // green

  // transform
  source_copy->Transform(T_source_target);

  // visualize
  auto visualizer = std::make_unique<open3d::visualization::Visualizer>();
  visualizer->CreateVisualizerWindow("ICP result", 1024, 768);
  visualizer->GetRenderOption().SetPointSize(1.0);
  visualizer->GetRenderOption().background_color_ = {0, 0, 0}; // black
  visualizer->AddGeometry(source_copy);
  visualizer->AddGeometry(target_copy);
  visualizer->Run();
}

std::shared_ptr<open3d::geometry::PointCloud> readBinFile(const std::string& file_path) {
  auto cloud = std::make_shared<open3d::geometry::PointCloud>();

  // open file
  std::ifstream file(file_path, std::ios::binary);
  if (!file.is_open())
    return cloud;

  // read file size
  file.seekg(0, std::ios::end);
  std::streampos file_size = file.tellg();
  file.seekg(0, std::ios::beg);

  // read file
  std::size_t point_size = sizeof(float) * 4;
  std::size_t num_points = file_size / point_size;
  std::vector<float> all_points(num_points * 4);
  if (!file.read(reinterpret_cast<char*>(all_points.data()), file_size)) {
    file.close();
    return cloud;
  }
  file.close();

  // populate cloud
  cloud->points_.resize(num_points);
  Eigen::Vector4f kitti_point;
  for (std::size_t i = 0; i < num_points; ++i) {
    std::copy_n(all_points.begin() + 4 * i, 4, kitti_point.data());
    cloud->points_[i] = kitti_point.head<3>().cast<double>();
  }

  return cloud;
}

std::shared_ptr<open3d::geometry::PointCloud> loadPointCloud(const std::string& path) {
  if (path.substr(path.size() - 4) == ".bin")
    return readBinFile(path);
  else
    return open3d::io::CreatePointCloudFromFile(path);
}

void runICP(const std::shared_ptr<open3d::geometry::PointCloud>& source,
            const std::shared_ptr<open3d::geometry::PointCloud>& target,
            const std::shared_ptr<open3d::geometry::PointCloud>& source_down,
            const std::shared_ptr<open3d::geometry::PointCloud>& target_down,
            double max_correspondence_dist,
            int iteration,
            Eigen::Matrix4d& T_source_target,
            const ICPMethod& method) {
  auto t_start = std::chrono::high_resolution_clock::now();
  switch (method) {
  case ICPMethod::GICP_Open3D: {
    auto reg_result = open3d::pipelines::registration::RegistrationGeneralizedICP(
        *source_down,
        *target_down,
        max_correspondence_dist,
        Eigen::Matrix4d::Identity(),
        open3d::pipelines::registration::TransformationEstimationForGeneralizedICP(),
        open3d::pipelines::registration::ICPConvergenceCriteria(1e-6, 1e-6, iteration));
    T_source_target = reg_result.transformation_;
    break;
  }
  case ICPMethod::GICP_direct: {
    GICP icp(ICP_BASE::SolverType::LeastSquares);
    icp.setIteration(iteration);
    icp.setMaxCorrespondenceDist(max_correspondence_dist);
    icp.align(*source_down, *target_down);
    T_source_target = icp.getResultTransform();
    break;
  }
  case ICPMethod::GICP_iterative: {
    GICP icp(ICP_BASE::SolverType::LeastSquaresUsingCeres);
    icp.setIteration(iteration);
    icp.setMaxCorrespondenceDist(max_correspondence_dist);
    icp.align(*source_down, *target_down);
    T_source_target = icp.getResultTransform();
    break;
  }
  case ICPMethod::P2P_ICP_Open3D: {
    auto reg_result = open3d::pipelines::registration::RegistrationICP(
        *source_down,
        *target_down,
        max_correspondence_dist,
        Eigen::Matrix4d::Identity(),
        open3d::pipelines::registration::TransformationEstimationPointToPlane(),
        open3d::pipelines::registration::ICPConvergenceCriteria(1e-6, 1e-6, iteration));
    T_source_target = reg_result.transformation_;
    break;
  }
  case ICPMethod::P2P_ICP_direct: {
    ICP_PLANE icp(ICP_BASE::SolverType::LeastSquares);
    icp.setIteration(iteration);
    icp.setMaxCorrespondenceDist(max_correspondence_dist);
    icp.align(*source_down, *target_down);
    T_source_target = icp.getResultTransform();
    break;
  }
  case ICPMethod::P2P_ICP_iterative: {
    ICP_PLANE icp(ICP_BASE::SolverType::LeastSquaresUsingCeres);
    icp.setIteration(iteration);
    icp.setMaxCorrespondenceDist(max_correspondence_dist);
    icp.align(*source_down, *target_down);
    T_source_target = icp.getResultTransform();
    break;
  }
  }
  auto t_end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();

  // results
  std::cout << ICPMethodToString(method) << " elapsed time : " << duration << "ms" << std::endl;
  std::cout << "T_source_target = \n" << T_source_target << std::endl;
  visualizeRegistration(*source, *target, T_source_target);
}

int main(int argc, char* argv[]) {
  CLI::App app{"ICP Example"};

  // arguments
  std::string source_cloud_path;
  std::string target_cloud_path;
  int iteration = 100;
  double voxel_size = 0.3;
  double max_correspondence_dist = 10.0;

  // parse command line arguments
  app.add_option("--source_cloud_path", source_cloud_path, "Path to source point cloud")->required();
  app.add_option("--target_cloud_path", target_cloud_path, "Path to target point cloud")->required();
  app.add_option("--iteration", iteration, "Number of ICP iterations");
  app.add_option("--voxel_size", voxel_size, "Voxel size for downsampling");
  app.add_option("--max_correspondence_dist", max_correspondence_dist, "Max correspondence distance");

  CLI11_PARSE(app, argc, argv);

  // load source and target point clouds
  std::shared_ptr<open3d::geometry::PointCloud> source = loadPointCloud(source_cloud_path);
  std::shared_ptr<open3d::geometry::PointCloud> target = loadPointCloud(target_cloud_path);
  if (source->IsEmpty() || target->IsEmpty()) {
    LOG(ERROR) << "Unable to load source or target files.";
    std::exit(1);
  }

  // init transformation matrix
  Eigen::Matrix4d T_source_target = Eigen::Matrix4d::Identity();

  // down-sample source and target and estimate normals
  std::shared_ptr<open3d::geometry::PointCloud> source_down = source->VoxelDownSample(voxel_size);
  std::shared_ptr<open3d::geometry::PointCloud> target_down = target->VoxelDownSample(voxel_size);
  source_down->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(voxel_size * 2.0, 30));
  target_down->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(voxel_size * 2.0, 30));

  // ICP
  std::cout << "Running ICP methods..." << std::endl;
  const std::vector<ICPMethod> icp_methods = {
      ICPMethod::GICP_Open3D, ICPMethod::GICP_direct, ICPMethod::P2P_ICP_Open3D, ICPMethod::P2P_ICP_direct};
  for (const auto& method : icp_methods) {
    runICP(source, target, source_down, target_down, max_correspondence_dist, iteration, T_source_target, method);
  }
  std::cout << "ICP methods complete." << std::endl;
  return 0;
}
