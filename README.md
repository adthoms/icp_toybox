# ICP Toybox

This repository implements Point-to-Plane ICP (P2P-ICP) and G-ICP algorithms using custom direct (i.e., least squares) and iterative (i.e., non-linear least squares) solvers. Custom direct solvers implement the active geometric degeneracy technique described in IV-A-1 of the following paper:
```bibtex
@ARTICLE{tuna2025informed,
  author={Tuna, Turcan and Nubert, Julian and Pfreundschuh, Patrick and Cadena, Cesar and Khattak, Shehryar and Hutter, Marco},
  journal={IEEE Transactions on Field Robotics},
  title={Informed, Constrained, Aligned: A Field Analysis on Degeneracy-Aware Point Cloud Registration in the Wild},
  year={2025},
  volume={2},
  number={},
  pages={485-515},
  doi={10.1109/TFR.2025.3576053}
}
```

## References

- P2P-ICP Algorithm : [Linear Least-Squares Optimization for Point-to-Plane ICP Surface Registration](https://www.comp.nus.edu.sg/~lowkl/publications/lowk_point-to-plane_icp_techrep.pdf) by Kok-Lim Low.
- G-ICP Algorithm : [Generalized-ICP](https://www.roboticsproceedings.org/rss05/p21.pdf) by Aleksandr V. Segal, Dirk Haehnel and Sebastian Thrun.

## Dependencies

- [Ubuntu 20.04](https://releases.ubuntu.com/focal/)
- [Gflags 2.2.2](https://github.com/gflags/gflags)
- [Glog 0.6.0](https://github.com/google/glog)
- [Ceres-Solver 2.2.0](http://ceres-solver.org/)
- [Open3D 0.17.0](http://www.open3d.org/)

## Install

### Packages

Apt Packages
```bash
sudo apt install build-essential
sudo apt-get install libatlas-base-dev libeigen3-dev libsuitesparse-dev
```

[CMake](https://apt.kitware.com/)

```bash
sudo apt-get update
sudo apt-get install ca-certificates gpg wget
test -f /usr/share/doc/kitware-archive-keyring/copyright || wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main' | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null
sudo apt-get update
test -f /usr/share/doc/kitware-archive-keyring/copyright || sudo rm /usr/share/keyrings/kitware-archive-keyring.gpg
sudo apt-get install kitware-archive-keyring
echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal-rc main' | sudo tee -a /etc/apt/sources.list.d/kitware.list >/dev/null
sudo apt-get update
sudo apt install cmake=3.27.7-0kitware1ubuntu20.04.1 cmake-data=3.27.7-0kitware1ubuntu20.04.1 # CMake>=3.26 required
```

### Libraries

To install the following libraries (from source), create the following directory:
```bash
cd && mkdir -p ~/Software
```

[Gflags](https://github.com/gflags/gflags/blob/master/INSTALL.md#compiling-the-source-code-with-cmake)

```bash
cd ~/Software
git clone --depth 1 --branch v2.2.2 git@github.com:gflags/gflags.git
cd gflags
mkdir -p build && cd build
cmake -DCMAKE_CXX_STANDARD=17 \
      -DCMAKE_CXX_STANDARD_REQUIRED=ON \
      -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_CXX_FLAGS=-fPIC \
      -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
sudo make install
```

[Glog](https://google.github.io/glog/stable/build/#cmake)

```bash
cd ~/Software
git clone --depth 1 --branch v0.6.0 git@github.com:google/glog.git
cd glog
cmake -S . -B build \
      -DCMAKE_CXX_STANDARD=17 \
      -DCMAKE_CXX_STANDARD_REQUIRED=ON \
      -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_CXX_FLAGS=-fPIC \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
sudo cmake --build build --target install
```

[Ceres](http://ceres-solver.org/installation.html#linux)

```bash
cd ~/Software
wget http://ceres-solver.org/ceres-solver-2.2.0.tar.gz && tar zxf ceres-solver-2.2.0.tar.gz
cd ceres-solver-2.2.0
mkdir -p build && cd build
cmake -DUSE_CUDA=OFF \
      -DCMAKE_CXX_STANDARD=17 \
      -DCMAKE_CXX_STANDARD_REQUIRED=ON \
      -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_CXX_FLAGS=-fPIC \
      -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
sudo make install
```

[Open3D](https://www.open3d.org/docs/release/compilation.html)

```bash
cd ~/Software
git clone --depth 1 --branch v0.17.0 https://github.com/isl-org/Open3D
cd Open3d && ./util/install_deps_ubuntu.sh
mkdir -p build && cd build
cmake -DCMAKE_CXX_STANDARD=17 \
      -DCMAKE_CXX_STANDARD_REQUIRED=ON \
      -DBUILD_SHARED_LIBS=ON \
      -DBUILD_PYTHON_MODULE=ON \
      -DBUILD_CUDA_MODULE=OFF \
      -DGLIBCXX_USE_CXX11_ABI=ON \
      -DBUILD_PYTORCH_OPS=OFF \
      -DBUILD_TENSORFLOW_OPS=OFF \
      -DBUNDLE_OPEN3D_ML=OFF ..
make -j$(nproc)
make python-package
sudo make install
```

## Build

```bash
cd ~/catkin_ws/src
git clone git@github.com:adthoms/icp_toybox.git
catkin build -j$(nproc) icp_toybox
```

## Examples

Run P2P-ICP and G-ICP algorithms using custom direct solvers, which are benchmarked against Open3D's implementation:
```bash
cd ~/catkin_ws/build/icp_toybox
./icp_example \
      --source_cloud_path=/path/to/source_cloud*.bin \
      --target_cloud_path=/path/to/source_cloud*.bin
```
See the full list of arguments with descriptions:
```bash
./icp_example --help
```
Examples include:
### [KITTI](http://www.cvlibs.net/datasets/kitti/)
```bash
./icp_example \
      --source_cloud_path=/home/fieldai/Projects/ros_ws/src/icp_toybox/data/kitti/0000000240.bin \
      --target_cloud_path=/home/fieldai/Projects/ros_ws/src/icp_toybox/data/kitti/0000000250.bin
```

### Primitive
```bash
./icp_example \
      --source_cloud_path=/home/fieldai/Projects/ros_ws/src/icp_toybox/data/primitive/source.pcd \
      --target_cloud_path=/home/fieldai/Projects/ros_ws/src/icp_toybox/data/primitive/target.pcd
```
For this example, we expect degeneracy in `z` and `yaw` and directions. To enable AGDM, set the following flags for P2P-ICP:
```bash
--eigenvalue_translation_threshold=10 --eigenvalue_rotation_threshold=100
```
and for G-ICP:
```bash
--eigenvalue_translation_threshold=400 --eigenvalue_rotation_threshold=500
```
Note that these values were emperically determined by visually observing degeneracy, recording the eigenvalue(s) corresponding to the degenerate direction, and then setting the threshold above these recorded values.

The point clouds used for the **Primitive** example can be visualized and generated as follows:
```bash
cd ~/catkin_ws/build/icp_toybox/scripts
python generate_primitive_point_clouds.py --visualize --save
```
See the full list of arguments with descriptions:
```bash
python generate_primitive_point_clouds.py --help
```

## Troubleshooting

1. Remove any apt packages for glog and gflags by first checking existing installs:
```bash
dpkg -l | grep gflags
dpkg -l | grep glog
```
and then removing them. For example:
```bash
sudo apt-get remove libgoogle-glog-dev libgoogle-glog0v5
sudo apt-get remove libgflags-dev  libgflags2.2
```

2. Ensure only one version (>=3.8) of python3 is installed.
