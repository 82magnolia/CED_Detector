#include "perception.h"
#include <pcl/io/ply_io.h>

using PointT = pcl::PointXYZRGBNormal;
using FeatureT = pcl::PFHRGBSignature250;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << "Usage: ./test_keypoint <config.yaml>" << std::endl;
    return 0;
  }

  // parse configuration
  YAML::Node root_node = YAML::LoadFile(argv[1]);
  YAML::Node test_kp_node = root_node["test_kp_config"];
  YAML::Node perception_node = root_node["perception_config"];

  const std::string keypoint_method = test_kp_node["keypoint_method"].as<std::string> ();
  const std::string src_filename = argv[2];
  std::string save_root(argv[3]);
  const std::string log_filename = save_root + "/keypoint_log.csv";
  const std::string save_cloud_to_file = save_root + "/cloud.ply";
  const std::string save_keypoint_to_file = save_root + "/keypoint.ply";
  const bool pcl_visualization = test_kp_node["pcl_visualization"].as<bool> ();

  // prepare point clouds
  pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
  pcl::PointCloud<PointT>::Ptr keypoint (new pcl::PointCloud<PointT>);
  if (pcl::io::loadPLYFile (src_filename, *cloud) == -1)
    return 0;
  else if (perception_node["verbose"].as<bool> ())
    std::cout << "Loaded " << cloud->size() << " points for source cloud" << std::endl;

  // run algorithms
  my::Perception<FeatureT> perception(perception_node, log_filename);
  perception.removeNaNPoints (cloud);
  if (perception_node["downsample"].as<bool> ())
    perception.downSampleVoxelGrids (cloud);
  if (perception_node["estimate_normal"].as<bool> ())
    perception.estimateNormals (cloud);
  perception.detectKeypoints(keypoint_method, cloud, keypoint);

  // visualization
  pcl::io::savePLYFile (save_cloud_to_file, *cloud);
  pcl::io::savePLYFile (save_keypoint_to_file, *keypoint);
}
