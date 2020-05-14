#include <math.h>
#include "edt.hpp"
// Octomap libaries
#include <octomap/octomap.h>
#include <octomap/ColorOcTree.h>
#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/conversions.h>
// ROS Libraries
#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/crop_box.h>
// Eigen
#include <Eigen/Core>

void index3_xyz(const int index, double point[3], double min[3], int size[3], double voxel_size)
{
  // x+y*sizx+z*sizx*sizy
  point[2] = min[2] + (index/(size[1]*size[0]))*voxel_size;
  point[1] = min[1] + ((index % (size[1]*size[0]))/size[0])*voxel_size;
  point[0] = min[0] + ((index % (size[1]*size[0])) % size[0])*voxel_size;
}

int xyz_index3(const double point[3], double min[3], int size[3], double voxel_size)
{
  int ind[3];
  for (int i=0; i<3; i++) ind[i] = round((point[i]-min[i])/voxel_size);
  return (ind[0] + ind[1]*size[0] + ind[2]*size[0]*size[1]);
}

bool CheckPointInBounds(double p[3], double min[3], double max[3])
{
  for (int i=0; i<3; i++) {
    if ((p[i] <= min[i]) || (p[i] >= max[i])) return false;
  }
  return true;
}

class NodeManager
{
  public:
    NodeManager()
    {
      map_octree = new octomap::OcTree(0.1);
    }
    std::string fixed_frame_id;
    sensor_msgs::PointCloud2 edt_msg;
    octomap::OcTree* map_octree;
    bool map_updated = false;
    float normal_z_threshold;
    float normal_curvature_threshold;
    void CallbackOctomap(const octomap_msgs::Octomap::ConstPtr msg);
    void UpdateEDT();
    void GetEdtMsg();
};

void CalculatePointCloudEDT(bool *occupied_mat, pcl::PointCloud<pcl::PointXYZI>::Ptr edt_cloud, double min[3], int size[3], double voxel_size)
{
  // Call EDT function
  float* dt = edt::edt<bool>(occupied_mat, /*sx=*/size[0], /*sy=*/size[1], /*sz=*/size[2],
  /*wx=*/1.0, /*wy=*/1.0, /*wz=*/1.0, /*black_border=*/false);

  // Parse EDT result into output PointCloud
  double max[3];
  for (int i=0; i<3; i++) max[i] = min[i] + (size[i]-1)*voxel_size;
  for (int i=0; i<edt_cloud->points.size(); i++) {
    double query[3] = {(double)edt_cloud->points[i].x, (double)edt_cloud->points[i].y, (double)edt_cloud->points[i].z};
    if (CheckPointInBounds(query, min, max)) {
      int idx = xyz_index3(query, min, size, voxel_size);
      float distance = (float)dt[idx]*voxel_size;
      // if (distance < edt_cloud->points[i].intensity) edt_cloud->points[i].intensity = distance;
      edt_cloud->points[i].intensity = distance;
    }
  }

  delete[] dt;
  return;
}

void NodeManager::CallbackOctomap(const octomap_msgs::Octomap::ConstPtr msg)
{
  if (msg->data.size() == 0) return;
  delete map_octree;
  map_octree = (octomap::OcTree*)octomap_msgs::binaryMsgToMap(*msg);
  map_updated = true;
}

void NodeManager::UpdateEDT()
{
  // Only run if the map has been updated
  if (map_updated) {
    map_updated = false;
  } else {
    return;
  }

  // Get map dimensions
  double x_min, y_min, z_min, x_max, y_max, z_max;
  map_octree->getMetricMin(x_min, y_min, z_min);
  map_octree->getMetricMax(x_max, y_max, z_max);
  double min[3], max[3];
  min[0] = x_min; min[1] = y_min; min[2] = z_min;
  max[0] = x_max; max[1] = y_max; max[2] = z_max;
  int size[3];
  for (int i=0; i<3; i++) {
    size[i] = std::round((max[i] - min[i])/map_octree->getResolution()) + 1;
  }

  // Allocate occupied flat matrix and free space pointcloud
  int flat_matrix_length = size[0]*size[1]*size[2];
  bool occupied_mat[flat_matrix_length];
  for (int i=0; i<flat_matrix_length; i++) occupied_mat[i] = true;
  pcl::PointCloud<pcl::PointXYZI>::Ptr edt_cloud (new pcl::PointCloud<pcl::PointXYZI>);

  map_octree->expand();
  ROS_INFO("Beginning tree iteration");
  for(octomap::OcTree::leaf_iterator it = map_octree->begin_leafs(),
       end=map_octree->end_leafs(); it!=end; ++it) {
    if (it->getOccupancy() >= 0.6)
    {
      // Add to occupied_mat
      double query[3];
      query[0] = it.getX(); query[1] = it.getY(); query[2] = it.getZ();
      if (CheckPointInBounds(query, min, max)) {
        int id = xyz_index3(query, min, size, map_octree->getResolution());
        occupied_mat[id] = false;
      }
    }
    else if (it->getOccupancy() <= 0.4)
    {
      // Add to edt_cloud
      double query[3];
      query[0] = it.getX(); query[1] = it.getY(); query[2] = it.getZ();
      if (CheckPointInBounds(query, min, max)) {
        pcl::PointXYZI query_point;
        query_point.x = query[0]; query_point.y = query[1]; query_point.z = query[2];
        edt_cloud->points.push_back(query_point);
      }
    }
  }

  // Run EDT
  CalculatePointCloudEDT(occupied_mat, edt_cloud, min, size, map_octree->getResolution());

  // Copy to msg
  sensor_msgs::PointCloud2 msg;
  pcl::toROSMsg(*edt_cloud, msg);
  msg.header.seq = 1;
  msg.header.stamp = ros::Time();
  msg.header.frame_id = fixed_frame_id;
  edt_msg = msg;

  return;
}

int main(int argc, char **argv)
{
  // Node declaration
  ros::init(argc, argv, "octomap_to_edt");
  ros::NodeHandle n;

  NodeManager node_manager;

  // Subscribers and Publishers
  ros::Subscriber sub = n.subscribe("octomap_binary", 1, &NodeManager::CallbackOctomap, &node_manager);
  ros::Publisher pub = n.advertise<sensor_msgs::PointCloud2>("edt", 5);

  ROS_INFO("Initialized subscriber and publishers.");

  // Params
  n.param<std::string>("octomap_to_edt/fixed_frame_id", node_manager.fixed_frame_id, "world");

  float update_rate;
  n.param("octomap_to_edt/update_rate", update_rate, (float)5.0);
  
  ros::Rate r(update_rate); // 5 Hz
  ROS_INFO("Finished reading params.");
  // Main Loop
  while (ros::ok())
  {
    r.sleep();
    ros::spinOnce();
    node_manager.UpdateEDT();
    if (node_manager.edt_msg.data.size() > 0) pub.publish(node_manager.edt_msg);
  }
}