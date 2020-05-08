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

struct RobotState
{
  Eigen::Vector3f position;
  double sensor_range;
};

class NodeManager
{
  public:
    NodeManager():
    ground_cloud (new pcl::PointCloud<pcl::PointXYZ>),
    edt_cloud (new pcl::PointCloud<pcl::PointXYZI>)
    {
      map_octree = new octomap::OcTree(0.1);
    }
    bool use_tf = false;
    std::string robot_frame_id;
    std::string fixed_frame_id;
    // sensor_msgs::PointCloud2 ground_msg;
    // sensor_msgs::PointCloud2 edt_msg;
    octomap::OcTree* map_octree;
    bool map_updated = false;
    bool position_updated = false;
    int min_cluster_size;
    RobotState robot;
    float normal_z_threshold;
    float normal_curvature_threshold;
    pcl::PointCloud<pcl::PointXYZ>::Ptr ground_cloud;
    pcl::PointCloud<pcl::PointXYZI>::Ptr edt_cloud;
    void CallbackOctomap(const octomap_msgs::Octomap::ConstPtr msg);
    void CallbackOdometry(const nav_msgs::Odometry msg);
    void FindGroundVoxels();
    void UpdateRobotState();
    sensor_msgs::PointCloud2 GetGroundMsg();
    // void FilterNormals();
    // void FilterContiguous();
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
      if (distance < edt_cloud->points[i].intensity) edt_cloud->points[i].intensity = distance;
    }
  }

  delete[] dt;
  return;
}

void NodeManager::CallbackOctomap(const octomap_msgs::Octomap::ConstPtr msg)
{
  delete map_octree;
  map_octree = (octomap::OcTree*)octomap_msgs::binaryMsgToMap(*msg);
  map_updated = true;
}

void NodeManager::CallbackOdometry(nav_msgs::Odometry msg)
{
  if (use_tf) return;
  robot.position[0] = msg.pose.pose.position.x;
  robot.position[1] = msg.pose.pose.position.y;
  robot.position[2] = msg.pose.pose.position.z;
  position_updated = true;
  return;
}

void NodeManager::UpdateRobotState()
{
  if (!(use_tf)) return;
  tf::TransformListener listener;
  tf::StampedTransform transform;
  try{
    listener.lookupTransform(robot_frame_id, fixed_frame_id,
                              ros::Time(0), transform);
    robot.position[0] = transform.getOrigin().x();
    robot.position[1] = transform.getOrigin().y();
    robot.position[2] = transform.getOrigin().z();
    position_updated = true;
  }
  catch (tf::TransformException ex){
    ROS_ERROR("%s",ex.what());
    robot.position[0] = 0.0;
    robot.position[1] = 0.0;
    robot.position[2] = 0.0;
  }
}

sensor_msgs::PointCloud2 NodeManager::GetGroundMsg()
{
  sensor_msgs::PointCloud2 msg;
  pcl::toROSMsg(*ground_cloud, msg);
  msg.header.seq = 1;
  msg.header.stamp = ros::Time();
  msg.header.frame_id = fixed_frame_id;
  return msg;
}

void NodeManager::FindGroundVoxels()
{
  if (map_updated) {
    map_updated = false;
  } else {
    return;
  }

  UpdateRobotState();
  if (!(position_updated)) return;

  // Store voxel_size, this shouldn't change throughout the node running,
  // but something weird will happen if it does.
  double voxel_size = map_octree->getResolution();

  // Get map minimum dimensions
  double x_min_tree, y_min_tree, z_min_tree;
  map_octree->getMetricMin(x_min_tree, y_min_tree, z_min_tree);
  double min_tree[3] = {x_min_tree, y_min_tree, z_min_tree};

  ROS_INFO("Calculating bounding box.");
  // ***** //
  // Find a bounding box around the robot's current position to limit the map queries
  double bbx_min_array[3];
  double bbx_max_array[3];

  for (int i=0; i<3; i++) {
    bbx_min_array[i] = min_tree[i] + std::round((robot.position[i] - robot.sensor_range - min_tree[i])/voxel_size)*voxel_size - 2.95*voxel_size;
    bbx_max_array[i] = min_tree[i] + std::round((robot.position[i] + robot.sensor_range - min_tree[i])/voxel_size)*voxel_size + 2.95*voxel_size;
  }
  Eigen::Vector4f bbx_min(bbx_min_array[0], bbx_min_array[1], bbx_min_array[2], 1.0);
  Eigen::Vector4f bbx_max(bbx_max_array[0], bbx_max_array[1], bbx_max_array[2], 1.0);
  octomap::point3d bbx_min_octomap(bbx_min_array[0], bbx_min_array[1], bbx_min_array[2]);
  octomap::point3d bbx_max_octomap(bbx_max_array[0], bbx_max_array[1], bbx_max_array[2]);
  // ***** //
  ROS_INFO("Box has x,y,z limits of [%0.1f to %0.1f, %0.1f to %0.1f, and %0.1f to %0.1f] meters.",
  bbx_min_array[0], bbx_max_array[0], bbx_min_array[1], bbx_max_array[1], bbx_min_array[2], bbx_max_array[2]);

  ROS_INFO("Allocating occupied bool flat matrix memory.");
  // Allocate memory for the occupied cells within the bounding box in a flat 3D boolean array
  int bbx_size[3];
  for (int i=0; i<3; i++) {
    bbx_size[i] = (int)std::round((bbx_max[i] - bbx_min[i])/voxel_size) + 1;
  }
  int bbx_mat_length = bbx_size[0]*bbx_size[1]*bbx_size[2];
  bool occupied_mat[bbx_mat_length];
  for (int i=0; i<bbx_mat_length; i++) occupied_mat[i] = true;

  ROS_INFO("Removing the voxels within the bounding box from the ground_cloud of length %d", ground_cloud->points.size());

  // ***** //
  // Iterate through that box
  ROS_INFO("Beginning tree iteration through map of size %d", map_octree->size());
  // Initialize a PCL object to hold preliminary ground voxels
  pcl::PointCloud<pcl::PointXYZ>::Ptr ground_cloud_prefilter(new pcl::PointCloud<pcl::PointXYZ>);

  for(octomap::OcTree::leaf_bbx_iterator
  it = map_octree->begin_leafs_bbx(bbx_min_octomap, bbx_max_octomap);
  it != map_octree->end_leafs_bbx(); ++it)
  {
    // ROS_INFO("Getting leaf size");
    double size = it.getSize();
    int size_in_voxels = std::round(it.getSize()/voxel_size);
    // ROS_INFO("Checking Occupancy of size %d leaf", size_in_voxels);
    if (it->getOccupancy() >= 0.3) {
      if (it->getOccupancy() >= 0.7) {
        // ROS_INFO("Leaf is occupied.");
        // ***** // func(it, &occupied_mat)
        double query[3];
        if (size_in_voxels == 1) { // Just add it if it's the lowest level voxel
          // ROS_INFO("Leaf is only one voxel big.");
          query[0] = it.getX();
          query[1] = it.getY();
          query[2] = it.getZ();
          int id = xyz_index3(query, bbx_min_array, bbx_size, voxel_size);
          if ((id >=0) && (id < (bbx_size[0]*bbx_size[1]*bbx_size[2]))) occupied_mat[id] = false;
        } else {
          // ROS_INFO("Leaf is multiple voxels, iterating through all of them");
          // Iterate through leaf and mark all voxels in occupied matrix structure as occupied.
          Eigen::Vector3f lower_left_corner;
          float d_corner = (size - voxel_size)/2.0;
          lower_left_corner[0] = it.getX() - d_corner;
          lower_left_corner[1] = it.getY() - d_corner;
          lower_left_corner[2] = it.getZ() - d_corner;
          for (int i=0; i<size_in_voxels; i++) {
            query[0] = lower_left_corner[0] + i*voxel_size;
            for (int j=0; j<size_in_voxels; j++) {
              query[1] = lower_left_corner[1] + j*voxel_size;
              for (int k=0; k<size_in_voxels; k++) {
                query[2] = lower_left_corner[2] + k*voxel_size;
                int id = xyz_index3(query, bbx_min_array, bbx_size, voxel_size);
                if ((id >=0) && (id < (bbx_size[0]*bbx_size[1]*bbx_size[2]))) occupied_mat[id] = false;
              }
            }
          }
        }
        // ***** //
      }
      continue;
    } else {
      // ROS_INFO("Leaf is free, checking for ground voxels beneath it.");
      double query[3];
      if (size_in_voxels == 1) { // Just add it if it's the lowest level voxel and adjacent to ground
        // ROS_INFO("Leaf is only one voxel big.");
        query[0] = it.getX();
        query[1] = it.getY();
        query[2] = it.getZ();
        if (!CheckPointInBounds(query, bbx_min_array, bbx_max_array)) continue;
        // ROS_INFO("Checking if (%0.2f, %0.2f %0.2f) is ground", query[0], query[1], query[2]);
        // ROS_INFO("Querying node below current voxel.");
        octomap::OcTreeNode* node = map_octree->search(query[0], query[1], query[2] - voxel_size);
        // if ((node == nullptr) || (node == 0)) {
        if (node) {
          // ROS_INFO("Node is seen.");
          if (node->getOccupancy() >= 0.48) { // include points that have bottom neighbors that are unseen
            // ROS_INFO("Voxel below is occupied adding ground voxel.");
            // If it is, add the point to the ground PCL
            pcl::PointXYZ ground_point;
            ground_point.x = query[0];
            ground_point.y = query[1];
            ground_point.z = query[2];
            ground_cloud_prefilter->points.push_back(ground_point);
            // ROS_INFO("Voxel added.");
          }
        }
        else {
          // ROS_INFO("Voxel below is unseen adding ground voxel.");
          // If it is, add the point to the ground PCL
          pcl::PointXYZ ground_point;
          ground_point.x = query[0];
          ground_point.y = query[1];
          ground_point.z = query[2];
          ground_cloud_prefilter->points.push_back(ground_point);
          // ROS_INFO("Voxel added.");
        }
        // ROS_INFO("Ground node query deleted.");
      } else {
        // ROS_INFO("Leaf at [%0.2f, %0.2f, %0.2f] is multiple voxels, iterating through the bottom ones.", it.getX(), it.getY(), it.getZ());
        // Iterate through bottom voxels of the leaf and mark all voxels in occupied matrix structure as occupied.
        Eigen::Vector3f lower_left_corner;
        float d_corner = (size - voxel_size)/2.0;
        lower_left_corner[0] = it.getX() - d_corner;
        lower_left_corner[1] = it.getY() - d_corner;
        lower_left_corner[2] = it.getZ() - d_corner;
        query[2] = lower_left_corner[2];
        for (int i=0; i<size_in_voxels; i++) {
          query[0] = lower_left_corner[0] + i*voxel_size;
          for (int j=0; j<size_in_voxels; j++) {
            query[1] = lower_left_corner[1] + j*voxel_size;
            // ROS_INFO("Checking if (%0.2f, %0.2f %0.2f) is ground", query[0], query[1], query[2]);
            if (!CheckPointInBounds(query, bbx_min_array, bbx_max_array)) continue;
            // Check if bottom neighbor is occupied or unseen
            // ROS_INFO("Querying node below current voxel.");
            octomap::OcTreeNode* node = map_octree->search(query[0], query[1], query[2] - voxel_size);
            if (node) {
              // ROS_INFO("Node is seen.");
              // std::cout << node << std::endl;
              if (node->getOccupancy() >= 0.48) { // include points that have bottom neighbors that are unseen
                // ROS_INFO("Voxel below is occupied or unseen, adding ground voxel.");
                // If it is, add the point to the ground PCL
                pcl::PointXYZ ground_point;
                ground_point.x = query[0];
                ground_point.y = query[1];
                ground_point.z = query[2];
                ground_cloud_prefilter->points.push_back(ground_point);
                // ROS_INFO("Voxel added.");
              }
            }
            else {
              // ROS_INFO("Voxel below is unseen, adding ground voxel.");
              // If it is, add the point to the ground PCL
              pcl::PointXYZ ground_point;
              ground_point.x = query[0];
              ground_point.y = query[1];
              ground_point.z = query[2];
              ground_cloud_prefilter->points.push_back(ground_point);
              // ROS_INFO("Voxel added.");
            }
            // ROS_INFO("Ground node query deleted.");
          }
        }
      }
    }
  }
  // ***** //

  ROS_INFO("Normal vector filtering initial ground PointCloud of length %d", ground_cloud_prefilter->points.size());
  // ***** //
  // Filter ground by local normal vector
  pcl::PointCloud<pcl::PointXYZ>::Ptr ground_cloud_normal_filtered (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_filter;
  normal_filter.setInputCloud(ground_cloud_prefilter);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree_normal (new pcl::search::KdTree<pcl::PointXYZ>());
  normal_filter.setSearchMethod(kdtree_normal);
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
  normal_filter.setRadiusSearch(3.0*map_octree->getResolution());
  normal_filter.setViewPoint(0.0, 0.0, 2.0);
  normal_filter.compute(*cloud_normals);

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
  for (int i=0; i<cloud_normals->points.size(); i++) {
    if (std::abs(cloud_normals->points[i].normal_z) >= normal_z_threshold) {
      if (std::abs(cloud_normals->points[i].curvature) <= normal_curvature_threshold) {
        ground_cloud_normal_filtered->points.push_back(ground_cloud_prefilter->points[i]);
      }
    }
  }
  // ***** //

  ROS_INFO("Contiguity filtering normal filtered cloud of length %d", ground_cloud_normal_filtered->points.size());
  // ***** //
  // Filter ground by contiguity (is this necessary?)
  pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
  kdtree->setInputCloud(cloud_filtered);

  // Initialize euclidean cluster extraction object
  ROS_INFO("Beginning Frontier Clustering");
  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> euclidean_cluster_extractor;
  euclidean_cluster_extractor.setClusterTolerance(1.5*voxel_size); // Clusters must be made of contiguous sections of ground (within sqrt(2)*voxel_size of each other)
  euclidean_cluster_extractor.setMinClusterSize(min_cluster_size); // Cluster must be at least 15 voxels in size
  euclidean_cluster_extractor.setSearchMethod(kdtree);
  euclidean_cluster_extractor.setInputCloud(ground_cloud_normal_filtered);
  euclidean_cluster_extractor.extract(cluster_indices);
  ROS_INFO("Clusters extracted.");

  // Extract a local bounding box from the ground_cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr ground_cloud_local (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::CropBox<pcl::PointXYZ> box_filter;
  box_filter.setMin(bbx_min);
  box_filter.setMax(bbx_max);
  box_filter.setNegative(true);
  box_filter.setInputCloud(ground_cloud);
  box_filter.filter(*ground_cloud_local);

  // Add the biggest (or the one with the robot in it) to the ground_cloud.
  if (cluster_indices.size() > 0) {
    for (int i=0; i<cluster_indices[0].indices.size(); i++) {
      double query[3];
      query[0] = ground_cloud_normal_filtered->points[cluster_indices[0].indices[i]].x;
      query[1] = ground_cloud_normal_filtered->points[cluster_indices[0].indices[i]].y;
      query[2] = ground_cloud_normal_filtered->points[cluster_indices[0].indices[i]].z;
      ground_cloud_local->points.push_back(ground_cloud_normal_filtered->points[cluster_indices[0].indices[i]]);
      // Remove all the occupied cells beneath the ground cloud voxels from the occupied_mat
      query[2] = query[2] - voxel_size;
      if (CheckPointInBounds(query, bbx_min_array, bbx_max_array)) {
        occupied_mat[xyz_index3(query, bbx_min_array, bbx_size, voxel_size)] = true;
      }
      query[2] = query[2] - voxel_size;
      if (CheckPointInBounds(query, bbx_min_array, bbx_max_array)) {
        occupied_mat[xyz_index3(query, bbx_min_array, bbx_size, voxel_size)] = true;
      }
    }
  } else {
    ROS_INFO("No new cloud entries, publishing previous cloud msg");
    return;
  }

  // Clear ground_cloud and deep copy it from ground_cloud_local
  ground_cloud->points.clear();
  for (int i=0; i<ground_cloud_local->points.size(); i++) {
    ground_cloud->points.push_back(ground_cloud_local->points[i]);
  }
}

int main(int argc, char **argv)
{
  // Node declaration
  ros::init(argc, argv, "traversability_mapping");
  ros::NodeHandle n;

  // double voxel_size;
  // n.param("traversability_mapping/voxel_size", voxel_size, 0.2);
  // NodeManager node_manager(voxel_size);
  NodeManager node_manager;

  // Subscribers and Publishers
  ros::Subscriber sub = n.subscribe("octomap_binary", 1, &NodeManager::CallbackOctomap, &node_manager);
  ros::Subscriber sub1 = n.subscribe("odometry", 1, &NodeManager::CallbackOdometry, &node_manager);
  ros::Publisher pub1 = n.advertise<sensor_msgs::PointCloud2>("ground", 5);
  ros::Publisher pub2 = n.advertise<sensor_msgs::PointCloud2>("edt", 5);

  ROS_INFO("Initialized subscriber and publishers.");

  // Params
  n.param("traversability_mapping/min_cluster_size", node_manager.min_cluster_size, 100);
  n.param("traversability_mapping/normal_z_threshold", node_manager.normal_z_threshold, (float)0.8);
  n.param("traversability_mapping/normal_curvature_threshold", node_manager.normal_curvature_threshold, (float)50.0);
  n.param<std::string>("traversability_mapping/robot_frame_id", node_manager.robot_frame_id, "base_link");
  n.param<std::string>("traversability_mapping/fixed_frame_id", node_manager.fixed_frame_id, "world");
  n.param("traversability_mapping/sensor_range", node_manager.robot.sensor_range, 5.0);
  n.param("traversability_mapping/use_tf", node_manager.use_tf, false);

  float update_rate;
  n.param("traversability_mapping/update_rate", update_rate, (float)5.0);
  
  ros::Rate r(update_rate); // 5 Hz
  ROS_INFO("Finished reading params.");
  // Main Loop
  while (ros::ok())
  {
    r.sleep();
    ros::spinOnce();
    node_manager.FindGroundVoxels();
    ROS_INFO("ground cloud currently has %d points", node_manager.ground_cloud->points.size());
    if (node_manager.ground_cloud->points.size() > 0) pub1.publish(node_manager.GetGroundMsg());
    // if (node_manager.edt_msg.data.size() > 0) pub2.publish(node_manager.edt_msg);
  }
}