#include <math.h>
#include "edt.hpp"
// Octomap libaries
#include <octomap/octomap.h>
#include <octomap/ColorOcTree.h>
#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/conversions.h>
// ROS Libraries
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

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

void PointCloudEDT(pcl::PointCloud<pcl::PointXYZ>::Ptr input, pcl::PointCloud<pcl::PointXYZ>::Ptr occupied, pcl::PointCloud<pcl::PointXYZI>::Ptr output, double min[3], int size[3], double voxel_size)
{
  // This function converts input into a nx*ny*nz binary image, calls the edt cpp library, and then stores the values in output.

  // Parse occupied pointcloud values into holder boolean array
  bool mat[size[0]*size[1]*size[2]];
  for (int i=0; i<size[0]*size[1]*size[2]; i++) mat[i] = true; // initialize holder array values to true
  for (int i=0; i<occupied->points.size(); i++) {
    double query[3] = {(double)occupied->points[i].x, (double)occupied->points[i].y, (double)occupied->points[i].z};
    int idx = xyz_index3(query, min, size, voxel_size);
    if ((idx >= 0) && (idx < size[0]*size[1]*size[2])) {
      mat[idx] = false;
      // ROS_INFO("Marking cell occupied at (%0.1f, %0.1f, %0.1f)", query[0], query[1], query[2]);
    }
  }

  // Call EDT function
  float* dt = edt::edt<bool>(mat, /*sx=*/size[0], /*sy=*/size[1], /*sz=*/size[2],
  /*wx=*/1.0, /*wy=*/1.0, /*wz=*/100.0, /*black_border=*/false);

  // Parse EDT result into output PointCloud
  for (int i=0; i<input->points.size(); i++) {
    double query[3] = {(double)input->points[i].x, (double)input->points[i].y, (double)input->points[i].z};
    int idx = xyz_index3(query, min, size, voxel_size);
    pcl::PointXYZI edt_point;
    edt_point.x = (float)query[0]; edt_point.y = (float)query[1]; edt_point.z = (float)query[2];
    edt_point.intensity = (float)dt[idx]*voxel_size;
    output->points.push_back(edt_point);
  }
  return;
}

// Holder class for params, callback, and published msg
class GroundFinder
{
  public:
    int min_cluster_size = 200;
    float normal_z_threshold;
    int vertical_padding;
    sensor_msgs::PointCloud2 ground_msg;
    sensor_msgs::PointCloud2 edt_msg;
    void callbackOctomap(const octomap_msgs::Octomap::ConstPtr msg);
};

void GroundFinder::callbackOctomap(const octomap_msgs::Octomap::ConstPtr msg)
{
	if (msg->data.size() == 0) return;

  // Convert Octomap msg to a tree object
  octomap::OcTree* tree = new octomap::OcTree(msg->resolution);
  tree = (octomap::OcTree*)octomap_msgs::binaryMsgToMap(*msg);

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_occupied(new pcl::PointCloud<pcl::PointXYZ>);
  // Iterate through all nodes to find voxels that are free with a bottom neighbor occupied
  // Loop through tree and extract occupancy info into esdf.data and seen/not into esdf.seen
  octomap::point3d query, query_neighbor;
  pcl::PointXYZ ground_point;
  pcl::PointXYZ occupied_point;
  tree->expand();
  ROS_INFO("Beginning tree iteration");
  for(octomap::OcTree::leaf_iterator it = tree->begin_leafs(),
       end=tree->end_leafs(); it!=end; ++it)
  {
    // Skip Occupied nodes
    // ROS_INFO("Checking occupancy @ (%0.1f, %0.1f, %0.1f)", it.getX(), it.getY(), it.getZ());
    if (it->getOccupancy() >= 0.3) {
      if (it->getOccupancy() >= 0.7) {
        occupied_point.x = it.getX();
        occupied_point.y = it.getY();
        occupied_point.z = it.getZ();
        cloud_occupied->points.push_back(occupied_point);
      }
      continue;
    }
    // Check if bottom neighbor or its neighbors are an occupied voxel
    // ROS_INFO("Node is free, checking bottom neighbor.");
    std::vector<octomap::OcTreeNode*> bottom_neighbors;
    octomap::OcTreeNode* node0 = tree->search(it.getX(), it.getY(), it.getZ() - tree->getResolution());
    octomap::OcTreeNode* node1 = tree->search(it.getX() - tree->getResolution(), it.getY(), it.getZ() - tree->getResolution());
    octomap::OcTreeNode* node2 = tree->search(it.getX() + tree->getResolution(), it.getY(), it.getZ() - tree->getResolution());
    octomap::OcTreeNode* node3 = tree->search(it.getX(), it.getY() - tree->getResolution(), it.getZ() - tree->getResolution());
    octomap::OcTreeNode* node4 = tree->search(it.getX(), it.getY() + tree->getResolution(), it.getZ() - tree->getResolution());
    bottom_neighbors.push_back(node0);
    bottom_neighbors.push_back(node1);
    bottom_neighbors.push_back(node2);
    bottom_neighbors.push_back(node3);
    bottom_neighbors.push_back(node4);

    if (bottom_neighbors[0] == NULL) { // include points that have bottom neighbors that are unseen
      ground_point.x = it.getX();
      ground_point.y = it.getY();
      ground_point.z = it.getZ();
      cloud->points.push_back(ground_point);
      continue;
    }

    int ground_neighbor_count = 0;
    for (int i=0; i<5; i++) {
      if (bottom_neighbors[i] != NULL) { // Might want to count nodes adjacent to nothing as ground as well.
        if (bottom_neighbors[i]->getOccupancy() >= 0.5) {
          ground_neighbor_count++;
          // Add to PCL
          if ((i == 0) || (ground_neighbor_count == 2)) {
            ground_point.x = it.getX();
            ground_point.y = it.getY();
            ground_point.z = it.getZ();
            cloud->points.push_back(ground_point);
            break;
          }
        }
      }
    }
  }
  ROS_INFO("Done.");

  // *** WANT TO ADD THIS SO THE ROBOT CAN CHOOSE BETWEEN NAVIGATING STAIRS OR NOT ***
  // Filter the PCL based upon local normals (to take out stairs and steep ramps)
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
  ne.setInputCloud(cloud);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree_normal (new pcl::search::KdTree<pcl::PointXYZ>());
  ne.setSearchMethod(kdtree_normal);
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
  ne.setRadiusSearch(3.0*tree->getResolution());
  ne.compute(*cloud_normals);

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
  for (int i=0; i<cloud_normals->points.size(); i++)
  {
    if (std::abs(cloud_normals->points[i].normal_z) >= normal_z_threshold) {
      cloud_filtered->points.push_back(cloud->points[i]);
    }
  }

  // Filter the PCL based upon adjacent or diagonal contuguity
  pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
  kdtree->setInputCloud(cloud_filtered);

  // Initialize euclidean cluster extraction object
  ROS_INFO("Beginning Frontier Clustering");
  std::vector<pcl::PointIndices> cluster_indices;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_clustered(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance(1.5*tree->getResolution()); // Clusters must be made of contiguous sections of ground (within sqrt(2)*voxel_size of each other)
  ec.setMinClusterSize(min_cluster_size); // Cluster must be at least 15 voxels in size
  // ec.setMaxClusterSize (30);
  ec.setSearchMethod(kdtree);
  ec.setInputCloud(cloud_filtered);
  ec.extract(cluster_indices);
  ROS_INFO("Clusters extracted.");

  // Extract the largest cluster
  // Also calculate min/max of PCL
  if (cluster_indices.size() == 0) return;
  double min[3], max[3];
  min[0] = cloud_filtered->points[cluster_indices[0].indices[0]].x;
  min[1] = cloud_filtered->points[cluster_indices[0].indices[0]].y;
  min[2] = cloud_filtered->points[cluster_indices[0].indices[0]].z;
  max[0] = cloud_filtered->points[cluster_indices[0].indices[0]].x;
  max[1] = cloud_filtered->points[cluster_indices[0].indices[0]].y;
  max[2] = cloud_filtered->points[cluster_indices[0].indices[0]].z;
  ROS_INFO("Filtering out largest cluster");
  for (int i=0; i<cluster_indices[0].indices.size(); i++) {
    int idx = cluster_indices[0].indices[i];
    cloud_clustered->points.push_back(cloud_filtered->points[idx]);
    pcl::PointXYZ padded_point;
    padded_point.x = cloud_filtered->points[idx].x;
    padded_point.y = cloud_filtered->points[idx].y;
    padded_point.z = cloud_filtered->points[idx].z;
    for (int j=0; j<vertical_padding; j++) {
      padded_point.z = padded_point.z + tree->getResolution();
      cloud_clustered->points.push_back(padded_point);
    }
    if (padded_point.x < min[0]) min[0] = padded_point.x;
    if (padded_point.y < min[1]) min[1] = padded_point.y;
    if (cloud_filtered->points[idx].z < min[2]) min[2] = cloud_filtered->points[idx].z;
    if (padded_point.x > max[0]) max[0] = padded_point.x;
    if (padded_point.y > max[1]) max[1] = padded_point.y;
    if (padded_point.z > max[2]) max[2] = padded_point.z;
  }
  ROS_INFO("Done.");
  // Publish the ground (maybe add in the a number of voxels above it for better fast marching)
  // Declare a new cloud to store the converted message
  ROS_INFO("Preparing PC2 message for publishing.");
  sensor_msgs::PointCloud2 new_PC2_msg;

  // Convert from pcl::PointCloud to sensor_msgs::PointCloud2
  pcl::toROSMsg(*cloud_clustered, new_PC2_msg);
  // pcl::toROSMsg(*cloud_occupied, new_PC2_msg);
  new_PC2_msg.header.seq = 1;
  new_PC2_msg.header.stamp = ros::Time();
  new_PC2_msg.header.frame_id = msg->header.frame_id;

  // Update the old message
  ground_msg = new_PC2_msg;

  // Call EDT and write to msg
  int size[3];
  for (int i=0; i<3; i++) {
    // Pad the 3d matrix size with empty cells to get rid of EDT edge calculations
    min[i] = min[i] - 3.0*tree->getResolution();
    max[i] = max[i] + 3.0*tree->getResolution();
    size[i] = round((max[i]-min[i])/tree->getResolution()) + 1;
  }
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_edt (new pcl::PointCloud<pcl::PointXYZI>);
  PointCloudEDT(cloud_clustered, cloud_occupied, cloud_edt, min, size, tree->getResolution());

  pcl::toROSMsg(*cloud_edt, new_PC2_msg);
  new_PC2_msg.header.seq = 1;
  new_PC2_msg.header.stamp = ros::Time();
  new_PC2_msg.header.frame_id = msg->header.frame_id;

  // Update the old message
  edt_msg = new_PC2_msg;
  ROS_INFO("Callback end.");

  return;
}

int main(int argc, char **argv)
{
  // Node declaration
  ros::init(argc, argv, "ground_finderr");
  ros::NodeHandle n;

  GroundFinder finder;

  // Subscribers and Publishers
  ros::Subscriber sub = n.subscribe("octomap_binary", 1, &GroundFinder::callbackOctomap, &finder);
  ros::Publisher pub1 = n.advertise<sensor_msgs::PointCloud2>("ground", 5);
  ros::Publisher pub2 = n.advertise<sensor_msgs::PointCloud2>("edt", 5);

  // Params
  n.param("ground_finder/min_cluster_size", finder.min_cluster_size, 100);
  n.param("ground_finder/normal_z_threshold", finder.normal_z_threshold, (float)0.8);
  n.param("ground_finder/vertical_padding", finder.vertical_padding, 2);

  float update_rate;
  n.param("ground_finder/update_rate", update_rate, (float)5.0);
  ros::Rate r(update_rate); // 5 Hz

  // Main Loop
  while (ros::ok())
  {
    r.sleep();
    ros::spinOnce();
    if (finder.ground_msg.data.size() > 0) pub1.publish(finder.ground_msg);
    if (finder.edt_msg.data.size() > 0) pub2.publish(finder.edt_msg);
  }
}