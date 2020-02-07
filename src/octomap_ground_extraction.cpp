#include <math.h>
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

// Holder class for params, callback, and published msg
class GroundFinder
{
  public:
    int min_cluster_size = 200;
    float normal_z_threshold;
    sensor_msgs::PointCloud2 ground_msg;
    void callback_octomap(const octomap_msgs::Octomap::ConstPtr msg);
};

void GroundFinder::callback_octomap(const octomap_msgs::Octomap::ConstPtr msg)
{
	if (msg->data.size() == 0) return;

  // Convert Octomap msg to a tree object
  octomap::OcTree* tree = new octomap::OcTree(msg->resolution);
  tree = (octomap::OcTree*)octomap_msgs::binaryMsgToMap(*msg);

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  // Iterate through all nodes to find voxels that are free with a bottom neighbor occupied
  // Loop through tree and extract occupancy info into esdf.data and seen/not into esdf.seen
  octomap::point3d query, query_neighbor;
  pcl::PointXYZ ground_point;
  tree->expand();
  ROS_INFO("Beginning tree iteration");
  for(octomap::OcTree::leaf_iterator it = tree->begin_leafs(),
       end=tree->end_leafs(); it!=end; ++it)
  {
    // Skip Occupied nodes
    // ROS_INFO("Checking occupancy @ (%0.1f, %0.1f, %0.1f)", it.getX(), it.getY(), it.getZ());
    if (it->getOccupancy() >= 0.3) {
      continue;
    }
    // Check if bottom neighbor is an occupied voxel
    // ROS_INFO("Node is free, checking bottom neighbor.");
    octomap::OcTreeNode* node = tree->search(it.getX(), it.getY(), it.getZ() - tree->getResolution());

    if (node != NULL) { // Might want to count nodes adjacent to nothing as ground as well.
      if (node->getOccupancy() >= 0.5) {
        // ROS_INFO("Bottom neighbor is occupied.  Add to cloud.");
        // Add to PCL
        ground_point.x = it.getX();
        ground_point.y = it.getY();
        ground_point.z = it.getZ();
        cloud->points.push_back(ground_point);
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
  ROS_INFO("Filtering out largest cluster");
  for (int i=0; i<cluster_indices[0].indices.size(); i++) {
    int idx = cluster_indices[0].indices[i];
    cloud_clustered->points.push_back(cloud_filtered->points[idx]);
  }
  ROS_INFO("Done.");
  // Publish the ground (maybe add in the a number of voxels above it for better fast marching)
  // Declare a new cloud to store the converted message
  ROS_INFO("Preparing PC2 message for publishing.");
  sensor_msgs::PointCloud2 new_PC2_msg;

  // Convert from pcl::PointCloud to sensor_msgs::PointCloud2
  pcl::toROSMsg(*cloud_clustered, new_PC2_msg);
  new_PC2_msg.header.seq = 1;
  new_PC2_msg.header.stamp = ros::Time();
  new_PC2_msg.header.frame_id = msg->header.frame_id;

  // Update the old message
  ground_msg = new_PC2_msg;
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
  ros::Subscriber sub = n.subscribe("octomap_binary", 1, &GroundFinder::callback_octomap, &finder);
  ros::Publisher pub = n.advertise<sensor_msgs::PointCloud2>("ground", 5);

  // Params
  n.param("ground_finder/min_cluster_size", finder.min_cluster_size, 100);
  n.param("ground_finder/normal_z_threshold", finder.normal_z_threshold, (float)0.8);

  float update_rate;
  n.param("ground_finder/update_rate", update_rate, (float)5.0);
  ros::Rate r(update_rate); // 5 Hz

  // Main Loop
  while (ros::ok())
  {
    r.sleep();
    ros::spinOnce();
    if (finder.ground_msg.data.size() > 0) pub.publish(finder.ground_msg);
  }
}