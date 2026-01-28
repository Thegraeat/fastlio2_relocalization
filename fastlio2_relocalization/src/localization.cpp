#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <tf/transform_broadcaster.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <Eigen/Dense>
#include <thread>
#include <mutex>

const double MAP_VOXEL_SIZE = 0.4;
const double SCAN_VOXEL_SIZE = 0.2;
const double FREQ_LOCALIZATION = 0.5; 
const double FREQ_TF_PUB = 50.0;      
const double LOCALIZATION_TH = 1.0;   

pcl::PointCloud<pcl::PointXYZ>::Ptr g_global_map(new pcl::PointCloud<pcl::PointXYZ>);
nav_msgs::Odometry::ConstPtr g_cur_odom;      // 来自 FAST-LIO 的 odom (camera_init -> body)
sensor_msgs::PointCloud2::ConstPtr g_cur_scan; 
geometry_msgs::PoseWithCovarianceStamped g_init_pose_msg;

bool g_initialized = false;
bool g_got_init_pose = false;
Eigen::Matrix4f g_T_map_to_odom = Eigen::Matrix4f::Identity(); 
std::mutex g_data_mutex;

Eigen::Matrix4f poseToMatrix(const geometry_msgs::Pose& pose) {
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    T.block<3,1>(0,3) = Eigen::Vector3f(pose.position.x, pose.position.y, pose.position.z);
    Eigen::Quaternionf q(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
    T.block<3,3>(0,0) = q.normalized().toRotationMatrix();
    return T;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr voxelDownsample(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, double size) {
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud);
    sor.setLeafSize(size, size, size);
    pcl::PointCloud<pcl::PointXYZ>::Ptr output(new pcl::PointCloud<pcl::PointXYZ>);
    sor.filter(*output);
    return output;
}

std::pair<Eigen::Matrix4f, float> doICP(const pcl::PointCloud<pcl::PointXYZ>::Ptr& scan, 
                                       const pcl::PointCloud<pcl::PointXYZ>::Ptr& map, 
                                       const Eigen::Matrix4f& guess) {
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(voxelDownsample(scan, SCAN_VOXEL_SIZE));
    icp.setInputTarget(map); // 地图通常已下采样
    icp.setMaximumIterations(30);
    icp.setTransformationEpsilon(1e-6);
    pcl::PointCloud<pcl::PointXYZ> final;
    icp.align(final, guess);
    return {icp.getFinalTransformation(), icp.getFitnessScore()};
}

void localizationLoop() {
    ros::Rate rate(FREQ_LOCALIZATION);
    while (ros::ok()) {
        rate.sleep();
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr scan_ptr(new pcl::PointCloud<pcl::PointXYZ>);
        nav_msgs::Odometry odom_data;
        Eigen::Matrix4f current_guess;

        {
            std::lock_guard<std::mutex> lock(g_data_mutex);
            if (!g_cur_scan || !g_cur_odom) continue;
            pcl::fromROSMsg(*g_cur_scan, *scan_ptr);
            odom_data = *g_cur_odom;
            
            if (!g_initialized) {
                if (!g_got_init_pose) continue;
                current_guess = poseToMatrix(g_init_pose_msg.pose.pose);
            } else {
                current_guess = g_T_map_to_odom; 
            }
        }

        ROS_INFO("ICP matching...");
        auto result = doICP(scan_ptr, g_global_map, current_guess);
        
        if (result.second < LOCALIZATION_TH) {
            std::lock_guard<std::mutex> lock(g_data_mutex);
            g_T_map_to_odom = result.first;
            if (!g_initialized) {
                g_initialized = true;
                ROS_INFO("Global Localization Success!");
            }
        } else {
            ROS_WARN("ICP Match Failed, score: %f", result.second);
        }
    }
}

void tfFusionLoop() {
    tf::TransformBroadcaster br;
    ros::Publisher pub_loc = ros::NodeHandle().advertise<nav_msgs::Odometry>("/localization", 10);
    ros::Rate rate(FREQ_TF_PUB);

    while (ros::ok()) {
        rate.sleep();
        if (!g_initialized) continue;

        Eigen::Matrix4f T_m2o;
        nav_msgs::Odometry odom_msg;
        {
            std::lock_guard<std::mutex> lock(g_data_mutex);
            T_m2o = g_T_map_to_odom;
            if (g_cur_odom) odom_msg = *g_cur_odom;
            else continue;
        }

        Eigen::Quaternionf q_m2o(T_m2o.block<3,3>(0,0));
        tf::Transform tf_m2o;
        tf_m2o.setOrigin(tf::Vector3(T_m2o(0,3), T_m2o(1,3), T_m2o(2,3)));
        tf_m2o.setRotation(tf::Quaternion(q_m2o.x(), q_m2o.y(), q_m2o.z(), q_m2o.w()));
        br.sendTransform(tf::StampedTransform(tf_m2o, ros::Time::now(), "map", "camera_init"));

        Eigen::Matrix4f T_o2b = poseToMatrix(odom_msg.pose.pose);
        Eigen::Matrix4f T_m2b = T_m2o * T_o2b;

        nav_msgs::Odometry loc_msg;
        loc_msg.header.stamp = ros::Time::now();
        loc_msg.header.frame_id = "map";
        loc_msg.child_frame_id = "body";
        
        loc_msg.pose.pose.position.x = T_m2b(0,3);
        loc_msg.pose.pose.position.y = T_m2b(1,3);
        loc_msg.pose.pose.position.z = T_m2b(2,3);
        Eigen::Quaternionf q_m2b(T_m2b.block<3,3>(0,0));
        loc_msg.pose.pose.orientation.x = q_m2b.x();
        loc_msg.pose.pose.orientation.y = q_m2b.y();
        loc_msg.pose.pose.orientation.z = q_m2b.z();
        loc_msg.pose.pose.orientation.w = q_m2b.w();
        pub_loc.publish(loc_msg);
    }
}

void scanCb(const sensor_msgs::PointCloud2::ConstPtr& msg) { std::lock_guard<std::mutex> l(g_data_mutex); g_cur_scan = msg; }
void odomCb(const nav_msgs::Odometry::ConstPtr& msg) { std::lock_guard<std::mutex> l(g_data_mutex); g_cur_odom = msg; }
void initialCb(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg) { 
    std::lock_guard<std::mutex> l(g_data_mutex); 
    g_init_pose_msg = *msg; 
    g_got_init_pose = true;
    ROS_INFO("Received Init Pose from Rviz.");
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "lio_relocalization");
    ros::NodeHandle nh;

    ros::Subscriber s1 = nh.subscribe("/cloud_registered", 1, scanCb);
    ros::Subscriber s2 = nh.subscribe("/Odometry", 1, odomCb);
    ros::Subscriber s3 = nh.subscribe("/initialpose", 1, initialCb);

    ROS_INFO("Waiting for /global_map...");
    auto map_msg = ros::topic::waitForMessage<sensor_msgs::PointCloud2>("/global_map", nh);
    pcl::fromROSMsg(*map_msg, *g_global_map);
    g_global_map = voxelDownsample(g_global_map, MAP_VOXEL_SIZE);
    ROS_INFO("Global Map Loaded. Points: %zu", g_global_map->size());

    std::thread t1(localizationLoop);
    std::thread t2(tfFusionLoop);

    ros::spin();
    
    t1.join();
    t2.join();
    return 0;
}