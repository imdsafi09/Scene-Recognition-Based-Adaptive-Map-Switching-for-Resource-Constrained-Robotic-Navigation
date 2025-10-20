#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#include <cmath>
#include <limits>

class CloudFilterNode : public rclcpp::Node
{
public:
  CloudFilterNode()
  : Node("cloud_filter_node"), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_)
  {
    // If you want to keep the original sensor frame (recommended), leave this empty.
    // Set to "base_link" only if you really want to transform the cloud.
    this->declare_parameter<std::string>("target_frame", "");  // "" => keep msg->header.frame_id
    this->get_parameter("target_frame", target_frame_);

    subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/ouster/points", rclcpp::SensorDataQoS(),
      std::bind(&CloudFilterNode::cloud_callback, this, std::placeholders::_1));

    scan_subscription_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
      "/ouster/scan", rclcpp::SensorDataQoS(),
      std::bind(&CloudFilterNode::scan_callback, this, std::placeholders::_1));

    // Publishers should also use SensorDataQoS to reduce queuing-induced skew.
    filtered_publisher_  = this->create_publisher<sensor_msgs::msg::PointCloud2>("/filtered_points", rclcpp::SensorDataQoS());
    original_publisher_  = this->create_publisher<sensor_msgs::msg::PointCloud2>("/original_points", rclcpp::SensorDataQoS());
    scan_publisher_      = this->create_publisher<sensor_msgs::msg::LaserScan>("/republished_scan", rclcpp::SensorDataQoS());
  }

private:
  void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
  {
    last_scan_msg_ = msg;
  }

  void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    // Decide the working frame
    const bool do_transform = (!target_frame_.empty() && target_frame_ != msg->header.frame_id);
    sensor_msgs::msg::PointCloud2 cloud_work = *msg;  // default: keep original frame & stamp

    if (do_transform)
    {
      try
      {
        // Transform at the *message timestamp* so it stays consistent with TF history.
        tf_buffer_.transform(*msg, cloud_work, target_frame_, tf2::durationFromSec(0.1));
      }
      catch (const tf2::TransformException &ex)
      {
        RCLCPP_WARN(this->get_logger(), "Could not transform %s -> %s at t=%.9f: %s",
                    msg->header.frame_id.c_str(), target_frame_.c_str(),
                    rclcpp::Time(msg->header.stamp).seconds(), ex.what());
        return;
      }
    }

    // Always preserve the original timestamp; DO NOT restamp with now().
    cloud_work.header.stamp = msg->header.stamp;

    // Publish original (possibly transformed) cloud for debugging
    original_publisher_->publish(cloud_work);

    // Republish scan with synchronized timestamp (optional convenience)
    if (last_scan_msg_)
    {
      auto synced_scan = *last_scan_msg_;
      synced_scan.header.stamp = msg->header.stamp;  // keep same timebase
      scan_publisher_->publish(synced_scan);
    }

    // Convert to PCL for filtering
    pcl::PointCloud<pcl::PointXYZI>::Ptr input(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(cloud_work, *input);

    // Filter parameters
    constexpr float x_min = 0.0f;
    constexpr float x_max = 6.0f;
    constexpr float z_min = -1.0f;
    constexpr float z_max = 1.5f;
    constexpr float tan_h_fov = std::tan(M_PI / 4);  // ±45°
    constexpr float tan_v_fov = std::tan(M_PI / 6);  // ±30°

    pcl::PointCloud<pcl::PointXYZI>::Ptr fov_filtered(new pcl::PointCloud<pcl::PointXYZI>);
    fov_filtered->reserve(input->size());

    for (const auto& p : input->points)
    {
      // Skip invalid points
      if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;

      if (p.x < x_min || p.x > x_max) continue;

      // Avoid division by zero around x ~ 0
      if (std::fabs(p.x) < 1e-3f) continue;

      const float inv_x = 1.0f / p.x;
      if (std::fabs(p.y * inv_x) > tan_h_fov) continue;
      if (std::fabs(p.z * inv_x) > tan_v_fov) continue;
      if (p.z < z_min || p.z > z_max) continue;

      fov_filtered->emplace_back(p);
    }

    // Voxel grid
    pcl::VoxelGrid<pcl::PointXYZI> voxel_filter;
    voxel_filter.setInputCloud(fov_filtered);
    voxel_filter.setLeafSize(0.01f, 0.01f, 0.01f);

    pcl::PointCloud<pcl::PointXYZI>::Ptr voxel_filtered(new pcl::PointCloud<pcl::PointXYZI>);
    voxel_filter.filter(*voxel_filtered);

    // Convert back and publish — keep original timestamp and (possibly transformed) frame
    sensor_msgs::msg::PointCloud2 out_msg;
    pcl::toROSMsg(*voxel_filtered, out_msg);
    out_msg.header = cloud_work.header;  // stamp & frame preserved
    filtered_publisher_->publish(out_msg);

    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                         "Filtered %zu -> %zu points (frame=%s, t=%.9f)",
                         input->points.size(), voxel_filtered->points.size(),
                         out_msg.header.frame_id.c_str(),
                         rclcpp::Time(out_msg.header.stamp).seconds());
  }

  std::string target_frame_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr   scan_subscription_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr original_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr    scan_publisher_;

  sensor_msgs::msg::LaserScan::SharedPtr last_scan_msg_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CloudFilterNode>());
  rclcpp::shutdown();
  return 0;
}
