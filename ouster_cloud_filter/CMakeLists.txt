cmake_minimum_required(VERSION 3.8)
project(ouster_cloud_filter)

# Required packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)

# Include paths
include_directories(
  include
  ${PCL_INCLUDE_DIRS}
)

# Node executable
add_executable(cloud_filter_node src/cloud_filter_node.cpp)

# Link dependencies
ament_target_dependencies(cloud_filter_node
  rclcpp
  sensor_msgs
  pcl_conversions
  tf2
  tf2_ros
  tf2_sensor_msgs
)

target_link_libraries(cloud_filter_node ${PCL_LIBRARIES})

# Install target
install(TARGETS
  cloud_filter_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()
