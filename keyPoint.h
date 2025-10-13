#pragma once
// harris_keypoints.h
#ifndef HARRIS_KEYPOINTS_H
#define HARRIS_KEYPOINTS_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <string>

pcl::PointCloud<pcl::PointXYZ>::Ptr detectHarrisKeypoints(
    const std::string& input_file,
    const std::string& output_file = "KeyPoint/V_keypointsHarris.pcd",
    float radius = 3.2f,
    float radius_search = 3.0f,
    float threshold = 0.0001f,
    int threads = 10,
    bool visualize = true);

bool processVGrooveAndExtractLine(const std::string& input_file, const std::string& output_file);
#endif // HARRIS_KEYPOINTS_H
