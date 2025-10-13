#include <pcl/io/pcd_io.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/common/intersections.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/geometry/polygon_operations.h>
#include <iostream>
#include <filesystem>
#include <cmath>
#include <vector>
#include <limits>
#include <algorithm>
#include <random>

namespace fs = std::filesystem;

// 计算点到直线的距离
float pointToLineDistance(const pcl::PointXYZ& point,
    const Eigen::Vector3f& line_point,
    const Eigen::Vector3f& line_direction)
{
    Eigen::Vector3f p(point.x, point.y, point.z);
    Eigen::Vector3f diff = p - line_point;
    Eigen::Vector3f cross = diff.cross(line_direction);
    return cross.norm() / line_direction.norm();
}

// 精确的交线生成函数 - 基于平面凸包的交点
pcl::PointCloud<pcl::PointXYZ>::Ptr generateExactIntersectionLine(
    const pcl::ModelCoefficients::Ptr& line_coeff,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& plane1,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& plane2,
    int num_points = 100)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr line_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    if (line_coeff->values.size() != 6 || plane1->points.empty() || plane2->points.empty()) {
        return line_cloud;
    }

    // 提取交线参数
    Eigen::Vector3f line_point(line_coeff->values[0], line_coeff->values[1], line_coeff->values[2]);
    Eigen::Vector3f line_direction(line_coeff->values[3], line_coeff->values[4], line_coeff->values[5]);
    line_direction.normalize();

    // 计算两个平面的凸包
    pcl::ConvexHull<pcl::PointXYZ> hull1, hull2;
    pcl::PointCloud<pcl::PointXYZ>::Ptr hull_points1(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr hull_points2(new pcl::PointCloud<pcl::PointXYZ>);

    hull1.setInputCloud(plane1);
    hull1.reconstruct(*hull_points1);

    hull2.setInputCloud(plane2);
    hull2.reconstruct(*hull_points2);

    // 找到交线与凸包的交点
    std::vector<float> intersection_params;

    // 检查平面1凸包的每条边与交线的交点
    for (size_t i = 0; i < hull_points1->size(); ++i) {
        size_t j = (i + 1) % hull_points1->size();
        Eigen::Vector3f p1(hull_points1->points[i].x, hull_points1->points[i].y, hull_points1->points[i].z);
        Eigen::Vector3f p2(hull_points1->points[j].x, hull_points1->points[j].y, hull_points1->points[j].z);
        Eigen::Vector3f edge_dir = p2 - p1;

        // 计算直线与线段的交点
        Eigen::Vector3f cross1 = line_direction.cross(edge_dir);
        if (cross1.norm() < 1e-6) continue; // 平行

        Eigen::Vector3f diff = p1 - line_point;
        Eigen::Matrix3f mat;
        mat.col(0) = line_direction;
        mat.col(1) = -edge_dir;
        mat.col(2) = diff.cross(line_direction);

        // 求解参数
        Eigen::Vector3f params = mat.inverse() * diff;
        float t = params[0];
        float u = params[1];

        // 检查交点是否在线段上
        if (u >= 0 && u <= 1) {
            intersection_params.push_back(t);
        }
    }

    // 检查平面2凸包的每条边与交线的交点
    for (size_t i = 0; i < hull_points2->size(); ++i) {
        size_t j = (i + 1) % hull_points2->size();
        Eigen::Vector3f p1(hull_points2->points[i].x, hull_points2->points[i].y, hull_points2->points[i].z);
        Eigen::Vector3f p2(hull_points2->points[j].x, hull_points2->points[j].y, hull_points2->points[j].z);
        Eigen::Vector3f edge_dir = p2 - p1;

        // 计算直线与线段的交点
        Eigen::Vector3f cross1 = line_direction.cross(edge_dir);
        if (cross1.norm() < 1e-6) continue; // 平行

        Eigen::Vector3f diff = p1 - line_point;
        Eigen::Matrix3f mat;
        mat.col(0) = line_direction;
        mat.col(1) = -edge_dir;
        mat.col(2) = diff.cross(line_direction);

        // 求解参数
        Eigen::Vector3f params = mat.inverse() * diff;
        float t = params[0];
        float u = params[1];

        // 检查交点是否在线段上
        if (u >= 0 && u <= 1) {
            intersection_params.push_back(t);
        }
    }

    // 如果没有找到足够的交点，使用保守的方法
    if (intersection_params.size() < 2) {
        std::cout << "警告：无法找到足够的交点，使用保守方法" << std::endl;

        // 计算两个平面点云在交线方向上的投影范围
        std::vector<float> t_values;

        // 处理第一个平面
        for (const auto& p : plane1->points) {
            Eigen::Vector3f p_vec(p.x, p.y, p.z);
            Eigen::Vector3f diff = p_vec - line_point;
            float t = diff.dot(line_direction);
            t_values.push_back(t);
        }

        // 处理第二个平面
        for (const auto& p : plane2->points) {
            Eigen::Vector3f p_vec(p.x, p.y, p.z);
            Eigen::Vector3f diff = p_vec - line_point;
            float t = diff.dot(line_direction);
            t_values.push_back(t);
        }

        if (!t_values.empty()) {
            std::sort(t_values.begin(), t_values.end());
            // 使用极值但稍微向内收缩一点
            float range = t_values.back() - t_values.front();
            float t_min = t_values.front() + range * 0.01f; // 从最小值向内收缩10%
            float t_max = t_values.back() - range * 0.01f;  // 从最大值向内收缩10%

            intersection_params = { t_min, t_max };
        }
    }

    if (intersection_params.size() >= 2) {
        float t_min = *std::min_element(intersection_params.begin(), intersection_params.end());
        float t_max = *std::max_element(intersection_params.begin(), intersection_params.end());

        // 生成交线上的点
        float step = (t_max - t_min) / (num_points - 1);
        for (int i = 0; i < num_points; ++i) {
            float t = t_min + i * step;
            pcl::PointXYZ pt;
            pt.getVector3fMap() = line_point + t * line_direction;
            line_cloud->points.push_back(pt);
        }
    }

    line_cloud->width = line_cloud->points.size();
    line_cloud->height = 1;
    line_cloud->is_dense = true;

    return line_cloud;
}

// 简化的交线生成函数 - 基于平面点云到交线的距离
pcl::PointCloud<pcl::PointXYZ>::Ptr generateSimpleIntersectionLine(
    const pcl::ModelCoefficients::Ptr& line_coeff,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& plane1,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& plane2,
    int num_points = 100)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr line_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    if (line_coeff->values.size() != 6 || plane1->points.empty() || plane2->points.empty()) {
        return line_cloud;
    }

    // 提取交线参数
    Eigen::Vector3f line_point(line_coeff->values[0], line_coeff->values[1], line_coeff->values[2]);
    Eigen::Vector3f line_direction(line_coeff->values[3], line_coeff->values[4], line_coeff->values[5]);
    line_direction.normalize();

    // 找到距离交线最近的平面点，确定交线范围
    std::vector<float> t_values;
    float max_distance_threshold = 0.01f; // 1cm阈值

    // 处理第一个平面
    for (const auto& p : plane1->points) {
        float distance = pointToLineDistance(p, line_point, line_direction);
        if (distance < max_distance_threshold) {
            Eigen::Vector3f p_vec(p.x, p.y, p.z);
            Eigen::Vector3f diff = p_vec - line_point;
            float t = diff.dot(line_direction);
            t_values.push_back(t);
        }
    }

    // 处理第二个平面
    for (const auto& p : plane2->points) {
        float distance = pointToLineDistance(p, line_point, line_direction);
        if (distance < max_distance_threshold) {
            Eigen::Vector3f p_vec(p.x, p.y, p.z);
            Eigen::Vector3f diff = p_vec - line_point;
            float t = diff.dot(line_direction);
            t_values.push_back(t);
        }
    }

    if (t_values.size() >= 2) {
        std::sort(t_values.begin(), t_values.end());
        // 使用极值但稍微向内收缩一点
        float range = t_values.back() - t_values.front();
        float t_min = t_values.front() + range * 0.01f; // 从最小值向内收缩2%
        float t_max = t_values.back() - range * 0.01f;  // 从最大值向内收缩2%

        // 生成交线上的点
        float step = (t_max - t_min) / (num_points - 1);
        for (int i = 0; i < num_points; ++i) {
            float t = t_min + i * step;
            pcl::PointXYZ pt;
            pt.getVector3fMap() = line_point + t * line_direction;
            line_cloud->points.push_back(pt);
        }
    }

    line_cloud->width = line_cloud->points.size();
    line_cloud->height = 1;
    line_cloud->is_dense = true;

    return line_cloud;
}

// 计算两个平面的交线
bool computePlaneIntersection(const pcl::ModelCoefficients::Ptr& coeff1,
    const pcl::ModelCoefficients::Ptr& coeff2,
    pcl::ModelCoefficients::Ptr& line_coeff)
{
    // 确保输入是平面系数 (ax + by + cz + d = 0)
    if (coeff1->values.size() != 4 || coeff2->values.size() != 4) {
        return false;
    }

    // 提取平面法向量和常数项
    Eigen::Vector4f plane1(coeff1->values[0], coeff1->values[1], coeff1->values[2], coeff1->values[3]);
    Eigen::Vector4f plane2(coeff2->values[0], coeff2->values[1], coeff2->values[2], coeff2->values[3]);

    // 计算交线的方向向量（两个法向量的叉积）
    Eigen::Vector3f n1 = plane1.head<3>();
    Eigen::Vector3f n2 = plane2.head<3>();
    Eigen::Vector3f direction = n1.cross(n2);

    // 如果方向向量长度接近0，说明平面平行或重合
    if (direction.norm() < 1e-6) {
        return false;
    }
    direction.normalize();

    // 求解交线上的一点
    Eigen::Matrix3f A;
    A << n1[0], n1[1], n1[2],
        n2[0], n2[1], n2[2],
        0, 0, 1;

    Eigen::Vector3f b(-plane1[3], -plane2[3], 0);

    if (std::abs(A.determinant()) < 1e-6) {
        A << n1[0], n1[1], n1[2],
            n2[0], n2[1], n2[2],
            1, 0, 0;
        b[2] = 0;

        if (std::abs(A.determinant()) < 1e-6) {
            A << n1[0], n1[1], n1[2],
                n2[0], n2[1], n2[2],
                0, 1, 0;
            b[2] = 0;
        }
    }

    if (std::abs(A.determinant()) < 1e-6) {
        return false;
    }

    Eigen::Vector3f point = A.inverse() * b;

    // 存储交线参数
    line_coeff->values.resize(6);
    line_coeff->values[0] = point[0];
    line_coeff->values[1] = point[1];
    line_coeff->values[2] = point[2];
    line_coeff->values[3] = direction[0];
    line_coeff->values[4] = direction[1];
    line_coeff->values[5] = direction[2];

    return true;
}

// 使用RANSAC拟合平面
bool fitPlaneRANSAC(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
    pcl::ModelCoefficients::Ptr coefficients)
{
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(1000);
    seg.setDistanceThreshold(0.005);
    seg.setInputCloud(cloud);

    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    seg.segment(*inliers, *coefficients);

    return inliers->indices.size() > 0;
}

// 区域生长分割
void regionGrowingSegmentation(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane1,
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane2,
    pcl::ModelCoefficients::Ptr coeff1,
    pcl::ModelCoefficients::Ptr coeff2)
{
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());

    ne.setSearchMethod(tree);
    ne.setInputCloud(cloud);
    ne.setKSearch(30);
    ne.compute(*normals);

    pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
    reg.setMinClusterSize(50);
    reg.setMaxClusterSize(1000);
    reg.setSearchMethod(tree);
    reg.setNumberOfNeighbours(30);
    reg.setInputCloud(cloud);
    reg.setInputNormals(normals);
    reg.setSmoothnessThreshold(2.0 / 180.0 * M_PI);
    reg.setCurvatureThreshold(0.80);

    std::vector<pcl::PointIndices> clusters;
    reg.extract(clusters);

    std::cout << "区域生长找到 " << clusters.size() << " 个区域" << std::endl;

    std::sort(clusters.begin(), clusters.end(),
        [](const pcl::PointIndices& a, const pcl::PointIndices& b) {
            return a.indices.size() > b.indices.size();
        });

    for (size_t i = 0; i < std::min(clusters.size(), size_t(2)); i++) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
        for (const auto& idx : clusters[i].indices) {
            cluster->points.push_back(cloud->points[idx]);
        }
        cluster->width = cluster->points.size();
        cluster->height = 1;
        cluster->is_dense = true;

        if (i == 0) {
            *plane1 = *cluster;
            if (!fitPlaneRANSAC(plane1, coeff1)) {
                std::cout << "无法拟合平面1的方程" << std::endl;
            }
        }
        else {
            *plane2 = *cluster;
            if (!fitPlaneRANSAC(plane2, coeff2)) {
                std::cout << "无法拟合平面2的方程" << std::endl;
            }
        }

        std::cout << "区域 " << i + 1 << " 包含 " << cluster->points.size() << " 个点" << std::endl;
    }
}

// 剩余点云提取
pcl::PointCloud<pcl::PointXYZ>::Ptr extractRemainingCloud(
    pcl::PointCloud<pcl::PointXYZ>::Ptr original_cloud,
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane1,
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane2)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr remaining_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    if (plane1->empty() && plane2->empty()) {
        *remaining_cloud = *original_cloud;
        return remaining_cloud;
    }

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(original_cloud);

    std::vector<bool> is_plane_point(original_cloud->size(), false);

    for (const auto& point : plane1->points) {
        std::vector<int> pointIdxNKNSearch(1);
        std::vector<float> pointNKNSquaredDistance(1);
        if (kdtree.nearestKSearch(point, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
            if (pointNKNSquaredDistance[0] < 0.001f) {
                is_plane_point[pointIdxNKNSearch[0]] = true;
            }
        }
    }

    for (const auto& point : plane2->points) {
        std::vector<int> pointIdxNKNSearch(1);
        std::vector<float> pointNKNSquaredDistance(1);
        if (kdtree.nearestKSearch(point, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
            if (pointNKNSquaredDistance[0] < 0.001f) {
                is_plane_point[pointIdxNKNSearch[0]] = true;
            }
        }
    }

    for (size_t i = 0; i < original_cloud->size(); ++i) {
        if (!is_plane_point[i]) {
            remaining_cloud->points.push_back(original_cloud->points[i]);
        }
    }

    remaining_cloud->width = remaining_cloud->points.size();
    remaining_cloud->height = 1;
    remaining_cloud->is_dense = true;

    return remaining_cloud;
}

// 平移交线点云函数
pcl::PointCloud<pcl::PointXYZ>::Ptr shiftIntersectionLine(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& line_cloud,
    float shift_distance = 0.0015f) // 默认平移1.5mm
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr shifted_line(new pcl::PointCloud<pcl::PointXYZ>);

    if (line_cloud->empty()) {
        return shifted_line;
    }

    // 计算两个平面的平均法向量方向（假设向上是Z轴正方向）
    // 这里我们简单地将所有点沿Z轴正方向平移
    for (const auto& point : line_cloud->points) {
        pcl::PointXYZ shifted_point = point;
        shifted_point.z += shift_distance; // 向上平移
        shifted_line->points.push_back(shifted_point);
    }

    shifted_line->width = shifted_line->points.size();
    shifted_line->height = 1;
    shifted_line->is_dense = true;

    return shifted_line;
}

// 主处理函数
bool processVGrooveAndExtractLine(const std::string& input_file, const std::string& output_file)
{
    // 创建输出目录
    fs::path output_path(output_file);
    fs::create_directories(output_path.parent_path());

    // 加载点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(input_file, *cloud) == -1)
    {
        std::cerr << "无法读取文件 " << input_file << std::endl;
        return false;
    }
    std::cout << "从 " << input_file << " 加载了 " << cloud->points.size() << " 个点" << std::endl;

    //// 下采样
    //pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
    //voxel_grid.setInputCloud(cloud);
    //float leafSize = 0.002f; // 2mm
    //voxel_grid.setLeafSize(leafSize, leafSize, leafSize);
    //pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZ>);
    //voxel_grid.filter(*cloud_downsampled);
    //std::cout << "下采样后点数: " << cloud_downsampled->points.size() << std::endl;

    // 初始化变量
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane1(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane2(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::ModelCoefficients::Ptr coeff1(new pcl::ModelCoefficients);
    pcl::ModelCoefficients::Ptr coeff2(new pcl::ModelCoefficients);
    pcl::ModelCoefficients::Ptr intersection_line(new pcl::ModelCoefficients);
    pcl::PointCloud<pcl::PointXYZ>::Ptr line_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr shifted_line_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    // 区域生长分割
    std::cout << "尝试区域生长分割..." << std::endl;
    regionGrowingSegmentation(cloud, plane1, plane2, coeff1, coeff2);

    // 计算交线
    if (plane1->points.size() > 0 && plane2->points.size() > 0 &&
        coeff1->values.size() == 4 && coeff2->values.size() == 4) {

        std::cout << "平面1方程: " << coeff1->values[0] << "x + "
            << coeff1->values[1] << "y + " << coeff1->values[2]
            << "z + " << coeff1->values[3] << " = 0" << std::endl;

        std::cout << "平面2方程: " << coeff2->values[0] << "x + "
            << coeff2->values[1] << "y + " << coeff2->values[2]
            << "z + " << coeff2->values[3] << " = 0" << std::endl;

        if (computePlaneIntersection(coeff1, coeff2, intersection_line)) {
            std::cout << "交线找到: 点(" << intersection_line->values[0] << ", "
                << intersection_line->values[1] << ", " << intersection_line->values[2]
                << "), 方向(" << intersection_line->values[3] << ", "
                << intersection_line->values[4] << ", " << intersection_line->values[5] << ")" << std::endl;

            // 生成交线点云
           // line_cloud = generateSimpleIntersectionLine(intersection_line, plane1, plane2, 200);

           // if (line_cloud->empty()) {
                std::cout << "简单方法失败，尝试精确方法..." << std::endl;
                line_cloud = generateExactIntersectionLine(intersection_line, plane1, plane2, 200);
            //}

            std::cout << "生成交线点云，包含 " << line_cloud->points.size() << " 个点" << std::endl;

            // 平移交线（向上平移1-2mm）
            float shift_distance = 0.002f; // 2mm
            shifted_line_cloud = shiftIntersectionLine(line_cloud, shift_distance);

            // 保存结果
            if (pcl::io::savePCDFile(output_file, *shifted_line_cloud) == 0) {
                std::cout << "交线点云保存到 " << output_file << std::endl;
                return true;
            }
            else {
                std::cerr << "无法保存文件 " << output_file << std::endl;
                return false;
            }
        }
        else {
            std::cout << "无法计算平面交线" << std::endl;
            return false;
        }
    }
    else {
        std::cout << "无法找到两个平面" << std::endl;
        return false;
    }
}

