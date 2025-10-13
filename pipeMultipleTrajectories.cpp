#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <thread>
#include <random>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_circle3D.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/centroid.h>
#include <Eigen/Geometry>
#include <vtkOutputWindow.h>

using namespace std;

// 随机颜色生成器
vector<float> pipegenerateRandomColor() {
    static random_device rd;
    static mt19937 gen(rd());
    static uniform_real_distribution<float> dis(0.4f, 1.0f); // 避免太暗的颜色

    return { dis(gen), dis(gen), dis(gen) };
}

// 计算旋转轴（打底轨迹端点连线方向）
Eigen::Vector3f pipecalculateRotationAxis(const pcl::PointCloud<pcl::PointNormal>::Ptr& path) {
    if (path->size() < 2) return Eigen::Vector3f::UnitZ();
    Eigen::Vector3f start = path->points.front().getVector3fMap();
    Eigen::Vector3f end = path->points.back().getVector3fMap();
    return (end - start).normalized();
}

// 生成带法向的单层单道焊接路径（绕端点轴旋转）
void pipegenerateSinglePassPath(
    const pcl::PointCloud<pcl::PointNormal>::Ptr& base_path,
    const Eigen::Vector3f& center,
    const Eigen::Vector3f& normal,
    float base_radius,
    float layer_offset,
    float pass_angle_deg,
    pcl::PointCloud<pcl::PointNormal>::Ptr& output_path)
{
    // 获取旋转轴（打底轨迹端点方向）
    Eigen::Vector3f rotation_axis = pipecalculateRotationAxis(base_path);
    float current_radius = base_radius + layer_offset;
    float pass_angle_rad = pass_angle_deg * M_PI / 180.0f;

    for (const auto& point : base_path->points) {
        Eigen::Vector3f pt = point.getVector3fMap();
        Eigen::Vector3f dir = (pt - center).normalized();

        // 绕端点轴旋转方向向量
        Eigen::Vector3f new_dir = Eigen::AngleAxisf(pass_angle_rad, rotation_axis) * dir;
        Eigen::Vector3f new_pt = center + current_radius * new_dir;

        pcl::PointNormal pn;
        pn.x = new_pt.x();
        pn.y = new_pt.y();
        pn.z = new_pt.z();
        pn.normal_x = normal.x();
        pn.normal_y = normal.y();
        pn.normal_z = normal.z();

        output_path->points.push_back(pn);
    }

    output_path->width = output_path->points.size();
    output_path->height = 1;
    output_path->is_dense = true;
}

// 生成姿态文件（更新坐标系计算）
void pipegeneratePoseFile(
    const pcl::PointCloud<pcl::PointNormal>::Ptr& path,
    const Eigen::Vector3f& center,
    const Eigen::Vector3f& rotation_axis,
    const string& filename)
{
    ofstream ofs(filename);
    if (!ofs.is_open()) {
        cerr << "Error: Could not open pose file " << filename << endl;
        return;
    }

    for (size_t i = 0; i < path->points.size(); ++i) {
        Eigen::Vector3f p_n = path->points[i].getVector3fMap();

        // Z轴：从圆心指向当前点
        Eigen::Vector3f z_dir = (p_n - center).normalized();

        // X轴：旋转轴方向（端点连线）
        Eigen::Vector3f x_dir = rotation_axis.normalized();

        // Y轴：Z × X
        Eigen::Vector3f y_dir = z_dir.cross(x_dir).normalized();

        // 重新正交化X轴
        x_dir = y_dir.cross(z_dir).normalized();

        // 计算欧拉角 (ZYX顺序)
        Eigen::Matrix3f R;
        R.col(0) = x_dir;
        R.col(1) = y_dir;
        R.col(2) = z_dir;
        Eigen::Vector3f euler_deg = R.eulerAngles(2, 1, 0) * 180.0f / M_PI;

        // 写入文件：X Y Z Roll Pitch Yaw
        ofs << p_n.x() << " " << p_n.y() << " " << p_n.z() << " "
            << euler_deg[2] << " " << euler_deg[1] << " " << euler_deg[0] << "\n";
    }
    ofs.close();
    cout << "Saved pose file: " << filename << endl;
}

// 生成完整焊接层（带法向）
void pipegenerateWeldingLayer(
    const pcl::PointCloud<pcl::PointNormal>::Ptr& base_path,
    const Eigen::Vector3f& center,
    const Eigen::Vector3f& normal,
    float base_radius,
    float layer_offset,
    int pass_count,
    float max_angle_deg,
    vector<pcl::PointCloud<pcl::PointNormal>::Ptr>& output_paths,
    const string& output_dir)
{
    Eigen::Vector3f rotation_axis = pipecalculateRotationAxis(base_path);
    vector<float> pass_angles;

    // 计算对称分布的角度
    if (pass_count == 1) {
        pass_angles.push_back(0.0f);
    }
    else {
        float angle_step = 2 * max_angle_deg / (pass_count - 1);
        for (int i = 0; i < pass_count; ++i) {
            pass_angles.push_back(-max_angle_deg + i * angle_step);
        }
    }

    // 生成每道路径
    for (size_t pass_idx = 0; pass_idx < pass_angles.size(); ++pass_idx) {
        pcl::PointCloud<pcl::PointNormal>::Ptr path(new pcl::PointCloud<pcl::PointNormal>);
        pipegenerateSinglePassPath(base_path, center, normal, base_radius,
            layer_offset, pass_angles[pass_idx], path);

        // 保存点云和姿态文件
        string pcd_filename = output_dir + "Flat_Path_Layer" + to_string(output_paths.size()) + ".pcd";
        string pose_filename = output_dir + to_string(output_paths.size()) + ".txt";

        pcl::io::savePCDFile(pcd_filename, *path, true);
        pipegeneratePoseFile(path, center, rotation_axis, pose_filename);

        output_paths.push_back(path);
    }
}

/**
 * @brief 生成坡口多层多道焊接路径
 *
 * @param keypoints_file 关键点文件路径
 * @param output_dir 输出目录
 * @param visualize 是否可视化结果
 * @return int 成功返回0，失败返回-1
 */
int pipegenerateWeldingPaths(
    const std::string& input_file, 
    const string& keypoints_file,
    const string& output_dir ,
    float distanceThreshold,
    bool visualize = true)
{

    // 1. 加载点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile(keypoints_file, *cloud) < 0) {
        cerr << "Error: Couldn't read point cloud file!" << endl;
        return -1;
    }

    // 2. 拟合3D圆
    pcl::SampleConsensusModelCircle3D<pcl::PointXYZ>::Ptr model_circle3D(
        new pcl::SampleConsensusModelCircle3D<pcl::PointXYZ>(cloud));
    pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model_circle3D);
    ransac.setDistanceThreshold(0.65);
    ransac.computeModel();

    pcl::IndicesPtr inliers(new vector<int>());
    ransac.getInliers(*inliers);

    // 转换为带法向的点云
    pcl::PointCloud<pcl::PointNormal>::Ptr circle_3D(new pcl::PointCloud<pcl::PointNormal>);
    pcl::copyPointCloud<pcl::PointXYZ>(*cloud, *inliers, *circle_3D);

    Eigen::VectorXf coeff;
    ransac.getModelCoefficients(coeff);
    Eigen::Vector3f center(coeff[0], coeff[1], coeff[2]);
    float radius = coeff[3];
    Eigen::Vector3f normal(coeff[4], coeff[5], coeff[6]);

    // 统一法向量方向
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*circle_3D, centroid);
    if (normal.dot(Eigen::Vector3f(centroid.head<3>()) - center) < 0)
        normal = -normal;
    normal.normalize();

    // 为内点添加法向量
    for (auto& point : circle_3D->points) {
        point.normal_x = normal.x();
        point.normal_y = normal.y();
        point.normal_z = normal.z();
    }

    // 3. 生成打底层路径（带法向）
    pcl::PointCloud<pcl::PointNormal>::Ptr base_path(new pcl::PointCloud<pcl::PointNormal>);
    Eigen::Vector3f u = normal.unitOrthogonal();
    Eigen::Vector3f v = normal.cross(u).normalized();

    vector<pair<float, pcl::PointNormal>> angle_points;
    for (size_t i = 0; i < circle_3D->size(); ++i) {
        Eigen::Vector3f pt = circle_3D->points[i].getVector3fMap();
        Eigen::Vector3f vec = pt - center;
        vec = vec - normal.dot(vec) * normal;
        vec.normalize();
        float angle = atan2(vec.dot(v), vec.dot(u));
        angle_points.emplace_back(angle, circle_3D->points[i]);
    }

    // 按角度排序
    sort(angle_points.begin(), angle_points.end(),
        [](const pair<float, pcl::PointNormal>& a, const pair<float, pcl::PointNormal>& b) {
            return a.first < b.first;
        });

    float angle_start = angle_points.front().first;
    float angle_end = angle_points.back().first;

    // 生成圆弧点（带法向）
    int arc_points = 100;
    for (int i = 0; i <= arc_points; ++i) {
        float angle = angle_start + (angle_end - angle_start) * float(i) / float(arc_points);
        Eigen::Vector3f pt = center + radius * (cos(angle) * u + sin(angle) * v);

        pcl::PointNormal pn;
        pn.x = pt.x();
        pn.y = pt.y();
        pn.z = pt.z();
        pn.normal_x = normal.x();
        pn.normal_y = normal.y();
        pn.normal_z = normal.z();

        base_path->points.push_back(pn);
    }
    base_path->width = base_path->points.size();
    base_path->height = 1;
    base_path->is_dense = true;

    // 4. 创建输出目录
    //const string output_dir = "PointCloud/S/Paths/";

    // 保存打底层（带法向）
    vector<pcl::PointCloud<pcl::PointNormal>::Ptr> all_paths;
    pcl::io::savePCDFile(output_dir+"Flat_Path_Layer" + "0.pcd", *base_path, true);
    pipegeneratePoseFile(base_path, center, pipecalculateRotationAxis(base_path),
        output_dir + "0.txt");
    all_paths.push_back(base_path);
    cout << "Generated base layer (radius: " << radius << "mm)" << endl;

    // 5. 生成多层焊接路径（优化第四层为5道）
    struct LayerConfig {
        float offset_mm;
        int pass_count;
        float max_angle_deg;
        string layer_name;
    };

    vector<LayerConfig> layers = {
        {3.0f,  2, 1.0f, "Layer2"},  // 第二层：2道，±2°
        {6.0f,  3, 2.0f, "Layer3"},  // 第三层：3道，±2°（含中道）
        {8.0f,  5, 3.2f, "Layer4"}   // 第四层：5道，±5°（新增两道）
    };

    for (const auto& layer : layers) {
        cout << "Generating " << layer.layer_name
            << " (offset:" << layer.offset_mm << "mm, "
            << "passes:" << layer.pass_count << ", "
            << "angle:" << layer.max_angle_deg << "°)" << endl;

        pipegenerateWeldingLayer(base_path, center, normal, radius,
            layer.offset_mm, layer.pass_count,
            layer.max_angle_deg, all_paths, output_dir);
    }
    if (visualize) {
        // 6. 可视化
        pcl::visualization::PCLVisualizer::Ptr viewer(
            new pcl::visualization::PCLVisualizer("Multi-layer Circular Welding Paths"));
        vtkOutputWindow::SetGlobalWarningDisplay(0);
        viewer->setBackgroundColor(0.1, 0.1, 0.2);

        // 显示所有路径（转换为XYZ显示）
        for (size_t i = 0; i < all_paths.size(); ++i) {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::copyPointCloud(*all_paths[i], *cloud_xyz);

            string name = "path_" + to_string(i);

            // 为每条路径生成随机颜色
            vector<float> color = pipegenerateRandomColor();
            float r = color[0];
            float g = color[1];
            float b = color[2];

            viewer->addPointCloud<pcl::PointXYZ>(cloud_xyz, name);
            viewer->setPointCloudRenderingProperties(
                pcl::visualization::PCL_VISUALIZER_COLOR,
                r, g, b, name);
            viewer->setPointCloudRenderingProperties(
                pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, name);

            // 显示法向量（每隔5个点显示一个）
            viewer->addPointCloudNormals<pcl::PointNormal>(all_paths[i], 5, 1.0, "normals_" + name);

            // 在路径起点添加标签
            pcl::PointXYZ start_point = cloud_xyz->points[0];
            viewer->addText3D("Path " + to_string(i), start_point, 0.3, r, g, b, "label_" + name);
        }

        // 添加坐标系
        viewer->addCoordinateSystem(50.0);
        viewer->addText("Multi-layer Circular Welding Paths", 10, 10, 20, 1, 1, 1);
        viewer->addText("Press 'Q' to exit", 10, 35, 15, 1, 1, 1);
        viewer->addText("Each path has random color", 10, 60, 15, 1, 1, 1);

        cout << "\nVisualization window opened..." << endl;
        while (!viewer->wasStopped()) {
            viewer->spinOnce(100);
            this_thread::sleep_for(chrono::milliseconds(100));
        }
	}
    return 0;

}
