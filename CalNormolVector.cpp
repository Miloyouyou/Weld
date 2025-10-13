#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <cmath>
#include <vector>
#include <thread>
#include <vtkOutputWindow.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/centroid.h>
#include <Eigen/Dense>

using namespace std;

void printVectorInfo(const string& name, const Eigen::Vector3f& vec)
{
	cout << name << "向量: (" << vec[0] << ", " << vec[1] << ", " << vec[2] << ")" << endl;
	cout << name << "向量长度: " << vec.norm() << endl;
}

Eigen::Vector3f calculateWeldingDirection(const std::string& file_path, bool show_visualization = false, Eigen::Vector3f x_axi = Eigen::Vector3f::Zero())
{
	vtkOutputWindow::SetGlobalWarningDisplay(0);

	// 加载点云
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	if (pcl::io::loadPCDFile<pcl::PointXYZ>(file_path, *cloud) == -1)
	{
		PCL_ERROR("读取源点云失败\n");
		throw runtime_error("Failed to load point cloud.");
	}
	cout << "加载点数：" << cloud->size() << endl;
	// 创建原始点云数据的副本
	// 复制原始点云数据到新变量cloud_original中，避免直接引用同一内存地址
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_original(new pcl::PointCloud<pcl::PointXYZ>(*cloud));

	// === 新增加的部分1：计算整个点云的质心 ===
	Eigen::Vector4f cloud_centroid;
	// 计算输入点云*cloud_original中所有点的三维几何中心
	pcl::compute3DCentroid(*cloud_original, cloud_centroid);
	Eigen::Vector3f viewpoint(cloud_centroid[0], cloud_centroid[1], cloud_centroid[2]);
	cout << "整个点云质心坐标: (" << viewpoint[0] << ", " << viewpoint[1] << ", " << viewpoint[2] << ")" << endl;

	pcl::PointCloud<pcl::PointXYZ>::Ptr planar_segment(new pcl::PointCloud<pcl::PointXYZ>);
	// 存储平面、球体等几何模型的参数信息
	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
	// 存储内点或特定点集的索引信息
	// 智能指针可以自动管理内存，避免内存泄漏
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
	// 通过随机采样一致性算法(SAC)来拟合模型并分割出符合特定几何模型的点云子集
	pcl::SACSegmentation<pcl::PointXYZ> seg;
	// 用于根据给定的索引集合从输入点云中提取或移除特定点
	pcl::ExtractIndices<pcl::PointXYZ> extract;

	// 分割块数为2
	int n_piece = 2;
	// 保存点云的法线信息
	std::vector<Eigen::Vector3f> normals;
	// 保存聚类中心点位置
	std::vector<pcl::PointXYZ> centroids;
	// 该变量用于创建和管理3D点云数据的可视化窗口
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
	if (show_visualization)
	{
		viewer.reset(new pcl::visualization::PCLVisualizer("Plane Normals Viewer"));
		viewer->getRenderWindow()->GlobalWarningDisplayOff();
		viewer->setBackgroundColor(0, 0, 0);
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> base_color(cloud, 255, 255, 255);
		viewer->addPointCloud<pcl::PointXYZ>(cloud, base_color, "original_cloud");
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "original_cloud");
	}

	std::vector<std::tuple<int, int, int>> colors = {
		{255, 0, 0},   // 红
		{0, 0, 255}    // 蓝
	};

	for (int i = 0; i < n_piece; ++i)
	{
		// 启用系数优化功能,提高计算精度和收敛速度
		seg.setOptimizeCoefficients(true);
		// 指定使用平面模型进行拟合
		seg.setModelType(pcl::SACMODEL_PLANE);
		// 使用RANSAC算法进行模型拟合和异常点剔除
		seg.setMethodType(pcl::SAC_RANSAC);
		// 设置距离阈值为1.0
		seg.setDistanceThreshold(1.0);
		// 设置输入点云
		seg.setInputCloud(cloud);
		// 根据之前设置的分割模型和参数
		// inliers 指向内点索引的指针，存储分割后符合模型的点的索引
		// coefficients 指向模型系数的指针，存储分割得到的几何模型参数
		seg.segment(*inliers, *coefficients);

		if (inliers->indices.empty())
		{
			cout << "未检测到平面。" << endl;
			break;
		}

		extract.setInputCloud(cloud);
		// 设置点云提取对象的内点索引
		extract.setIndices(inliers);
		// 关闭负向提取模式，使提取器只保留满足条件的点，而不是排除它们
		extract.setNegative(false);
		// 使用指定的平面分割器对点云进行滤波处理
		// 提取出符合平面模型的点集
		extract.filter(*planar_segment);
		// 启用负向提取，即提取不符合指定条件的数据点
		extract.setNegative(true);
		// 对输入的点云数据*cloud应用滤波操作
		extract.filter(*cloud);
		// 将平面方程的法向量系数转换为标准化的法向量表示
		Eigen::Vector3f normal(coefficients->values[0], coefficients->values[1], coefficients->values[2]);
		normal.normalize();

		// === 新增加的部分2：确保法向量朝向传感器位置 ===
		Eigen::Vector4f centroid4f;
		// 4维向量存储计算得到的质心坐标(x, y, z, 1)
		pcl::compute3DCentroid(*planar_segment, centroid4f);
		pcl::PointXYZ centroid(centroid4f[0], centroid4f[1], centroid4f[2]);

		// 计算从质心指向传感器位置的向量
		Eigen::Vector3f point_to_view = viewpoint - Eigen::Vector3f(centroid.x, centroid.y, centroid.z);

		// 确保法向量朝向外部（指向传感器位置）
		//if (normal.dot(point_to_view) < 0) {
		//    normal = -normal;
		//    cout << "平面 " << i << " 法向量已翻转（朝向传感器）" << endl;
		//}

		// 原有的基于前一个质心的翻转逻辑
		if (!centroids.empty()) {
			//  centroids.back()：获取容器中最后一个元素
			Eigen::Vector3f prev_cent(centroids.back().x, centroids.back().y, centroids.back().z);
			if (normal.dot(prev_cent) > 0) {
				// 当点积大于0时，说明两个法向量夹角小于90度，方向基本一致
				// 为了保持法向量指向相反方向的约束，将当前法向量取反
				normal = -normal;
				cout << "基于前一个平面翻转平面 " << i << " 法向量" << endl;
			}
		}
		// 将计算得到的法向量(normal)和质心坐标(centroid)分别添加到对应的容器中
		// normals.push_back()将法向量存入法向量集合
		// centroids.push_back()将质心坐标存入质心集合
		normals.push_back(normal);
		centroids.push_back(centroid);

		if (show_visualization)
		{
			auto [R, G, B] = colors[i];
			stringstream ss; ss << "plane_" << i;
			// 使同一分割区域的点云显示为统一颜色，便于区分不同分割结果
			pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> seg_color(planar_segment, R, G, B);
			// 将分割出的平面点云数据添加到PCL可视化器中进行显示
			viewer->addPointCloud<pcl::PointXYZ>(planar_segment, seg_color, ss.str());
			// 设置点云的渲染大小3
			viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, ss.str());
			// 计算垂直于平面的箭头终点坐标
			pcl::PointXYZ p2(centroid.x + 20 * normal[0], centroid.y + 20 * normal[1],
				centroid.z + 20 * normal[2]);
			string arrow_id = "normal_arrow_" + std::to_string(i);
			// 在点云可视化窗口中添加一个箭头对象
			viewer->addArrow(p2, centroid, R / 255.0, G / 255.0, B / 255.0, false, arrow_id);
		}
	}

	// 焊枪方向计算
	Eigen::Vector3f welding_direction = Eigen::Vector3f::Zero();
	if (normals.size() == 2 && centroids.size() == 2)
	{
		Eigen::Vector3f sum_normal = normals[0] + normals[1];
		sum_normal.normalize();

		if (sum_normal.z() < 0) {
			sum_normal = -sum_normal;
			cout << "焊枪方向Z分量为正，已自动反转" << endl;
		}

		welding_direction = sum_normal;

		if (show_visualization)
		{
			// 计算中点：通过两个质心点centroids[0]和centroids[1]的坐标平均值
			// 计算出它们的中点mid_point
			pcl::PointXYZ mid_point(
				(centroids[0].x + centroids[1].x) / 2,
				(centroids[0].y + centroids[1].y) / 2,
				(centroids[0].z + centroids[1].z) / 2
			);
			// 计算终点：以中点为起点，沿着焊接方向welding_direction延伸20个单位长度
			// 计算出终点sum_end
			pcl::PointXYZ sum_end(
				mid_point.x + 20 * welding_direction[0],
				mid_point.y + 20 * welding_direction[1],
				mid_point.z + 20 * welding_direction[2]
			);

			viewer->addArrow(sum_end, mid_point, 0.0, 1.0, 0.0, false, "sum_normal_arrow");
			viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 4, "sum_normal_arrow");
			viewer->addText3D("sum_normal", sum_end, 5, 0.0, 1.0, 0.0, "sum_normal_label");

			////        // === 平面交线单位向量计算 ===
			// 计算两个平面法向量的叉积
			Eigen::Vector3f cross_vec = normals[0].cross(normals[1]);
			// 用于检测两个向量是否平行或共线，因为平行向量的叉积为零向量
			if (cross_vec.norm() < 1e-6) {
				cout << "两个法向量几乎平行，无法计算交线。" << endl;
			}
			else {
				cross_vec.normalize();
				cout << "\n=== 平面交线单位方向向量 ===" << endl;
				printVectorInfo("交线", cross_vec);
				Eigen::Vector3f x_axis;
				x_axis << mid_point.x + 20 * cross_vec[0],
					mid_point.y + 20 * cross_vec[1],
					mid_point.z + 20 * cross_vec[2]; // 假设x轴为[1, 0, 0]
				/*pcl::PointXYZ line_end(
					mid_point.x + 20 * cross_vec[0],
					mid_point.y + 20 * cross_vec[1],
					mid_point.z + 20 * cross_vec[2]
				);*/

				//    viewer->addArrow(line_end, mid_point, 1.0, 1.0, 0.0, false, "cross_line_arrow");
				//    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "cross_line_arrow");
				//    viewer->addText3D("交线方向", line_end, 0.5, 1.0, 1.0, 0.0, "cross_label");

			}
		}
	}

	// 启动可视化（如果需要）
	if (show_visualization)
	{
		// 检测可视化窗口是否被用户关闭
		while (!viewer->wasStopped())
		{
			viewer->spinOnce(100);
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
		}
	}

	return welding_direction;
}
