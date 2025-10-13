#include <iostream>
#include <vector>
#include <string>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_line.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <vtkOutputWindow.h>
#include <thread>
#include "CalNormolVector.h"
using namespace std;

// 辅助函数声明
void generatePoseFile(const pcl::PointCloud<pcl::PointXYZ>::Ptr& path,
	const Eigen::Vector3f& line_dir,
	const Eigen::Vector3f& normal,
	const string& filename);

void generateSinglePassPath(const pcl::PointCloud<pcl::PointXYZ>::Ptr& base_path,
	const Eigen::Vector3f& line_dir,
	const Eigen::Vector3f& normal,
	float lateral_offset,
	float vertical_offset,
	float angle_deg,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& output_path);


/**
 * @brief 生成坡口多层多道焊接路径
 *
 * @param keypoints_file 关键点文件路径
 * @param output_dir 输出目录
 * @param visualize 是否可视化结果
 * @return int 成功返回0，失败返回-1
 */
int VpogenerateWeldingPaths(
	const std::string& input_file,
	const string& keypoints_file,
	const string& output_dir,
	float distanceThreshold,
	bool visualize = true)
{
	// 1. 加载关键点云
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	if (pcl::io::loadPCDFile(keypoints_file, *cloud) < 0) {
		cerr << "Error: Couldn't read point cloud file: " << keypoints_file << endl;
		return -1;
	}

	// 2. RANSAC拟合直线
	// 创建一个用于直线拟合的RANSAC模型对象
	pcl::SampleConsensusModelLine<pcl::PointXYZ>::Ptr model_line(
		new pcl::SampleConsensusModelLine<pcl::PointXYZ>(cloud));
	// 创建了一个基于随机采样一致性的算法对象
	// 通过随机采样点集来拟合直线模型
	pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model_line);
	// 设置RANSAC算法的距离阈值
	ransac.setDistanceThreshold(distanceThreshold);
	// 通过鲁棒的迭代方式从含噪声的数据中估计数学模型
	ransac.computeModel();

	// 得到符合RANSAC拟合模型的所有内点组成的点云数据
	pcl::IndicesPtr inliers(new vector<int>());
	ransac.getInliers(*inliers);
	pcl::PointCloud<pcl::PointXYZ>::Ptr line_points(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::copyPointCloud<pcl::PointXYZ>(*cloud, *inliers, *line_points);
	// 从RANSAC拟合结果中提取三维直线的几何参数
	Eigen::VectorXf coeff;
	ransac.getModelCoefficients(coeff);
	Eigen::Vector3f point_on_line(coeff[0], coeff[1], coeff[2]);
	Eigen::Vector3f line_direction(coeff[3], coeff[4], coeff[5]);
	//line_direction.normalize();

	// 3. 平面法向量
	Eigen::Vector3f line_direction1;
	Eigen::Vector3f normalTemp = calculateWeldingDirection(input_file, visualize, line_direction1);
	line_direction.normalize();
	std::cout << "normalTemp1" << normalTemp[0] << std::endl;
	std::cout << "normalTemp2" << normalTemp[1] << std::endl;
	std::cout << "normalTemp3" << normalTemp[2] << std::endl;
	/*return 0;*/
	Eigen::Vector3f normal(normalTemp[0], normalTemp[1], normalTemp[2]);
	normal.normalize();


	// 4. 生成打底层路径
	vector<pair<float, pcl::PointXYZ>> sorted_points;
	// 实现对points容器中每个元素的遍历，point依次引用容器中的每个点对象
	for (const auto& point : line_points->points) {
		// point对象转换为3维浮点向量
		Eigen::Vector3f pt = point.getVector3fMap();
		float projection = (pt - point_on_line).dot(line_direction);
		sorted_points.emplace_back(projection, point);
	}
	// 升序排列
	sort(sorted_points.begin(), sorted_points.end(),
		[](const auto& a, const auto& b) { return a.first < b.first; });

	pcl::PointCloud<pcl::PointXYZ>::Ptr base_path(new pcl::PointCloud<pcl::PointXYZ>);
	Eigen::Vector3f start_point = sorted_points.front().second.getVector3fMap();
	Eigen::Vector3f end_point = sorted_points.back().second.getVector3fMap();

	// 通过线性插值在起点和终点之间均匀生成一系列中间点
	int num_points = 100;
	for (int i = 0; i <= num_points; ++i) {
		float t = static_cast<float>(i) / num_points;
		Eigen::Vector3f pt = start_point + t * (end_point - start_point);
		base_path->points.emplace_back(pt.x(), pt.y(), pt.z());
	}
	base_path->width = base_path->points.size();
	base_path->height = 1;
	base_path->is_dense = true;

	// 5. 创建输出目录
	system(("mkdir -p " + output_dir).c_str());

	// 保存打底层
	vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> all_paths;
	pcl::io::savePCDFile(output_dir + "Flat_Path_Layer0.pcd", *base_path, true);
	generatePoseFile(base_path, line_direction, normal,
		output_dir + "L11.txt");
	all_paths.push_back(base_path);

	// 6. 多层多道焊接参数
	struct WeldingPass {
		float vertical_offset; // 垂直偏移（层高）
		float lateral_offset;  // 横向偏移
		float angle_deg;       // 倾斜角度
		string name;          //几层几道名称
	};

	vector<vector<WeldingPass>> layers = {
		// 第二层（2道）
		{
			{2.0f, 0.0f, 0.0f,"L21"},  // 中道
			//{4.0f, 1.5f, 0.0f,"L22"}    // 右道
		},
		// 第三层（3道）
		{
			{5.0f, 0.0f, 0.0f,"L31"},  // 左道
			//{8.0f, 4.0f, 0.0f,"L32"},  // 右道
			//{6.0f, 0.0f, 0.0f,"L33" }   // 中道
		},
		// 第四层（1道）
		{
			{8.0f, 0.0f, 0.0f,"L41"},  // 左道
			//{6.0f, 4.0f, 0.0f,"L42"},  // 右道
			//{9.0f, 0.0f, 0.0f,"L41" },
		},
		{
			{11.0f, 0.0f, 0.0f,"L51"},  // 左道
			//{6.0f, 4.0f, 0.0f,"L42"},  // 右道
			//{9.0f, 0.0f, 0.0f,"L41" },
		}
	};

	// 7. 生成多层多道路径
	for (size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
		for (size_t pass_idx = 0; pass_idx < layers[layer_idx].size(); ++pass_idx) {
			const auto& pass = layers[layer_idx][pass_idx];
			pcl::PointCloud<pcl::PointXYZ>::Ptr path(new pcl::PointCloud<pcl::PointXYZ>);

			generateSinglePassPath(base_path, line_direction, normal,
				pass.lateral_offset, pass.vertical_offset,
				pass.angle_deg, path);

			string pcd_filename = output_dir + "Flat_Path_Layer" +
				to_string(all_paths.size()) + ".pcd";
			string pose_filename = output_dir + "" +
				pass.name + ".txt";

			pcl::io::savePCDFile(pcd_filename, *path, true);
			generatePoseFile(path, line_direction, normal, pose_filename);
			all_paths.push_back(path);
		}
	}

	// 8. 可视化
	if (visualize) {
		pcl::visualization::PCLVisualizer::Ptr viewer(
			new pcl::visualization::PCLVisualizer("Flat Multi-layer Welding Paths"));
		vtkOutputWindow::SetGlobalWarningDisplay(0);
		viewer->getRenderWindow()->GlobalWarningDisplayOff();
		viewer->setBackgroundColor(0.1, 0.1, 0.2);

		// 颜色配置（打底层 + 3层×多道）
		vector<vector<float>> colors = {
			{0, 1, 1},    // 打底层 - 青色
			{1, 0, 1},    // 第一层左道 - 紫色
			{1, 1, 0},    // 第一层右道 - 黄色
			{0, 1, 0},    // 第二层左道 - 绿色
			{1, 0, 0},    // 第二层中道 - 红色
			{0, 0, 1},    // 第二层右道 - 蓝色
			{1, 0.5, 0},  // 第三层第1道 - 橙色
			{0, 0.5, 1},  // 第三层第2道 - 天蓝
			{0.5, 0, 1},  // 第三层第3道 - 紫罗兰
			{1, 0, 0.5}   // 第三层第4道 - 粉红
		};

		// 显示所有路径
		for (size_t i = 0; i < all_paths.size(); ++i) {
			string name = "path_" + to_string(i);
			viewer->addPointCloud<pcl::PointXYZ>(all_paths[i], name);
			viewer->setPointCloudRenderingProperties(
				pcl::visualization::PCL_VISUALIZER_COLOR,
				colors[i][0], colors[i][1], colors[i][2], name);
			viewer->setPointCloudRenderingProperties(
				pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, name);
		}

		// 添加坐标系和说明
		viewer->addCoordinateSystem(50.0);
		viewer->addText("Flat Welding Paths - Press 'Q' to exit", 10, 10, 20, 1, 1, 1);

		cout << "\nVisualization window opened..." << endl;
		while (!viewer->wasStopped()) {
			viewer->spinOnce(100);
			this_thread::sleep_for(chrono::milliseconds(100));
		}
		// 安全关闭
		viewer->close();
		viewer->removeAllPointClouds();
		viewer->removeAllShapes();
	}

	return 0;
}

// 生成单道焊接路径（基于直线偏移）
void generateSinglePassPath(
	const pcl::PointCloud<pcl::PointXYZ>::Ptr& base_path,
	const Eigen::Vector3f& line_direction,
	const Eigen::Vector3f& normal,
	float lateral_offset,
	float vertical_offset,
	float angle_deg,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& output_path)
{
	Eigen::Vector3f lateral_dir = normal.cross(line_direction).normalized();
	float angle_rad = angle_deg * M_PI / 180.0f;

	for (const auto& point : base_path->points) {
		Eigen::Vector3f pt = point.getVector3fMap();

		// 垂直偏移（层高）
		Eigen::Vector3f new_pt = pt + vertical_offset * normal;

		// 横向偏移和角度偏移
		Eigen::Vector3f offset = lateral_offset * lateral_dir;
		if (angle_deg != 0) {
			Eigen::AngleAxisf rotation(angle_rad, line_direction);
			offset = rotation * offset;
		}

		new_pt += offset;
		output_path->points.emplace_back(new_pt.x(), new_pt.y(), new_pt.z());
	}

	output_path->width = output_path->points.size();
	output_path->height = 1;
	output_path->is_dense = true;
}

// 生成姿态文件
void generatePoseFile(
	const pcl::PointCloud<pcl::PointXYZ>::Ptr& path,
	const Eigen::Vector3f& line_direction,
	const Eigen::Vector3f& normal,
	const string& filename)
{
	ofstream ofs(filename);
	if (!ofs.is_open()) {
		cerr << "Error: Could not open pose file " << filename << endl;
		return;
	}
	/*std::cout << "line_direction" << line_direction << std::endl;
	std::cout << "normal" << normal << std::endl;*/
	for (size_t i = 0; i < path->points.size(); ++i) {

		Eigen::Vector3f pt = path->points[i].getVector3fMap();
		Eigen::Vector3f x_axis = line_direction.normalized();
		Eigen::Vector3f z_axis = normal.normalized();

		// 确保正交
		if (abs(z_axis.dot(x_axis)) > 0.001) {
			z_axis = z_axis - z_axis.dot(x_axis) * x_axis;
			z_axis.normalize();
		}

		Eigen::Vector3f y_axis = z_axis.cross(x_axis).normalized();
		z_axis = x_axis.cross(y_axis).normalized(); // 重新正交化

		// 计算欧拉角 (ZYX顺序)
		Eigen::Matrix3f R;
		R.col(0) = x_axis;
		R.col(1) = y_axis;
		R.col(2) = z_axis;
		Eigen::Vector3f euler_deg = R.eulerAngles(2, 1, 0) * 180.0f / M_PI;
		/*ofs << pt.x() << " " << pt.y() << " " << pt.z() + 20 << " "
			<< euler_deg[2] << " " << euler_deg[1] << " " << euler_deg[0] << "\n";*/
		ofs << pt.x() << " " << pt.y() << " " << pt.z() + 9 << " "
			<< -3.599 << " " << 1.585 << " " << 59.575 << "\n";

		//Eigen::Vector3f normalNew(-normal[0], -normal[1], normal[2]);
		//normalNew.normalize();
		//Eigen::Vector3f z_axis = normalNew;
		//Eigen::Vector3f x_axis = Eigen::Vector3f::UnitY().cross(z_axis).normalized();
		//Eigen::Vector3f y_axis = z_axis.cross(x_axis).normalized();

		//Eigen::Matrix3f R;
		//R.col(0) = x_axis;
		//R.col(1) = y_axis;
		//R.col(2) = z_axis;

		//// 从旋转矩阵提取欧拉角 (Z-Y-X顺序)
		//float roll = atan2(R(2, 1), R(2, 2));
		//float pitch = atan2(-R(2, 0), sqrt(R(2, 1) * R(2, 1) + R(2, 2) * R(2, 2)));
		//float yaw = atan2(R(1, 0), R(0, 0));
		//ofs << pt.x() << " " << pt.y() << " " << pt.z()+12 << " "
		//    << roll << " " << pitch << " " << yaw << "\n";
	}
	ofs.close();
	cout << "Saved pose file: " << filename << endl;
}