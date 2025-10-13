#include <iostream>
#include "keyPoint.h"
#include "MultipleTrajectories.h"
#include "CalNormolVector.h"
#include "Robot.h"
#include "robot_types.h"
#include "VSensor.h"
#include <string>
#include <memory>
#include <direct.h>
#include <io.h>
#include <fstream>
#include <thread>
#include <chrono>
#include <cstdlib>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/passthrough.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common_headers.h>
#include <boost/thread/thread.hpp>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/time.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <Eigen/Dense>

using namespace std;
using namespace pcl;

typedef PointXYZRGB PointT;
typedef PointCloud<PointT> PointCloudT;

#pragma comment(lib, "VSensorSDK.lib")

using namespace std;
using namespace VSENSOR;
using Clock = std::chrono::high_resolution_clock;

//================= 读取txt点云，只保留label=7 =================
PointCloudT::Ptr readTxtLabel7(const std::string& file) {
    PointCloudT::Ptr cloud(new PointCloudT);
    std::ifstream infile(file);
    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        float x, y, z, nx, ny, nz, label;
        if (!(iss >> x >> y >> z >> nx >> ny >> nz >> label)) continue;
        if (std::fabs(label - 7.0f) > 1e-3) continue; // 容忍小数误差
        PointT pt; pt.x = x; pt.y = y; pt.z = z; pt.r = 255; pt.g = 255; pt.b = 255;
        cloud->push_back(pt);
    }
    return cloud;
}

//================= 读取Pose.txt的第一行，返回平移和旋转 =================
bool readFirstPose(const std::string& file, float& tx, float& ty, float& tz,
    float& rx, float& ry, float& rz) {
    std::ifstream infile(file);
    std::string line;
    if (!std::getline(infile, line)) return false;
    std::istringstream iss(line);
    std::vector<float> vals;
    float v;
    while (iss >> v) vals.push_back(v);
    if (vals.size() < 6) return false;
    tx = vals[0]; ty = vals[1]; tz = vals[2];
    rx = vals[3]; ry = vals[4]; rz = vals[5];
    return true;
}

//================= 生成变换矩阵 =================
Eigen::Affine3f getTransformationMatrix(float tx, float ty, float tz,
    float rx, float ry, float rz) {
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.rotate(Eigen::AngleAxisf(pcl::deg2rad(rz), Eigen::Vector3f::UnitZ()));
    transform.rotate(Eigen::AngleAxisf(pcl::deg2rad(ry), Eigen::Vector3f::UnitY()));
    transform.rotate(Eigen::AngleAxisf(pcl::deg2rad(rx), Eigen::Vector3f::UnitX()));
    transform.translation() << tx, ty, tz;
    return transform;
}

//================= 工具函数 =================
bool EnsureFolder(const string& folderName) {
    if (_access(folderName.c_str(), 0) != 0) {
        if (_mkdir(folderName.c_str()) != 0) {
            cerr << "Create folder failed: " << folderName << endl;
            return false;
        }
    }
    return true;
}

//================= 拍摄并保存点云 =================
bool CapturePointCloud(const string& folderName, string& pcdFile, double& elapsed) {
    auto t1 = Clock::now();

    unique_ptr<VSensor> pVSensor = make_unique<VSensor>();
    VSensorCameraInfo cameraList[10];
    int num = 0;
    pVSensor->GetDeviceList(cameraList, &num);
    if (num == 0) {
        cerr << "No devices found." << endl;
        return false;
    }

    int status = pVSensor->DeviceConnect(0);
    if (status != CAMERA_STATUS_SUCCESS) {
        cerr << "Connect device failed, status = " << status << endl;
        return false;
    }

    pVSensor->DeviceParameterInit();
    pVSensor->SetUserSettingMode(emVSensorUserSettingMode::MODE_EXPERT);
    pVSensor->SetCaptureMode(emVSensorCaptureMode::MODE_CAPTURE);
    pVSensor->SetDownsampling(true);

    unique_ptr<VSensorResult> pResult = make_unique<VSensorResult>();
    status = pVSensor->SingleRestruction(pResult.get(), emVSensorCaptureOutputMode::OUTPUT_MODE_ALL);
    if (status != CAMERA_STATUS_SUCCESS) {
        cerr << "Single restruction failed, status = " << status << endl;
        return false;
    }

    pcdFile = folderName + "\\test.pcd";
    status = pVSensor->Save3DCloud(pcdFile, pResult.get(), emVSensorPointType::POINT_TYPR_DISORDER);
    if (status != CAMERA_STATUS_SUCCESS) {
        cerr << "Save 3D points failed, status = " << status << endl;
        return false;
    }
    pVSensor->DeviceUnInit();

    auto t2 = Clock::now();
    elapsed = std::chrono::duration<double>(t2 - t1).count();
    return true;
}

//================= 读取点云并计算法线 =================
bool ComputeNormals(const string& pcdFile,
    pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    pcl::PointCloud<pcl::Normal>::Ptr& normals,
    double& elapsed) {
    auto t1 = Clock::now();

    cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcdFile, *cloud) == -1) {
        PCL_ERROR("Could not read file %s\n", pcdFile.c_str());
        return false;
    }
    cout << "加载点数: " << cloud->size() << endl;

    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
    ne.setNumberOfThreads(std::thread::hardware_concurrency());
    ne.setInputCloud(cloud);

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    ne.setSearchMethod(tree);
    ne.setKSearch(20);

    normals.reset(new pcl::PointCloud<pcl::Normal>);
    ne.compute(*normals);

    auto t2 = Clock::now();
    elapsed = std::chrono::duration<double>(t2 - t1).count();
    return true;
}

//================= 保存结果为txt =================
bool SaveCloudWithNormals(const string& txtFile,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    const pcl::PointCloud<pcl::Normal>::Ptr& normals,
    double& elapsed) {
    auto t1 = Clock::now();

    ofstream ofs(txtFile);
    if (!ofs.is_open()) {
        PCL_ERROR("Could not open output file %s\n", txtFile.c_str());
        return false;
    }
    for (size_t i = 0; i < cloud->size(); ++i) {
        const auto& p = cloud->points[i];
        const auto& n = normals->points[i];
        ofs << p.x << " " << p.y << " " << p.z << " "
            << n.normal_x << " " << n.normal_y << " " << n.normal_z << "\n";
    }
    ofs.close();

    auto t2 = Clock::now();
    elapsed = std::chrono::duration<double>(t2 - t1).count();
    return true;
}

//================= 调用 bat 脚本 =================
void RunBatFile(const string& batFile) {
    if (_access(batFile.c_str(), 0) == 0) {
        cout << "正在执行: " << batFile << endl;
        int ret = system(batFile.c_str());
        if (ret != 0) {
            cerr << "运行 bat 文件失败，返回码 = " << ret << endl;
        }
    }
    else {
        cerr << "未找到 bat 文件: " << batFile << endl;
    }
}

// 保存机械臂当前位姿到文件
int savePoseMem(FRRobot& robot, const std::string& poseFilePath) {
    std::ofstream outFile(poseFilePath);
    if (!outFile) {
        std::cerr << "File open error!" << std::endl;
        return 1;
    }

    int flag = 0;
    DescPose tcp;
    robot.GetActualTCPPose(flag, &tcp);

    printf("desc_pos.tran.x = %f;\ndesc_pos.tran.y = %f;\ndesc_pos.tran.z = %f;\ndesc_pos.rpy.rx = %f;\ndesc_pos.rpy.ry = %f;\ndesc_pos.rpy.rz = %f;\n",
        tcp.tran.x, tcp.tran.y, tcp.tran.z, tcp.rpy.rx, tcp.rpy.ry, tcp.rpy.rz);

    std::ostringstream oss;
    oss << tcp.tran.x << " " << tcp.tran.y << " " << tcp.tran.z << " "
        << tcp.rpy.rx << " " << tcp.rpy.ry << " " << tcp.rpy.rz;

    outFile << oss.str();
    outFile.close();

    return 1;
}

int  type = 1;                  // 工件类型 
bool isMove = 0;                // 是否移动机械臂
bool isCamera = 1;              // 是否使用相机
//bool isNeedChangePlace = 1;     // 是否需要转换位置
bool isGetWeldPose = 1;         // 是否获取焊接路径
bool isMoveRobot = 1;           // 是否移动机械臂到焊接位置

int main()
{
	FRRobot robot;
	robot.RPC("192.168.58.2"); // 与机器人建立通信
    
    if (isMove)
    {
        DescPose Pose_start;
        Pose_start.tran.x = 37.507;
        Pose_start.tran.y = 785.608;
        Pose_start.tran.z = 277.78;
        Pose_start.rpy.rx = 4.604;
        Pose_start.rpy.ry = -8.457;
        Pose_start.rpy.rz = 53.186;

        JointPos Joint_start{};
        if (robot.GetInverseKin(0, &Pose_start, -1, &Joint_start) != 0) {
            cerr << "安全起始位姿逆解失败" << endl;
        }

        int tool = 10, user = 0;
        float vel = 20.0, acc = 100.0, ovl = 100.0, blendR = -1.0, beginVel = 60.0;
        ExaxisPos epos;
        memset(&epos, 0, sizeof(ExaxisPos));
        uint8_t search = 0, flag = 0;
        DescPose offset_pos;

        int err1 = robot.MoveL(&Joint_start, &Pose_start, tool, user, beginVel,
            acc, ovl, blendR, &epos, search, flag, &offset_pos);
        if (err1 != 0) {
            cerr << "起始位姿移动失败: 错误码 " << err1 << endl;
        }

    }
    if (isCamera)
    {
        string folderName = "D:\\PointROINet\\test\\00000004";
        if (!EnsureFolder(folderName)) return 1;

        string pcdFile;
        double capture_time = 0.0;
        if (!CapturePointCloud(folderName, pcdFile, capture_time)) return 1;
        cout << "点云采集+保存耗时: " << capture_time << " 秒" << endl;

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
        pcl::PointCloud<pcl::Normal>::Ptr normals;
        double normal_time = 0.0;
        if (!ComputeNormals(pcdFile, cloud, normals, normal_time)) return 1;
        cout << "法线计算耗时: " << normal_time << " 秒" << endl;

        string txtFile = folderName + "\\cloud_with_normals.txt";
        double save_time = 0.0;
        if (!SaveCloudWithNormals(txtFile, cloud, normals, save_time)) return 1;
        cout << "结果保存耗时: " << save_time << " 秒" << endl;

        cout << "总耗时: " << (capture_time + normal_time + save_time) << " 秒" << endl;
        cout << "流程执行成功！" << endl;

        //RunBatFile(folderName + "\\run_script.bat");

    }

    if (isGetWeldPose)
    {
        // Eigen 声明一个4x4矩阵
        Eigen::Matrix4f eyeToHand;

        eyeToHand <<
            0.615, -0.792, 0.024, -161.940,
            -0.790, -0.614, -0.055, -65.229,
            0.055, 0.007, -1.002, 224.334,
            0.000, 0.000, 0.000, 1.000;

        std::string pose_file = "KeyPoint\\Pose.txt";
        std::string out_pcd_file = "KeyPoint\\1.pcd";
        std::string input_file = "D:/PointROINet/weld/PonintNet_Weld_identification/PonintNet_0.txt";

        //savePoseMem(robot, pose_file);
        savePose(robot);

        //  读点云
        auto cloud = readTxtLabel7(input_file);
        if (cloud->empty()) {
            std::cerr << "点云为空或没有label=7的数据！" << std::endl;
            return -1;
        }

        //  读位姿
        float tx, ty, tz, rx, ry, rz;
        if (!readFirstPose(pose_file, tx, ty, tz, rx, ry, rz)) {
            std::cerr << "位姿文件读取失败或格式错误！" << std::endl;
            return -2;
        }

        //  构造变换
        Eigen::Affine3f HandToBase = Eigen::Affine3f::Identity();
        HandToBase.rotate(Eigen::AngleAxisf(pcl::deg2rad(rz), Eigen::Vector3f::UnitZ()));
        HandToBase.rotate(Eigen::AngleAxisf(pcl::deg2rad(ry), Eigen::Vector3f::UnitY()));
        HandToBase.rotate(Eigen::AngleAxisf(pcl::deg2rad(rx), Eigen::Vector3f::UnitX()));
        HandToBase.translation() << tx, ty, tz;
        Eigen::Affine3f trans = HandToBase * Eigen::Affine3f(eyeToHand);

        //  变换
        PointCloudT::Ptr cloudTrans(new PointCloudT);
        pcl::transformPointCloud(*cloud, *cloudTrans, trans);

        //  保存
        if (pcl::io::savePCDFileBinary(out_pcd_file, *cloudTrans) == -1) {
            std::cerr << "保存点云失败: " << out_pcd_file << std::endl;
            return -3;
        }
        std::cout << "已保存转换后点云: " << out_pcd_file << std::endl;

        // 输入点云文件路径
        std::string out_file = "KeyPoint/V_keypointsHarris.pcd";
        bool succes = processVGrooveAndExtractLine(out_pcd_file, out_file);
        std::string output_dir = "result/Flat_welding/";

        //多层多道轨迹
        int result = VpogenerateWeldingPaths(
            out_pcd_file,
            out_file,      //
            output_dir,    //多层多通道路径文件目录
            0.65,          //粘合直线，设置阈值为0.85单位,小于这个为内点
            true);
        // 4. 输出结果
        if (result == 0) {
            std::cout << "Welding path generation completed successfully!" << std::endl;
        }
        else {
            std::cerr << "Welding path generation failed!" << std::endl;
        }
    }

	if (isMoveRobot) {
		runWeldingProcess(robot); // 执行焊接过程
	}

	return 0;
}

