#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <cstring>
#include <thread>
#include <chrono>
#include <filesystem>
#include "Robot.h"
#include "robot_types.h"

using namespace std;
namespace fs = std::filesystem;

/**
 * @brief 解析位姿数据行，提取位置和旋转信息并存储到DescPose结构体中
 * @param line 输入的字符串行，包含位姿数据
 * @param pose 输出的位姿结构体，存储解析出的位置和旋转信息
 * @param addHeight 高度偏移量，会加到解析出的z坐标上
 * @return 解析成功返回true，解析失败返回false
 */
bool parsePoseLine(const string& line, DescPose& pose, float addHeight) {
	istringstream iss(line);
	float x, y, z, rx_deg, ry_deg, rz_deg;

	if (!(iss >> x >> y >> z >> rx_deg >> ry_deg >> rz_deg)) {
		cerr << "跳过格式错误行: " << line << endl;
		return false;
	}

	memset(&pose, 0, sizeof(DescPose));
	pose.tran.x = x;
	pose.tran.y = y;
	pose.tran.z = z + addHeight;
	pose.rpy.rx = rx_deg;
	pose.rpy.ry = ry_deg;
	pose.rpy.rz = rz_deg;
	return true;
}

// 读取单个位姿文件的首尾位姿
bool readFirstAndLastPose(const string& filepath, DescPose& first_pose, DescPose& last_pose, float addHeight) {
	ifstream ifs(filepath);
	if (!ifs.is_open()) {
		cerr << "无法打开姿态文件: " << filepath << endl;
		return false;
	}

	bool got_first = false;
	string line;
	while (getline(ifs, line)) {
		DescPose current_pose;
		if (!parsePoseLine(line, current_pose, addHeight)) continue;

		if (!got_first) {
			first_pose = current_pose;
			got_first = true;
		}
		last_pose = current_pose;
	}
	ifs.close();
	return got_first;
}


/*
* @brief 保存机械臂当前位姿到文件
*/
int savePose(FRRobot& robot) {

	// std::ofstream是C++标准库中的文件输出流类
	// 创建了一个名为outFile的文件输出流对象
	std::ofstream outFile("KeyPoint\\Pose.txt");
	// 判断是否打开成功
	if (!outFile) {
		std::cerr << "File open error!" << std::endl;
		return 1;
	}

	//FRRobot robot;
	//robot.RPC("192.168.58.2");
	int flag = 0;
	JointPos j_deg;
	DescPose flange, current_pos, tcp, initial_pos;
	int index = 0;
	// 获取当前位姿并输出
	robot.GetActualTCPPose(flag, &tcp);
	printf("desc_pos%d.tran.x = %f;\ndesc_pos%d.tran.y = %f;\ndesc_pos%d.tran.z = %f;\ndesc_pos%d.rpy.rx = %f;\ndesc_pos%d.rpy.ry = %f;\ndesc_pos%d.rpy.rz = %f;\n",
		index, tcp.tran.x,
		index, tcp.tran.y,
		index, tcp.tran.z,
		index, tcp.rpy.rx,
		index, tcp.rpy.ry,
		index, tcp.rpy.rz);
	// 使用字符串流构建完整行
	// std::ostringstream是C++标准库中的输出字符串流类
	// 用于将数据格式化输出到字符串中
	// 后续可以通过.str()方法获取转换后的字符串结果。
	std::ostringstream oss;
	oss << tcp.tran.x << " " << tcp.tran.y << " " << tcp.tran.z << " "
		<< tcp.rpy.rx << " " << tcp.rpy.ry << " " << tcp.rpy.rz;
	// 将oss中的字符串内容写入到文件流outFile中
	outFile << oss.str();
	outFile.close();

	return 1;
}


// 封装的主函数功能
int runWeldingProcess(FRRobot& robot) {
	//FRRobot robot;
	//robot.RPC("192.168.58.2");

	string folder_path = "result/Flat_welding/"; // 目标文件夹
	int tool = 10, user = 0;
	float vel = 1.6, acc = 100.0, ovl = 100.0, blendR = -1.0, beginVel = 60.0;
	ExaxisPos epos;
	memset(&epos, 0, sizeof(ExaxisPos));
	uint8_t search = 0, flag = 0;
	DescPose offset_pos;


	//移动到初始点上方
	string file_pathT = folder_path + "L11" + ".txt";
	DescPose first_poseT, last_poseT;
	if (!readFirstAndLastPose(file_pathT, first_poseT, last_poseT, 0)) {
		cerr << "文件读取失败，跳过: " << file_pathT << endl;
		return 0;
	}
	JointPos j1T{};
	if (robot.GetInverseKin(0, &first_poseT, -1, &j1T) != 0) {
		cerr << "第一个位姿逆解失败，跳过该文件。\n";
		return 0;
	}
	DescPose addPoseT;
	addPoseT.tran.x = first_poseT.tran.x;
	addPoseT.tran.y = first_poseT.tran.y;
	addPoseT.tran.z = first_poseT.tran.z + 30;
	addPoseT.rpy.rx = first_poseT.rpy.rx;
	addPoseT.rpy.ry = first_poseT.rpy.ry;
	addPoseT.rpy.rz = first_poseT.rpy.rz;
	JointPos addj1T{};
	if (robot.GetInverseKin(0, &addPoseT, -1, &addj1T) != 0) {
		cerr << "位姿逆解失败，请重新试。\n";
	}
	int err3 = robot.MoveL(&addj1T, &addPoseT, tool, user, beginVel, acc, ovl, blendR, &epos, search, flag, &offset_pos);
	if (err3 != 0) {
		cerr << "抬高位姿移动失败: 错误码 " << err3 << endl;
	}

	// 焊接参数
	double startCurrent = 0;
	double startVoltage = 0;
	double startTime = 0;
	double weldCurrent = 0;
	double weldVoltage = 0;
	double endCurrent = 0;
	double endVoltage = 0;
	double endTime = 0;
	robot.WeldingGetProcessParam(1, startCurrent, startVoltage, startTime, weldCurrent, weldVoltage, endCurrent, endVoltage, endTime);
	cout << "the Num 1 process param is " << startCurrent << " " << startVoltage << " " << startTime << " " << weldCurrent << " " << weldVoltage << " " << endCurrent << " " << endVoltage << " " << endTime << endl;

	int file_index = 0;
	int num = 0;
	string name = "";
	while (true) {
		if (file_index == 0) {
			int ne = robot.WeaveSetPara(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5);
			name = "L11";
		}
		else if (file_index == 1) {
			int ne = robot.WeaveSetPara(0, 0, 1.0, 0, 3.5, 0, 0, 0, 150.0, 150.0, 0, 0, 5);
			vel = 0.8;
			name = "L2" + to_string(file_index);
		}
		else if (file_index == 2) {
			vel = 0.8;
			int ne = robot.WeaveSetPara(0, 0, 0.9, 0, 5.0, 0, 0, 0, 150.0, 10.0, 0, 0, 5);
			name = "L3" + to_string(file_index - 1);
		}
		else if (file_index == 3) {
			vel = 0.7;
			int ne = robot.WeaveSetPara(0, 0, 0.7, 0, 7, 0, 0, 0, 200.0, 200.0, 0, 0, 5);
			name = "L4" + to_string(file_index - 2);
		}
		else if (file_index == 4) {
			int ne = robot.WeaveSetPara(0, 0, 0.4, 0, 10.0, 0, 0, 0, 500.0, 250.0, 0, 0, 5);
			vel = 0.5;
			name = "L5" + to_string(file_index - 3);
		}
		string file_path = folder_path + name + ".txt";
		std::cout << "folder_path---" << file_path << std::endl;
		if (!fs::exists(file_path) || file_index > 4) {
			cout << "所有姿态文件读取完毕，结束运行。\n";
			break;
		}
		cout << "\n>>> 读取姿态文件: " << file_path << endl;
		float addHeight = (file_index == 0 ? 0.0 : 10);
		DescPose first_pose, last_pose;
		if (!readFirstAndLastPose(file_path, first_pose, last_pose, 0)) {
			cerr << "文件读取失败，跳过: " << file_path << endl;
			file_index++;
			continue;
		}

		JointPos j1{}, j2{};
		if (robot.GetInverseKin(0, &first_pose, -1, &j1) != 0) {
			cerr << "第一个位姿逆解失败，跳过该文件。\n";
			file_index++;
			continue;
		}
		if (robot.GetInverseKin(0, &last_pose, -1, &j2) != 0) {
			cerr << "最后一个位姿逆解失败，跳过该文件。\n";
			file_index++;
			continue;
		}

		int err1 = robot.MoveL(&j1, &first_pose, tool, user, beginVel, acc, ovl, blendR, &epos, search, flag, &offset_pos);
		if (err1 != 0) {
			cerr << "起始位姿移动失败: 错误码 " << err1 << endl;
			file_index++;
			continue;
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(500));
		if (num > 0) {
			robot.WeaveStart(0);
		}

		cout << "********开始启弧********.\n";
		int err2 = robot.MoveL(&j2, &last_pose, tool, user, vel, acc, ovl, blendR, &epos, search, flag, &offset_pos);
		if (err2 != 0) {
			cerr << "终止位姿移动失败: 错误码 " << err2 << endl;
			file_index++;
			continue;
		}
		if (num > 0) {
			robot.WeaveEnd(0);
		}
		num++;
		std::this_thread::sleep_for(std::chrono::milliseconds(300));
		cout << "********停止启弧********.\n";
		std::this_thread::sleep_for(std::chrono::milliseconds(1500));

		// 抬高100mm
		DescPose addPose;
		addPose.tran.x = last_pose.tran.x;
		addPose.tran.y = last_pose.tran.y;
		addPose.tran.z = last_pose.tran.z + 100;
		addPose.rpy.rx = last_pose.rpy.rx;
		addPose.rpy.ry = last_pose.rpy.ry;
		addPose.rpy.rz = last_pose.rpy.rz;
		JointPos addj1{};
		if (robot.GetInverseKin(0, &addPose, -1, &addj1) != 0) {
			cerr << "位姿逆解失败，请重新试。\n";
			continue;
		}
		int err3 = robot.MoveL(&addj1, &addPose, tool, user, beginVel, acc, ovl, blendR, &epos, search, flag, &offset_pos);
		if (err3 != 0) {
			cerr << "抬高位姿移动失败: 错误码 " << err2 << endl;
			file_index++;
			continue;
		}
		cout << "文件 " << file_index << ".txt 处理完成。\n";
		file_index++;
	}

	return 0;
}
