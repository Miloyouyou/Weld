// welding_path_generator.h
#ifndef WELDING_PATH_GENERATOR_H
#define WELDING_PATH_GENERATOR_H

#include <string>
#include "Robot.h"
#include "robot_types.h"

/**
 * @brief 生成V坡口多层多道焊接路径
 *
 * @param keypoints_file 关键点文件路径
 * @param output_dir 输出目录
 * @param distanceThreshold  设置阈值为0.85单位,小于这个为内点
 * @param visualize 是否可视化结果
 * @return int 成功返回0，失败返回-1
 */
int VpogenerateWeldingPaths(
    const std::string& input_file,
    const std::string& keypoints_file,
    const std::string& output_dir = "result/Flat_welding/",
    float distanceThreshold = 0.85,
    bool visualize = true);
/*
* @brief 保存机械臂当前位姿到文件
*/
int savePose(FRRobot& robot);
/*
* * @brief 执行焊接路径生成过程
*/
int runWeldingProcess(FRRobot& robot);
#endif // WELDING_PATH_GENERATOR_H#pragma once
