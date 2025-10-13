#ifndef WELDING_DIRECTION_H
#define WELDING_DIRECTION_H

#include <string>
#include <Eigen/Dense>

/**
 * @brief 计算V型坡口的焊枪末端方向向量
 *
 * @param file_path PCD文件路径
 * @param show_visualization 是否显示可视化窗口(默认为false)
 * @return Eigen::Vector3f 焊枪末端方向向量(归一化)，如果失败返回零向量
 */
Eigen::Vector3f calculateWeldingDirection(const std::string& file_path, bool show_visualization = false, Eigen::Vector3f x_axi = Eigen::Vector3f::Zero());

#endif // WELDING_DIRECTION_H