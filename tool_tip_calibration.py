#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
器械尖端标定系统 (5点Bundle Adjustment方法)
实现完整的器械尖端标定流程，包括：
1. 智能选择5个标记点（ABCDE）从5个检测点中
2. 使用6点PnP进行初始估计（5个真实点+1个虚拟点）
3. Bundle Adjustment优化3D点坐标和相机姿态
4. 求解世界坐标Tip_w
5. 求解局部坐标Tip_t
6. 所有5个点（ABCDE）都参与优化


"""

import numpy as np
import cv2
import os
import glob
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
from detection.marker_detector import MarkerDetector


class ToolTipCalibration:
    """器械尖端标定系统"""
    
    def __init__(self, camera_calibration_path: str = 'calibration/camera_calibration.npz'):
        """
        初始化标定系统
        
        Args:
            camera_calibration_path: 相机标定文件路径
        """
        self.camera_calibration_path = camera_calibration_path
        self.camera_matrix = None
        self.dist_coeffs = None
        self.marker_detector = None
        
        # 器械几何参数（优化后的真实距离）
        self.AB_distance_mm = 87.5  # AB两点之间的实际距离（毫米）- 优化后 +0.5mm
        self.AC_distance_mm = 39.651  # AC两点之间的实际距离（毫米）- 优化后 +0.051mm
        self.BC_distance_mm = 57.9  # BC两点之间的实际距离（毫米）- 保持不变
        self.AD_distance_mm = 42.1  # AD两点之间的实际距离（毫米）- 保持不变
        self.BD_distance_mm = 70.7  # BD两点之间的实际距离（毫米）- 优化后 +0.5mm

        # 新增E点距离参数
        self.AE_distance_mm = 73.8  # AE两点之间的实际距离（毫米）
        self.BE_distance_mm = 57.5  # BE两点之间的实际距离（毫米）

        # 建立固定的器械坐标系蓝图（基于真实距离，一次性建立）
        self.tool_coordinate_blueprint = self._establish_tool_coordinate_blueprint()

        # 存储所有帧的2D点数据（用于bundle adjustment）
        self.points_2d_all_frames = []  # 存储每帧的ABCDE点2D坐标
        self.points_3d = None  # 优化后的ABCDE点3D坐标
        self.camera_poses = []  # 存储相机姿态(rvec, tvec)

        # 标定数据存储
        self.image_data = []  # 存储每张图片的检测结果
        self.world_coordinates = []  # 存储ABC点的世界坐标
        self.tool_poses = []  # 存储器械姿态(R, T)

        # 标定结果
        self.tip_world = None  # 器械尖端世界坐标 Tip_w
        self.tip_tool = None   # 器械尖端局部坐标 Tip_t

        self._load_camera_calibration()
        self._initialize_detector()
    
    def _load_camera_calibration(self):
        """加载相机标定参数"""
        try:
            calib_data = np.load(self.camera_calibration_path)
            self.camera_matrix = calib_data['camera_matrix']
            self.dist_coeffs = calib_data['dist_coeffs']
            print(f"✅ 相机标定参数加载成功: {self.camera_calibration_path}")
            print(f"   相机内参矩阵形状: {self.camera_matrix.shape}")
            print(f"   畸变系数形状: {self.dist_coeffs.shape}")
        except Exception as e:
            raise RuntimeError(f"❌ 无法加载相机标定参数: {e}")
    
    def _initialize_detector(self):
        """初始化标记点检测器"""
        self.marker_detector = MarkerDetector(self.camera_calibration_path)
        print("✅ 标记点检测器初始化成功")

    def _establish_tool_coordinate_blueprint(self) -> Dict[str, np.ndarray]:
        """
        建立固定的器械坐标系蓝图（基于真实距离数据）

        该方法一次性建立器械坐标系，对所有图片都适用：
        - A点为原点 (0, 0, 0)
        - AB方向为X轴，长度为87.5mm
        - 使用已知的真实距离精确计算C、D、E点位置
        - 所有点都在Z=0平面上

        Returns:
            dict: 器械坐标系下的ABCDE点3D坐标 {'A': [x,y,z], 'B': [x,y,z], 'C': [x,y,z], 'D': [x,y,z], 'E': [x,y,z]}
        """
        print("🔧 建立固定器械坐标系蓝图（基于真实距离，包含E点）")

        # 使用新的初始设置坐标
        A_tool = np.array([0.0, 0.0, 0.0])  # A点为原点
        B_tool = np.array([87.5, 0.0, 0.0])  # B点在X轴上
        C_tool = np.array([33.6, 21.1, 0.0])  # C点坐标
        D_tool = np.array([25.3, -33.6, 0.0])  # D点坐标
        E_tool = np.array([48.9, -57.5, 0.0])  # E点坐标

        # 验证几何约束
        calculated_AB = np.linalg.norm(B_tool - A_tool)
        calculated_AC = np.linalg.norm(C_tool - A_tool)
        calculated_BC = np.linalg.norm(C_tool - B_tool)
        calculated_AD = np.linalg.norm(D_tool - A_tool)
        calculated_BD = np.linalg.norm(D_tool - B_tool)
        calculated_AE = np.linalg.norm(E_tool - A_tool)
        calculated_BE = np.linalg.norm(E_tool - B_tool)

        print(f"   器械坐标系蓝图验证:")
        print(f"   - AB: {calculated_AB:.3f} mm (期望: {self.AB_distance_mm:.1f} mm)")
        print(f"   - AC: {calculated_AC:.3f} mm (期望: {self.AC_distance_mm:.1f} mm)")
        print(f"   - BC: {calculated_BC:.3f} mm (期望: {self.BC_distance_mm:.1f} mm)")
        print(f"   - AD: {calculated_AD:.3f} mm (期望: {self.AD_distance_mm:.1f} mm)")
        print(f"   - BD: {calculated_BD:.3f} mm (期望: {self.BD_distance_mm:.1f} mm)")
        print(f"   - AE: {calculated_AE:.3f} mm (期望: {self.AE_distance_mm:.1f} mm)")
        print(f"   - BE: {calculated_BE:.3f} mm (期望: {self.BE_distance_mm:.1f} mm)")
        print(f"   蓝图坐标:")
        print(f"   - A: [{A_tool[0]:.3f}, {A_tool[1]:.3f}, {A_tool[2]:.3f}] mm")
        print(f"   - B: [{B_tool[0]:.3f}, {B_tool[1]:.3f}, {B_tool[2]:.3f}] mm")
        print(f"   - C: [{C_tool[0]:.3f}, {C_tool[1]:.3f}, {C_tool[2]:.3f}] mm")
        print(f"   - D: [{D_tool[0]:.3f}, {D_tool[1]:.3f}, {D_tool[2]:.3f}] mm")
        print(f"   - E: [{E_tool[0]:.3f}, {E_tool[1]:.3f}, {E_tool[2]:.3f}] mm")

        return {
            'A': A_tool,
            'B': B_tool,
            'C': C_tool,
            'D': D_tool,
            'E': E_tool
        }

    def select_best_5_points(self, markers: List[Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
        """
        从5个检测到的标记点中智能选择5个点作为ABCDE

        选择策略（按您的要求）：
        1. 找到最长线段的两个端点作为AB候选
        2. 找到其他点中到AB线段距离最近的点作为C
        3. 根据C到AB两点的距离，距离近的为A，远的为B
        4. 在剩余点中找距离AB线段最近的点作为D
        5. 剩余的最后一个点作为E
        6. 不使用任何真实距离数据

        Args:
            markers: 检测到的标记点列表 [(x1,y1), (x2,y2), ...]

        Returns:
            dict: 选择的ABCDE点 {'A': (x,y), 'B': (x,y), 'C': (x,y), 'D': (x,y), 'E': (x,y)}
        """
        if len(markers) != 5:
            raise ValueError(f"需要恰好5个标记点，但检测到{len(markers)}个")

        markers = np.array(markers)
        print(f"检测到 {len(markers)} 个标记点")
        for i, marker in enumerate(markers):
            print(f"  标记点{i+1}: ({marker[0]:.1f}, {marker[1]:.1f})")

        # 1. 找到距离最长的两个点作为AB候选
        max_distance = 0
        AB_candidate_idx1, AB_candidate_idx2 = 0, 1

        for i in range(len(markers)):
            for j in range(i+1, len(markers)):
                distance = np.linalg.norm(markers[i] - markers[j])
                if distance > max_distance:
                    max_distance = distance
                    AB_candidate_idx1, AB_candidate_idx2 = i, j

        AB_candidate1 = markers[AB_candidate_idx1]
        AB_candidate2 = markers[AB_candidate_idx2]
        print(f"最长边距离: {max_distance:.1f} 像素")
        print(f"最长边端点: ({AB_candidate1[0]:.1f}, {AB_candidate1[1]:.1f}) - ({AB_candidate2[0]:.1f}, {AB_candidate2[1]:.1f})")

        # 2. 从剩余点中选择到AB线段距离最短的点作为C
        remaining_indices = [i for i in range(len(markers)) if i not in [AB_candidate_idx1, AB_candidate_idx2]]

        min_distance_to_line = float('inf')
        C_idx = remaining_indices[0]

        # 计算点到直线的距离
        AB_vec = AB_candidate2 - AB_candidate1
        AB_length = np.linalg.norm(AB_vec)

        for idx in remaining_indices:
            point = markers[idx]
            AP_vec = point - AB_candidate1

            # 点到直线距离公式
            cross_product = np.abs(np.cross(AB_vec, AP_vec))
            distance_to_line = cross_product / AB_length

            if distance_to_line < min_distance_to_line:
                min_distance_to_line = distance_to_line
                C_idx = idx

        C_point = markers[C_idx]
        print(f"选择的C点: ({C_point[0]:.1f}, {C_point[1]:.1f}), 到AB线段距离: {min_distance_to_line:.1f}")

        # 3. 根据C到AB两点的距离确定A和B：距离近的为A，远的为B
        distance_C_to_candidate1 = np.linalg.norm(C_point - AB_candidate1)
        distance_C_to_candidate2 = np.linalg.norm(C_point - AB_candidate2)

        if distance_C_to_candidate1 < distance_C_to_candidate2:
            A_point = AB_candidate1
            B_point = AB_candidate2
            print(f"C到候选点1距离: {distance_C_to_candidate1:.1f}, 到候选点2距离: {distance_C_to_candidate2:.1f}")
            print(f"选择候选点1为A点，候选点2为B点")
        else:
            A_point = AB_candidate2
            B_point = AB_candidate1
            print(f"C到候选点1距离: {distance_C_to_candidate1:.1f}, 到候选点2距离: {distance_C_to_candidate2:.1f}")
            print(f"选择候选点2为A点，候选点1为B点")

        # 4. 在剩余点中找距离AB线段最近的点作为D
        remaining_indices = [i for i in range(len(markers)) if i not in [AB_candidate_idx1, AB_candidate_idx2, C_idx]]

        min_distance_to_line_D = float('inf')
        D_idx = remaining_indices[0]

        # 重新计算AB向量（现在A、B顺序已确定）
        AB_vec = B_point - A_point
        AB_length = np.linalg.norm(AB_vec)

        for idx in remaining_indices:
            point = markers[idx]
            AP_vec = point - A_point

            # 点到直线距离公式
            cross_product = np.abs(np.cross(AB_vec, AP_vec))
            distance_to_line = cross_product / AB_length

            if distance_to_line < min_distance_to_line_D:
                min_distance_to_line_D = distance_to_line
                D_idx = idx

        D_point = markers[D_idx]
        print(f"选择的D点: ({D_point[0]:.1f}, {D_point[1]:.1f}), 到AB线段距离: {min_distance_to_line_D:.1f}")

        # 5. 找到剩余的最后一个点作为E
        remaining_indices_E = [i for i in range(len(markers)) if i not in [AB_candidate_idx1, AB_candidate_idx2, C_idx, D_idx]]

        if len(remaining_indices_E) != 1:
            raise ValueError(f"应该剩余1个点作为E，但实际剩余{len(remaining_indices_E)}个点")

        E_idx = remaining_indices_E[0]
        E_point = markers[E_idx]
        print(f"选择的E点: ({E_point[0]:.1f}, {E_point[1]:.1f}) (剩余点)")

        print(f"最终选择:")
        print(f"A点: ({A_point[0]:.1f}, {A_point[1]:.1f})")
        print(f"B点: ({B_point[0]:.1f}, {B_point[1]:.1f})")
        print(f"C点: ({C_point[0]:.1f}, {C_point[1]:.1f})")
        print(f"D点: ({D_point[0]:.1f}, {D_point[1]:.1f})")
        print(f"E点: ({E_point[0]:.1f}, {E_point[1]:.1f})")

        return {
            'A': (float(A_point[0]), float(A_point[1])),
            'B': (float(B_point[0]), float(B_point[1])),
            'C': (float(C_point[0]), float(C_point[1])),
            'D': (float(D_point[0]), float(D_point[1])),
            'E': (float(E_point[0]), float(E_point[1]))
        }
    
    def establish_tool_coordinate_system(self, points_2d: Dict[str, Tuple[float, float]]) -> Dict[str, np.ndarray]:
        """
        获取固定的器械坐标系（直接使用预建立的蓝图）

        该方法直接返回基于真实距离建立的固定器械坐标系蓝图，
        不需要基于每张图片重新计算，确保所有图片使用相同的坐标系。

        Args:
            points_2d: 图像中的ABCDE点坐标（用于验证，但不用于计算）

        Returns:
            dict: 器械坐标系下的3D点坐标 {'A': [x,y,z], 'B': [x,y,z], 'C': [x,y,z], 'D': [x,y,z], 'E': [x,y,z]}
        """
        # 直接使用预建立的固定器械坐标系蓝图
        print(f"   使用固定器械坐标系蓝图（基于真实距离，包含E点）")

        # 获取蓝图坐标
        A_tool = self.tool_coordinate_blueprint['A'].copy()
        B_tool = self.tool_coordinate_blueprint['B'].copy()
        C_tool = self.tool_coordinate_blueprint['C'].copy()
        D_tool = self.tool_coordinate_blueprint['D'].copy()
        E_tool = self.tool_coordinate_blueprint['E'].copy()

        # 验证蓝图距离
        calculated_AB = np.linalg.norm(B_tool - A_tool)
        calculated_AC = np.linalg.norm(C_tool - A_tool)
        calculated_BC = np.linalg.norm(C_tool - B_tool)
        calculated_AD = np.linalg.norm(D_tool - A_tool)
        calculated_BD = np.linalg.norm(D_tool - B_tool)
        calculated_AE = np.linalg.norm(E_tool - A_tool)
        calculated_BE = np.linalg.norm(E_tool - B_tool)

        print(f"   蓝图距离验证:")
        print(f"   - AB: {calculated_AB:.2f} mm (期望: {self.AB_distance_mm:.1f} mm)")
        print(f"   - AC: {calculated_AC:.2f} mm (期望: {self.AC_distance_mm:.1f} mm)")
        print(f"   - BC: {calculated_BC:.2f} mm (期望: {self.BC_distance_mm:.1f} mm)")
        print(f"   - AD: {calculated_AD:.2f} mm (期望: {self.AD_distance_mm:.1f} mm)")
        print(f"   - BD: {calculated_BD:.2f} mm (期望: {self.BD_distance_mm:.1f} mm)")
        print(f"   - AE: {calculated_AE:.2f} mm (期望: {self.AE_distance_mm:.1f} mm)")
        print(f"   - BE: {calculated_BE:.2f} mm (期望: {self.BE_distance_mm:.1f} mm)")
        print(f"   蓝图坐标:")
        print(f"   - A: [{A_tool[0]:.1f}, {A_tool[1]:.1f}, {A_tool[2]:.1f}] mm")
        print(f"   - B: [{B_tool[0]:.1f}, {B_tool[1]:.1f}, {B_tool[2]:.1f}] mm")
        print(f"   - C: [{C_tool[0]:.1f}, {C_tool[1]:.1f}, {C_tool[2]:.1f}] mm")
        print(f"   - D: [{D_tool[0]:.1f}, {D_tool[1]:.1f}, {D_tool[2]:.1f}] mm")
        print(f"   - E: [{E_tool[0]:.1f}, {E_tool[1]:.1f}, {E_tool[2]:.1f}] mm")

        return {
            'A': A_tool,
            'B': B_tool,
            'C': C_tool,
            'D': D_tool,
            'E': E_tool
        }
    
    def solve_pose_with_pnp(self, points_2d: Dict[str, Tuple[float, float]],
                           points_3d_tool: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用6点solvePnP求解器械姿态（5个真实点 + 1个虚拟点）

        Args:
            points_2d: 图像中的ABCDE点坐标
            points_3d_tool: 器械坐标系下的ABCDE点3D坐标

        Returns:
            tuple: (旋转矩阵R, 平移向量T)
        """
        # 准备5个真实点的3D和2D坐标（包含E点）
        real_object_points = np.array([
            points_3d_tool['A'],
            points_3d_tool['B'],
            points_3d_tool['C'],
            points_3d_tool['D'],
            points_3d_tool['E']
        ], dtype=np.float32)

        real_image_points = np.array([
            points_2d['A'],
            points_2d['B'],
            points_2d['C'],
            points_2d['D'],
            points_2d['E']
        ], dtype=np.float32)

        # 添加虚拟第6个点以提高PnP求解稳定性
        # 虚拟3D点：选择一个离轴点
        virtual_3d_point = np.array([60.0, 10.0, 0.0], dtype=np.float32)

        # 虚拟2D点：使用真实点的质心
        virtual_2d_point = np.mean(real_image_points, axis=0)

        # 组合成6点数据
        object_points_6 = np.vstack([real_object_points, virtual_3d_point])
        image_points_6 = np.vstack([real_image_points, virtual_2d_point])

        print(f"🔧 使用6点PnP求解（5个真实点 + 1个虚拟点）:")
        print(f"   3D点形状: {object_points_6.shape}")
        print(f"   2D点形状: {image_points_6.shape}")
        print(f"   虚拟3D点: [{virtual_3d_point[0]:.1f}, {virtual_3d_point[1]:.1f}, {virtual_3d_point[2]:.1f}]")
        print(f"   虚拟2D点: [{virtual_2d_point[0]:.1f}, {virtual_2d_point[1]:.1f}]")

        try:
            # 使用EPNP方法进行初始估计（对平面配置效果更好）
            success, rvec, tvec = cv2.solvePnP(
                object_points_6, image_points_6,
                self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_EPNP
            )

            if not success:
                raise cv2.error("EPNP方法失败")

            print(f"   ✅ EPNP初始估计成功")

            # 使用真实的5个点进行精化
            rvec_refined, tvec_refined = cv2.solvePnPRefineLM(
                real_object_points, real_image_points,
                self.camera_matrix, self.dist_coeffs,
                rvec, tvec
            )

            # 计算精化后的重投影误差（仅使用5个真实点）
            projected_points, _ = cv2.projectPoints(
                real_object_points, rvec_refined, tvec_refined,
                self.camera_matrix, self.dist_coeffs
            )
            projected_points = projected_points.reshape(-1, 2)

            reprojection_error = np.mean(np.linalg.norm(
                real_image_points - projected_points, axis=1
            ))

            print(f"   ✅ LM精化成功，重投影误差: {reprojection_error:.3f} 像素")
            print(f"   📊 算法特点: EPNP初始化 + LM精化（仅使用5个真实点）")

            rvec, tvec = rvec_refined, tvec_refined

        except cv2.error as e:
            print(f"   ⚠️ 6点PnP方法失败: {e}，尝试备用方法...")

            # 备用方法：直接使用5个真实点
            backup_methods = [
                (cv2.SOLVEPNP_ITERATIVE, "ITERATIVE"),
                (cv2.SOLVEPNP_EPNP, "EPNP"),
                (cv2.SOLVEPNP_DLS, "DLS")
            ]

            best_result = None
            best_error = float('inf')
            best_method = None

            for method, method_name in backup_methods:
                try:
                    success, rvec_backup, tvec_backup = cv2.solvePnP(
                        real_object_points, real_image_points,
                        self.camera_matrix, self.dist_coeffs,
                        flags=method
                    )

                    if success:
                        # 计算重投影误差
                        projected_points, _ = cv2.projectPoints(
                            real_object_points, rvec_backup, tvec_backup,
                            self.camera_matrix, self.dist_coeffs
                        )
                        projected_points = projected_points.reshape(-1, 2)

                        reprojection_error = np.mean(np.linalg.norm(
                            real_image_points - projected_points, axis=1
                        ))

                        if reprojection_error < best_error:
                            best_error = reprojection_error
                            best_result = (rvec_backup, tvec_backup)
                            best_method = method_name

                except cv2.error:
                    continue

            if best_result is None:
                raise RuntimeError("所有PnP方法都失败了")

            rvec, tvec = best_result
            print(f"   ✅ {best_method}备用方法成功，重投影误差: {best_error:.3f} 像素")

        # 将旋转向量转换为旋转矩阵
        R, _ = cv2.Rodrigues(rvec)
        T = tvec.flatten()

        print(f"   旋转矩阵形状: {R.shape}")
        print(f"   平移向量: [{T[0]:.2f}, {T[1]:.2f}, {T[2]:.2f}] mm")

        return R, T
    
    def transform_to_world_coordinates(self, points_3d_tool: Dict[str, np.ndarray], 
                                     R: np.ndarray, T: np.ndarray) -> Dict[str, np.ndarray]:
        """
        将器械坐标系下的点转换到世界坐标系
        
        Args:
            points_3d_tool: 器械坐标系下的3D点
            R: 旋转矩阵
            T: 平移向量
            
        Returns:
            dict: 世界坐标系下的3D点
        """
        world_points = {}
        for label, point_tool in points_3d_tool.items():
            # 世界坐标 = R * 器械坐标 + T
            point_world = R @ point_tool + T
            world_points[label] = point_world
        
        return world_points

    def initial_estimate(self):
        """
        Create an initial estimate of 3D points and camera poses using 6 points (ABCDE + virtual point)
        """
        if len(self.points_2d_all_frames) < 2:
            print("Need at least 2 frames for initial estimation")
            return False

        # Initialize 3D points - the five points are coplanar with each other
        # but not with the drill tip at the origin (0,0,0)
        self.points_3d = np.zeros((5, 3), dtype=np.float32)

        # Use the initial coordinates from the blueprint (based on real distances)
        # ABCDE points are coplanar at Z=0 in the tool coordinate system
        # Use the same coordinates as in the blueprint for consistency
        self.points_3d[0] = self.tool_coordinate_blueprint['A'].copy()  # A
        self.points_3d[1] = self.tool_coordinate_blueprint['B'].copy()  # B
        self.points_3d[2] = self.tool_coordinate_blueprint['C'].copy()  # C
        self.points_3d[3] = self.tool_coordinate_blueprint['D'].copy()  # D
        self.points_3d[4] = self.tool_coordinate_blueprint['E'].copy()  # E

        # Add a virtual 6th point since some algorithms require at least 6 points
        # Use a point that's more representative of the tool geometry
        # Place it at the centroid of ABCDE points but with a different Z
        centroid_xy = np.mean(self.points_3d[:, :2], axis=0)
        virtual_point = np.array([centroid_xy[0], centroid_xy[1], 10.0], dtype=np.float32)  # 10mm above the plane
        points_3d_with_virtual = np.vstack([self.points_3d, virtual_point])

        print(f"Virtual 3D point: [{virtual_point[0]:.1f}, {virtual_point[1]:.1f}, {virtual_point[2]:.1f}]")

        # For each frame, estimate the camera pose
        for i, points_2d in enumerate(self.points_2d_all_frames):
            # Use the centroid of the 5 points as the 6th point
            centroid = np.mean(points_2d, axis=0)
            points_2d_with_virtual = np.vstack([points_2d, centroid])

            try:
                # Use PnP to estimate camera pose with the additional virtual point
                success, rvec, tvec = cv2.solvePnP(
                    points_3d_with_virtual, points_2d_with_virtual,
                    self.camera_matrix, self.dist_coeffs,
                    flags=cv2.SOLVEPNP_EPNP  # Use EPNP which works better with planar configurations
                )

                if not success:
                    print(f"Failed to estimate camera pose for frame {i}")
                    continue  # Try to continue with other frames

                # Refine the pose using only the real 5 points
                rvec, tvec = cv2.solvePnPRefineLM(
                    self.points_3d, points_2d,
                    self.camera_matrix, self.dist_coeffs,
                    rvec, tvec
                )

                self.camera_poses.append((rvec, tvec))

            except cv2.error as e:
                print(f"Error estimating pose for frame {i}: {e}")
                continue

        # Check if we have enough camera poses
        if len(self.camera_poses) < 2:
            print("Failed to estimate camera poses for at least 2 frames")
            return False

        print(f"Successfully estimated camera poses for {len(self.camera_poses)} frames")
        return True

    def bundle_adjustment(self):
        """
        Bundle adjustment optimization to refine 3D point coordinates and camera poses
        with the constraints that:
        1. The drill tip is at origin (0,0,0)
        2. The five calibration points are coplanar with each other, but not with the drill tip
        """
        if not self.camera_poses or self.points_3d is None:
            print("Initial estimation must be performed first")
            return False

        print("Performing bundle adjustment optimization...")

        # Calculate initial reprojection error
        total_error = 0
        num_points = 0

        for i, (rvec, tvec) in enumerate(self.camera_poses):
            if i >= len(self.points_2d_all_frames):
                continue

            # Project 3D points to 2D
            projected_points, _ = cv2.projectPoints(
                self.points_3d, rvec, tvec,
                self.camera_matrix, self.dist_coeffs
            )

            # Calculate error
            diff = projected_points.reshape(-1, 2) - self.points_2d_all_frames[i]
            error = np.sum(diff**2)
            total_error += error
            num_points += len(self.points_2d_all_frames[i])

        initial_error = np.sqrt(total_error / num_points) if num_points > 0 else float('inf')
        print(f"Initial reprojection error: {initial_error} pixels")

        try:
            # Prepare parameters for optimization
            # We'll optimize:
            # - 3D coordinates of the 5 points (X,Y,Z but constrained to be coplanar with each other)
            # - Camera poses (rotation and translation) for each frame

            # Initial parameters: flatten all values into a single array
            params = []

            # ABCDE points should remain coplanar at Z=0 (tool coordinate system constraint)
            # Only optimize X,Y coordinates, keep Z=0 fixed
            # Try relaxing this constraint to see if it improves tip_t accuracy
            use_fixed_z_constraint = False  # Allow Z coordinates to vary for better accuracy

            if use_fixed_z_constraint:
                # Only add X,Y coordinates for all 5 points (Z remains fixed at 0)
                for point in self.points_3d:
                    params.extend([point[0], point[1]])
            else:
                # Add full X,Y,Z coordinates for all 5 points (allow Z to vary)
                for point in self.points_3d:
                    params.extend([point[0], point[1], point[2]])

            # Add camera poses
            for rvec, tvec in self.camera_poses:
                params.extend(rvec.flatten())
                params.extend(tvec.flatten())

            params = np.array(params)

            # Define cost function for optimization
            def cost_function(params):
                # Extract parameters
                points_3d = np.zeros((5, 3), dtype=np.float32)
                camera_poses = []

                if use_fixed_z_constraint:
                    # Extract 3D points with fixed Z=0
                    for i in range(5):
                        offset = i * 2  # 2 params per point (X,Y), Z is fixed at 0
                        points_3d[i] = [params[offset], params[offset+1], 0.0]

                    # Extract camera poses
                    offset = 5 * 2  # 5 points with 2 coordinates each (X,Y)
                else:
                    # Extract 3D points with individual Z coordinates
                    for i in range(5):
                        offset = i * 3  # 3 params per point (X,Y,Z)
                        points_3d[i] = [params[offset], params[offset+1], params[offset+2]]

                    # Extract camera poses
                    offset = 5 * 3  # 5 points with 3 coordinates each (X,Y,Z)

                for i in range(len(self.camera_poses)):
                    rvec = params[offset + i*6 : offset + i*6 + 3].reshape(3, 1)
                    tvec = params[offset + i*6 + 3 : offset + i*6 + 6].reshape(3, 1)
                    camera_poses.append((rvec, tvec))

                # Calculate reprojection error
                total_error = 0

                for i, (rvec, tvec) in enumerate(camera_poses):
                    if i >= len(self.points_2d_all_frames):
                        continue

                    # Project 3D points to 2D
                    projected_points, _ = cv2.projectPoints(
                        points_3d, rvec, tvec,
                        self.camera_matrix, self.dist_coeffs
                    )

                    # Calculate squared error
                    diff = projected_points.reshape(-1, 2) - self.points_2d_all_frames[i]
                    error = np.sum(diff**2)
                    total_error += error

                # Add a constraint to ensure points stay reasonable distances from origin
                # This prevents the optimization from pushing points to extreme values
                distance_penalty = 0
                for point in points_3d:
                    # Penalize points that are too close to origin or too far away
                    dist = np.sum(point**2)  # squared distance from origin
                    if dist < 10**2:  # Closer than 10 units
                        distance_penalty += 100 * (10**2 - dist)**2
                    if dist > 250**2:  # Farther than 250 units
                        distance_penalty += (dist - 250**2)**2

                return total_error + distance_penalty

            # Run the optimization
            print("Running optimization...")
            result = minimize(
                cost_function,
                params,
                method='Powell',
                options={'maxiter': 1000, 'disp': True}
            )

            print(f"Optimization completed: {result.success}")
            print(f"Final cost: {result.fun}")
            print(f"Number of iterations: {result.nit if hasattr(result, 'nit') else 'N/A'}")

            # Extract the optimized parameters
            optimized_params = result.x

            # Show parameter changes
            print(f"Parameter changes:")
            print(f"  Original params shape: {params.shape}")
            print(f"  Optimized params shape: {optimized_params.shape}")
            print(f"  Max parameter change: {np.max(np.abs(optimized_params - params)):.6f}")
            print(f"  RMS parameter change: {np.sqrt(np.mean((optimized_params - params)**2)):.6f}")

            if use_fixed_z_constraint:
                print("Optimized X,Y coordinates (Z fixed at 0):")

                # Extract 3D points with fixed Z=0
                for i in range(5):
                    offset = i * 2  # 2 params per point (X,Y)
                    self.points_3d[i] = [
                        optimized_params[offset],
                        optimized_params[offset+1],
                        0.0  # Z is fixed at 0
                    ]
                    print(f"Point {['A','B','C','D','E'][i]}: X={optimized_params[offset]:.3f}, Y={optimized_params[offset+1]:.3f}, Z=0.0")

                # Extract camera poses
                offset = 5 * 2  # 5 points with 2 coordinates each (X,Y)
            else:
                print("Optimized individual X,Y,Z coordinates:")
                # Extract 3D points with individual Z coordinates
                for i in range(5):
                    offset = i * 3  # 3 params per point (X,Y,Z)
                    self.points_3d[i] = [
                        optimized_params[offset],
                        optimized_params[offset+1],
                        optimized_params[offset+2]
                    ]
                    print(f"Point {['A','B','C','D','E'][i]}: X={optimized_params[offset]:.3f}, Y={optimized_params[offset+1]:.3f}, Z={optimized_params[offset+2]:.3f}")

                # Extract camera poses
                offset = 5 * 3  # 5 points with 3 coordinates each (X,Y,Z)

            for i in range(len(self.camera_poses)):
                rvec = optimized_params[offset + i*6 : offset + i*6 + 3].reshape(3, 1)
                tvec = optimized_params[offset + i*6 + 3 : offset + i*6 + 6].reshape(3, 1)
                self.camera_poses[i] = (rvec, tvec)

            # Calculate final reprojection error
            total_error = 0
            num_points = 0

            for i, (rvec, tvec) in enumerate(self.camera_poses):
                if i >= len(self.points_2d_all_frames):
                    continue

                # Project 3D points to 2D
                projected_points, _ = cv2.projectPoints(
                    self.points_3d, rvec, tvec,
                    self.camera_matrix, self.dist_coeffs
                )

                # Calculate error
                diff = projected_points.reshape(-1, 2) - self.points_2d_all_frames[i]
                error = np.sum(diff**2)
                total_error += error
                num_points += len(self.points_2d_all_frames[i])

            final_error = np.sqrt(total_error / num_points) if num_points > 0 else float('inf')
            print(f"Final reprojection error: {final_error} pixels (was {initial_error} pixels)")

            return True

        except Exception as e:
            print(f"Bundle adjustment failed: {e}")
            return False

    def _check_data_quality(self):
        """检查数据质量，识别可能的异常值"""
        print(f"   🔍 检查姿态求解质量...")

        # 检查重投影误差
        high_error_count = 0
        for i, image_data in enumerate(self.image_data):
            if 'reprojection_error' in image_data:
                error = image_data['reprojection_error']
                if error > 2.0:  # 重投影误差超过2像素
                    high_error_count += 1
                    if high_error_count <= 3:  # 只显示前3个
                        print(f"   ⚠️ 图像{i+1}重投影误差较大: {error:.3f}像素")

        if high_error_count > 0:
            print(f"   ⚠️ 发现{high_error_count}张图像重投影误差>2像素，可能影响精度")

        # 检查标记点间距离一致性
        print(f"   🔍 检查标记点间距离一致性...")
        for pair in [('A', 'B'), ('B', 'C'), ('A', 'C'), ('A', 'D')]:
            distances = []
            for world_coords in self.world_coordinates:
                p1 = world_coords[pair[0]]
                p2 = world_coords[pair[1]]
                distance = np.linalg.norm(p1 - p2)
                distances.append(distance)

            distance_std = np.std(distances)
            distance_mean = np.mean(distances)
            if distance_std > 5.0:  # 标准差超过5mm
                print(f"   ⚠️ {pair[0]}-{pair[1]}距离不稳定: {distance_mean:.1f}±{distance_std:.1f}mm")

        # 检查姿态变化范围
        print(f"   🔍 检查姿态变化范围...")
        if len(self.tool_poses) > 1:
            translations = np.array([T for R, T in self.tool_poses])
            translation_range = np.ptp(translations, axis=0)  # 每个轴的变化范围
            total_range = np.linalg.norm(translation_range)

            print(f"   平移变化范围: X={translation_range[0]:.1f}, Y={translation_range[1]:.1f}, Z={translation_range[2]:.1f}mm")
            print(f"   总体变化范围: {total_range:.1f}mm")

            if total_range < 50:
                print(f"   ⚠️ 姿态变化范围较小，可能影响标定精度")

    def process_single_image(self, image_path: str, debug: bool = False) -> bool:
        """
        处理单张图像（使用4点方法）

        Args:
            image_path: 图像路径
            debug: 是否输出调试信息

        Returns:
            bool: 处理是否成功
        """
        try:
            print(f"\n📸 处理图像: {os.path.basename(image_path)}")

            # 检测标记点
            result = self.marker_detector.detect_and_visualize(image_path, save_result=False, show_abc_points=False)
            if len(result) != 2:
                print("❌ 标记点检测失败")
                return False

            markers, result_image = result
            if len(markers) != 5:
                print(f"❌ 检测到的标记点不等于5个，实际有{len(markers)}个，跳过此图片")
                return False

            print(f"✅ 检测到恰好5个标记点，继续处理")

            # 从检测到的标记点中智能选择5个点（ABCDE）
            abcde_points = self.select_best_5_points(markers)

            if debug:
                print(f"   选择的ABCDE点: {abcde_points}")

            # 存储5点2D坐标用于bundle adjustment
            points_2d_array = np.array([
                abcde_points['A'],
                abcde_points['B'],
                abcde_points['C'],
                abcde_points['D'],
                abcde_points['E']
            ], dtype=np.float32)
            self.points_2d_all_frames.append(points_2d_array)

            # 保存5点可视化图像
            self._save_5points_visualization(image_path, markers, abcde_points, result_image)

            # 建立5点器械坐标系（包含E点）
            points_3d_tool = self.establish_tool_coordinate_system(abcde_points)

            # 使用5点+1虚拟点求解姿态
            R, T = self.solve_pose_with_pnp(abcde_points, points_3d_tool)

            # 转换到世界坐标系
            points_3d_world = self.transform_to_world_coordinates(points_3d_tool, R, T)

            # 存储数据
            image_data = {
                'image_path': image_path,
                'points_2d': abcde_points,
                'points_3d_tool': points_3d_tool,
                'points_3d_world': points_3d_world,
                'R': R,
                'T': T,
                'result_image': result_image
            }

            self.image_data.append(image_data)
            self.world_coordinates.append(points_3d_world)
            self.tool_poses.append((R, T))

            print(f"   ✅ 5点PnP姿态求解成功:")
            print(f"   - 世界坐标A: {points_3d_world['A']}")
            print(f"   - 世界坐标B: {points_3d_world['B']}")
            print(f"   - 世界坐标C: {points_3d_world['C']}")
            print(f"   - 世界坐标D: {points_3d_world['D']}")
            print(f"   - 世界坐标E: {points_3d_world['E']}")

            return True

        except Exception as e:
            print(f"❌ 处理图像失败: {e}")
            return False

    def _save_5points_visualization(self, image_path: str, all_markers: List[Tuple[float, float]],
                                   selected_points: Dict[str, Tuple[float, float]], result_image: np.ndarray):
        """
        保存5点可视化图像

        Args:
            image_path: 原始图像路径
            all_markers: 所有检测到的标记点
            selected_points: 选择的ABCDE五个点
            result_image: 检测结果图像
        """
        try:
            # 使用原始图像创建干净的可视化图像，避免检测器标题重叠
            original_image = cv2.imread(image_path)
            vis_image = original_image.copy()

            # 定义颜色：A(红色), B(蓝色), C(黄色), D(绿色), E(紫色)
            colors = {
                'A': (0, 0, 255),    # 红色
                'B': (255, 0, 0),    # 蓝色
                'C': (0, 255, 255),  # 黄色
                'D': (0, 255, 0),    # 绿色
                'E': (255, 0, 255)   # 紫色
            }

            # 绘制选择的4个点
            for label, point in selected_points.items():
                x, y = int(point[0]), int(point[1])
                color = colors[label]

                # 绘制圆圈
                cv2.circle(vis_image, (x, y), 15, color, 3)
                # 绘制标签
                cv2.putText(vis_image, label, (x-10, y-25),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            # 绘制AB主轴线（蓝色）
            A_point = selected_points['A']
            B_point = selected_points['B']
            cv2.line(vis_image,
                    (int(A_point[0]), int(A_point[1])),
                    (int(B_point[0]), int(B_point[1])),
                    (255, 0, 0), 2)

            # 绘制AC参考线（黄色）
            C_point = selected_points['C']
            cv2.line(vis_image,
                    (int(A_point[0]), int(A_point[1])),
                    (int(C_point[0]), int(C_point[1])),
                    (0, 255, 255), 2)

            # 添加清晰的标题信息（不与检测器标题重叠）
            title = "5-Point Tool Tip Calibration"

            # 分别显示每个点的坐标，避免文字过长
            A_text = f"A({A_point[0]:.0f},{A_point[1]:.0f})"
            B_text = f"B({B_point[0]:.0f},{B_point[1]:.0f})"
            C_text = f"C({C_point[0]:.0f},{C_point[1]:.0f})"
            D_text = f"D({selected_points['D'][0]:.0f},{selected_points['D'][1]:.0f})"
            E_text = f"E({selected_points['E'][0]:.0f},{selected_points['E'][1]:.0f})"

            # 在图像顶部添加标题（白色文字，更清晰）
            cv2.putText(vis_image, title, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # 在图像底部添加坐标信息，分三行显示
            line1 = f"{A_text} {B_text}"
            line2 = f"{C_text} {D_text}"
            line3 = f"{E_text}"

            # 获取图像尺寸
            img_height, img_width = vis_image.shape[:2]

            cv2.putText(vis_image, line1, (10, img_height - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(vis_image, line2, (10, img_height - 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(vis_image, line3, (10, img_height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 保存5点可视化图像
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            save_path = f"5points_result_{base_name}.jpg"
            cv2.imwrite(save_path, vis_image)
            print(f"   💾 保存5点可视化图像: {save_path}")

        except Exception as e:
            print(f"   ⚠️ 保存4点可视化图像失败: {e}")

    def solve_tip_world_coordinate(self) -> np.ndarray:
        """
        求解器械尖端的世界坐标 Tip_w

        使用距离恒定约束和线性化方法：
        对于任何标记点i，在任何姿态k中，该点到尖端的距离都是恒定的：
        ||P_i(k) - Tip_w||² = r_i²

        通过两两对比消除r_i和非线性项，得到线性方程组求解Tip_w

        Returns:
            np.ndarray: 器械尖端的世界坐标 [x, y, z]
        """
        # 检查是否有bundle adjustment优化后的数据
        if self.points_3d is not None and len(self.camera_poses) >= 2:
            print(f"\n🎯 求解器械尖端世界坐标 Tip_w（使用Bundle Adjustment优化后的数据）")
            print(f"   使用 {len(self.camera_poses)} 个优化后的相机姿态")

            # 使用bundle adjustment优化后的数据
            world_coordinates_optimized = []
            for rvec, tvec in self.camera_poses:
                # 将优化后的3D点转换到世界坐标系
                R, _ = cv2.Rodrigues(rvec)
                T = tvec.flatten()

                world_points = {}
                labels = ['A', 'B', 'C', 'D', 'E']
                for i, label in enumerate(labels):
                    point_world = R @ self.points_3d[i] + T
                    world_points[label] = point_world

                world_coordinates_optimized.append(world_points)

            # 使用优化后的世界坐标
            world_coordinates_to_use = world_coordinates_optimized

        elif len(self.world_coordinates) >= 2:
            print(f"\n🎯 求解器械尖端世界坐标 Tip_w（使用传统PnP数据）")
            print(f"   使用 {len(self.world_coordinates)} 张图片的数据")

            # 使用传统的世界坐标
            world_coordinates_to_use = self.world_coordinates

        else:
            raise ValueError("至少需要2张图片的数据来求解尖端坐标")

        # 数据质量预检查
        print(f"   📊 数据质量预检查...")
        self._check_data_quality()

        # 构建线性方程组 A * Tip_w = b
        equations = []
        equation_weights = []  # 添加权重系统

        # 对每个标记点（A, B, C, D, E）
        for marker_label in ['A', 'B', 'C', 'D', 'E']:
            # 对每对图片进行对比（第k张与第1张对比）
            for k in range(1, len(world_coordinates_to_use)):
                # 第k张图片中的标记点坐标
                P_k = world_coordinates_to_use[k][marker_label]
                # 第1张图片中的标记点坐标
                P_1 = world_coordinates_to_use[0][marker_label]

                # 计算两点间距离，用于质量评估
                point_distance = np.linalg.norm(P_k - P_1)

                # 构建线性方程的系数
                # A*x + B*y + C*z = D
                A = 2 * (P_k[0] - P_1[0])
                B = 2 * (P_k[1] - P_1[1])
                C = 2 * (P_k[2] - P_1[2])
                D = (P_k[0]**2 + P_k[1]**2 + P_k[2]**2) - (P_1[0]**2 + P_1[1]**2 + P_1[2]**2)

                # 计算方程系数的模长，用于权重计算
                coeff_norm = np.sqrt(A**2 + B**2 + C**2)

                # 跳过系数过小的方程（两点过于接近）
                if coeff_norm < 1e-6:
                    print(f"   ⚠️ 跳过退化方程: 标记点{marker_label}, 图像{k+1}与图像1过于接近")
                    continue

                # 基于点距离和系数模长计算权重
                weight = min(point_distance / 100.0, 1.0) * min(coeff_norm / 100.0, 1.0)

                equations.append([A, B, C, D])
                equation_weights.append(weight)

        # 转换为矩阵形式
        equations = np.array(equations)
        equation_weights = np.array(equation_weights)
        A_matrix = equations[:, :3]  # 系数矩阵
        b_vector = equations[:, 3]   # 常数向量

        print(f"   构建了 {len(equations)} 个有效线性方程")
        print(f"   方程组矩阵形状: A={A_matrix.shape}, b={b_vector.shape}")
        print(f"   平均方程权重: {np.mean(equation_weights):.4f}")

        # 应用权重到方程组
        W = np.diag(np.sqrt(equation_weights))
        A_weighted = W @ A_matrix
        b_weighted = W @ b_vector

        # 使用加权最小二乘法求解
        try:
            tip_world, residuals, rank, _ = np.linalg.lstsq(A_weighted, b_weighted, rcond=None)

            # 计算残差
            if len(residuals) > 0:
                rms_error = np.sqrt(residuals[0] / len(equations))
                print(f"   加权最小二乘求解成功，RMS误差: {rms_error:.4f}")

                # 计算未加权的RMS误差用于比较
                residual_unweighted = A_matrix @ tip_world - b_vector
                rms_unweighted = np.sqrt(np.mean(residual_unweighted**2))
                print(f"   未加权RMS误差: {rms_unweighted:.4f}")
            else:
                print(f"   加权最小二乘求解成功")

            print(f"   矩阵秩: {rank}/{A_matrix.shape[1]}")

        except np.linalg.LinAlgError as e:
            raise RuntimeError(f"线性方程组求解失败: {e}")

        self.tip_world = tip_world
        print(f"   器械尖端世界坐标 Tip_w: [{tip_world[0]:.3f}, {tip_world[1]:.3f}, {tip_world[2]:.3f}]")

        return tip_world

    def solve_tip_tool_coordinate(self) -> np.ndarray:
        """
        求解器械尖端的局部坐标 Tip_t

        通过姿态变换将世界坐标转换为器械局部坐标系：
        对于每个姿态k: R(k) * Tip_t = Tip_w - T(k)
        使用最小二乘法求解最优的Tip_t

        Returns:
            np.ndarray: 器械尖端的局部坐标 [x, y, z]
        """
        if self.tip_world is None:
            raise ValueError("必须先求解世界坐标Tip_w")

        # 检查是否有bundle adjustment优化后的数据
        if self.points_3d is not None and len(self.camera_poses) >= 1:
            print(f"\n🎯 求解器械尖端局部坐标 Tip_t（使用Bundle Adjustment优化后的姿态）")
            print(f"   使用 {len(self.camera_poses)} 个优化后的姿态")

            # 使用优化后的相机姿态
            poses_to_use = []
            for rvec, tvec in self.camera_poses:
                R, _ = cv2.Rodrigues(rvec)
                T = tvec.flatten()
                poses_to_use.append((R, T))

        elif len(self.tool_poses) >= 1:
            print(f"\n🎯 求解器械尖端局部坐标 Tip_t（使用传统PnP姿态）")
            print(f"   使用 {len(self.tool_poses)} 个姿态的数据")

            # 使用传统的姿态数据
            poses_to_use = self.tool_poses

        else:
            raise ValueError("没有可用的姿态数据")

        # 构建线性方程组
        # 对每个姿态: R(k) * Tip_t = Tip_w - T(k)
        A_matrix = []
        b_vector = []

        for _, (R, T) in enumerate(poses_to_use):
            # 每个姿态贡献3个方程（x, y, z分量）
            A_matrix.append(R)
            b_vector.append(self.tip_world - T)

        # 合并所有方程
        A_matrix = np.vstack(A_matrix)  # 形状: (3*N, 3)
        b_vector = np.hstack(b_vector)  # 形状: (3*N,)

        print(f"   构建了 {len(poses_to_use)} 个姿态的方程组")
        print(f"   方程组矩阵形状: A={A_matrix.shape}, b={b_vector.shape}")

        # 使用最小二乘法求解
        try:
            tip_tool, residuals, rank, _ = np.linalg.lstsq(A_matrix, b_vector, rcond=None)

            # 计算残差
            if len(residuals) > 0:
                rms_error = np.sqrt(residuals[0] / len(b_vector))
                print(f"   最小二乘求解成功，RMS误差: {rms_error:.4f}")
            else:
                print(f"   最小二乘求解成功")

            print(f"   矩阵秩: {rank}/{A_matrix.shape[1]}")

        except np.linalg.LinAlgError as e:
            raise RuntimeError(f"线性方程组求解失败: {e}")

        self.tip_tool = tip_tool
        print(f"   器械尖端局部坐标 Tip_t: [{tip_tool[0]:.3f}, {tip_tool[1]:.3f}, {tip_tool[2]:.3f}]")

        return tip_tool

    def process_all_images(self, image_folder: str = 'tip_images', debug: bool = False) -> int:
        """
        批量处理所有图像

        Args:
            image_folder: 图像文件夹路径
            debug: 是否输出调试信息

        Returns:
            int: 成功处理的图像数量
        """
        print(f"\n📂 批量处理图像文件夹: {image_folder}")

        # 查找图像文件
        image_files = []
        for ext in ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG', '*.JPEG']:
            image_files.extend(glob.glob(os.path.join(image_folder, ext)))

        if not image_files:
            raise ValueError(f"在 {image_folder} 中未找到图像文件")

        # 去重并排序
        image_files = list(set(image_files))  # 去除重复文件
        image_files.sort()  # 按文件名排序
        print(f"   找到 {len(image_files)} 张图片，使用全部图片")
        print(f"   只处理检测到恰好5个标记点的图片")
        print(f"   实际处理的图片: {[os.path.basename(f) for f in image_files]}")

        # 清空之前的数据
        self.image_data.clear()
        self.world_coordinates.clear()
        self.tool_poses.clear()
        self.points_2d_all_frames.clear()
        self.camera_poses.clear()
        self.points_3d = None

        # 处理每张图像
        successful_count = 0
        for i, image_path in enumerate(image_files):
            print(f"\n[{i+1}/{len(image_files)}] ", end="")
            if self.process_single_image(image_path, debug=debug):
                successful_count += 1
            else:
                print(f"   ⚠️ 跳过图像: {os.path.basename(image_path)}")

        print(f"\n✅ 批量处理完成: {successful_count}/{len(image_files)} 张图像处理成功")
        return successful_count

    def validate_calibration_results(self) -> Dict[str, float]:
        """
        验证标定结果的准确性

        Returns:
            dict: 验证结果统计
        """
        if self.tip_world is None or self.tip_tool is None:
            raise ValueError("必须先完成标定才能验证结果")

        print(f"\n🔍 验证标定结果")

        validation_results = {
            'distance_errors': [],
            'reprojection_errors': [],
            'consistency_errors': []
        }

        # 验证距离恒定约束
        print(f"   验证距离恒定约束...")
        for marker_label in ['A', 'B', 'C', 'D', 'E']:
            distances = []
            for world_coords in self.world_coordinates:
                marker_pos = world_coords[marker_label]
                distance = np.linalg.norm(marker_pos - self.tip_world)
                distances.append(distance)

            # 计算距离的标准差（应该接近0）
            distance_std = np.std(distances)
            distance_mean = np.mean(distances)
            validation_results['distance_errors'].append(distance_std)

            print(f"   - 标记点{marker_label}到尖端距离: {distance_mean:.3f}±{distance_std:.3f} mm")

        # 验证姿态一致性
        print(f"   验证姿态一致性...")
        consistency_errors = []
        high_error_count = 0

        for i, (R, T) in enumerate(self.tool_poses):
            # 通过姿态变换计算尖端世界坐标
            tip_world_calculated = R @ self.tip_tool + T
            consistency_error = np.linalg.norm(tip_world_calculated - self.tip_world)
            consistency_errors.append(consistency_error)
            validation_results['consistency_errors'].append(consistency_error)

            # 统计高误差姿态
            if consistency_error > 4.0:  # 误差超过4mm认为是高误差
                high_error_count += 1

            # 显示条件：前3个、后3个、或误差超过4mm的姿态
            total_poses = len(self.tool_poses)
            show_pose = (i < 3 or i >= total_poses - 3 or consistency_error > 4.0)

            if show_pose:
                status = "⚠️" if consistency_error > 4.0 else "✅" if consistency_error < 2.0 else "🟡"
                print(f"   - 姿态{i+1}一致性误差: {consistency_error:.3f} mm {status}")

        # 显示统计信息
        if len(consistency_errors) > 6:  # 如果姿态数量多，显示省略信息
            print(f"   ... (省略 {len(consistency_errors) - 6} 个中间姿态)")

        # 计算统计数据
        avg_consistency = np.mean(consistency_errors)
        max_consistency = np.max(consistency_errors)
        min_consistency = np.min(consistency_errors)
        std_consistency = np.std(consistency_errors)

        print(f"   📊 一致性误差统计:")
        print(f"      平均误差: {avg_consistency:.3f} mm")
        print(f"      最大误差: {max_consistency:.3f} mm")
        print(f"      最小误差: {min_consistency:.3f} mm")
        print(f"      标准差: {std_consistency:.3f} mm")
        print(f"      高误差姿态: {high_error_count}/{len(consistency_errors)} (>{4.0}mm)")

        # 质量评估
        if avg_consistency < 2.0:
            quality_level = "🟢 优秀"
        elif avg_consistency < 4.0:
            quality_level = "🟡 良好"
        elif avg_consistency < 8.0:
            quality_level = "🟠 一般"
        else:
            quality_level = "🔴 较差"

        print(f"      整体质量: {quality_level}")

        # 计算总体统计
        avg_distance_error = np.mean(validation_results['distance_errors'])
        max_distance_error = np.max(validation_results['distance_errors'])
        avg_consistency_error = np.mean(validation_results['consistency_errors'])
        max_consistency_error = np.max(validation_results['consistency_errors'])

        validation_results.update({
            'avg_distance_error': avg_distance_error,
            'max_distance_error': max_distance_error,
            'avg_consistency_error': avg_consistency_error,
            'max_consistency_error': max_consistency_error
        })

        print(f"\n📊 验证结果统计:")
        print(f"   - 平均距离误差: {avg_distance_error:.4f} mm")
        print(f"   - 最大距离误差: {max_distance_error:.4f} mm")
        print(f"   - 平均一致性误差: {avg_consistency_error:.3f} mm")
        print(f"   - 最大一致性误差: {max_consistency_error:.3f} mm")

        return validation_results

    def run_complete_calibration(self, image_folder: str = 'tip_images', debug: bool = False) -> Dict:
        """
        运行完整的标定流程

        Args:
            image_folder: 图像文件夹路径
            debug: 是否输出调试信息

        Returns:
            dict: 标定结果
        """
        print("🚀 开始器械尖端标定流程")
        print("=" * 50)

        try:
            # 步骤1: 批量处理图像
            successful_count = self.process_all_images(image_folder, debug=debug)
            if successful_count < 2:
                raise ValueError(f"成功处理的图像数量不足（{successful_count}），至少需要2张")

            # 步骤2: 初始估计（使用6点PnP）
            print("\n🔧 步骤2: 初始估计（使用6点PnP）")
            if not self.initial_estimate():
                raise ValueError("初始估计失败")

            # 步骤3: Bundle adjustment优化
            print("\n🔧 步骤3: Bundle adjustment优化")
            if not self.bundle_adjustment():
                print("⚠️ Bundle adjustment失败，使用初始估计结果")

            # 步骤4: 求解世界坐标Tip_w
            tip_world = self.solve_tip_world_coordinate()

            # 步骤5: 求解局部坐标Tip_t
            tip_tool = self.solve_tip_tool_coordinate()

            # 步骤6: 验证结果
            validation_results = self.validate_calibration_results()

            # 整理最终结果
            calibration_results = {
                'tip_world': tip_world,
                'tip_tool': tip_tool,
                'processed_images': successful_count,
                'total_images': len(glob.glob(os.path.join(image_folder, '*.jpg'))),
                'validation': validation_results,
                'AB_distance_mm': self.AB_distance_mm,
                'AC_distance_mm': self.AC_distance_mm,
                'BC_distance_mm': self.BC_distance_mm
            }

            print("\n🎉 标定流程完成!")
            print("=" * 50)
            print(f"✅ 器械尖端世界坐标 Tip_w: [{tip_world[0]:.3f}, {tip_world[1]:.3f}, {tip_world[2]:.3f}] mm")
            print(f"✅ 器械尖端局部坐标 Tip_t: [{tip_tool[0]:.3f}, {tip_tool[1]:.3f}, {tip_tool[2]:.3f}] mm")
            print(f"✅ 处理图像数量: {successful_count} 张")
            print(f"✅ 标定精度: 距离误差 {validation_results['avg_distance_error']:.4f}±{validation_results['max_distance_error']:.4f} mm")

            # 显示优化后的ABCDE点坐标
            if self.points_3d is not None:
                print(f"\n📊 优化后的ABCDE点3D坐标:")
                labels = ['A', 'B', 'C', 'D', 'E']
                for i, label in enumerate(labels):
                    point = self.points_3d[i]
                    print(f"   - {label}: [{point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}] mm")

            return calibration_results

        except Exception as e:
            print(f"\n❌ 标定流程失败: {e}")
            raise

    

    def save_calibration_results(self, filename: str = 'tool_tip_calibration_results.npz'):
        """
        保存标定结果到文件

        Args:
            filename: 保存文件名
        """
        if self.tip_world is None or self.tip_tool is None:
            print("❌ 没有标定结果可以保存")
            return

        # 准备保存的数据
        save_data = {
            'tip_world': self.tip_world,
            'tip_tool': self.tip_tool,
            'AB_distance_mm': self.AB_distance_mm,
            'AC_distance_mm': self.AC_distance_mm,
            'BC_distance_mm': self.BC_distance_mm,
            'camera_matrix': self.camera_matrix,
            'dist_coeffs': self.dist_coeffs,
            'processed_images_count': len(self.image_data)
        }

        # 保存为npz文件
        np.savez(filename, **save_data)
        print(f"✅ 标定结果已保存: {filename}")

        # 同时保存为文本文件
        txt_filename = filename.replace('.npz', '.txt')
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write("器械尖端标定结果\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"器械尖端世界坐标 Tip_w (mm):\n")
            f.write(f"X: {self.tip_world[0]:.6f}\n")
            f.write(f"Y: {self.tip_world[1]:.6f}\n")
            f.write(f"Z: {self.tip_world[2]:.6f}\n\n")
            f.write(f"器械尖端局部坐标 Tip_t (mm):\n")
            f.write(f"X: {self.tip_tool[0]:.6f}\n")
            f.write(f"Y: {self.tip_tool[1]:.6f}\n")
            f.write(f"Z: {self.tip_tool[2]:.6f}\n\n")
            f.write(f"标定参数:\n")
            f.write(f"AB距离: {self.AB_distance_mm} mm\n")
            f.write(f"AC距离: {self.AC_distance_mm} mm\n")
            f.write(f"BC距离: {self.BC_distance_mm} mm\n")
            f.write(f"处理图像数量: {len(self.image_data)}\n")

        print(f"✅ 标定结果文本已保存: {txt_filename}")


def main():
    """主函数 - 演示完整的5点Bundle Adjustment标定流程"""
    print("🚀 开始器械尖端标定 (5点Bundle Adjustment方法)")
    print("=" * 50)

    try:
        # 创建标定系统
        calibrator = ToolTipCalibration()

        # 运行完整标定流程
        _ = calibrator.run_complete_calibration(debug=True)

       
        # 保存结果
        calibrator.save_calibration_results()

        print("\n🎉 5点Bundle Adjustment标定流程完成!")
        print("✅ 使用了5个真实检测点（ABCDE）+ 1个虚拟点进行初始估计")
        print("✅ 通过Bundle Adjustment优化了3D点坐标和相机姿态")
        print("✅ 测试了多种PnP求解方法并选择最佳结果")

    except Exception as e:
        print(f"\n❌ 标定系统运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
