#!/usr/bin/env python3
"""
标记点检测器

专门用于检测图像中的圆形标记点
支持多种检测算法和参数优化

作者: AI Assistant
日期: 2025-01-18
"""

import cv2
import numpy as np
import os
from typing import List, Tuple, Optional
# 优化后的检测参数配置
OPTIMIZED_PARAMS = {
    'min_threshold': 40,
    'max_threshold': 160,
    'threshold_step': 5,
    'min_area': 100,
    'max_area': 200,        # 优化后: 200
    'min_circularity': 0.8, # 优化后: 0.8
    'min_convexity': 0.85,
    'min_inertia_ratio': 0.3,
    'filter_by_color': True,
    'blob_color': 0,
    'clahe_clip_limit': 2.0,
    'clahe_tile_size': 8,
}

class MarkerDetector:
    """标记点检测器"""
    
    def __init__(self, camera_calibration_file=None):
        """
        初始化标记点检测器
        
        Args:
            camera_calibration_file: 相机标定文件路径（可选）
        """
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # 如果提供了标定文件，加载相机参数
        if camera_calibration_file:
            self.load_camera_parameters(camera_calibration_file)
    
    def load_camera_parameters(self, calibration_file):
        """加载相机标定参数"""
        # 智能查找标定文件
        if not os.path.exists(calibration_file):
            possible_paths = [
                calibration_file,
                f'calibration/{calibration_file}',
                f'../calibration/{calibration_file}',
                'calibration/camera_calibration.npz',
                '../calibration/camera_calibration.npz'
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    calibration_file = path
                    break
            else:
                print(f"警告: 找不到相机标定文件 {calibration_file}")
                return
        
        try:
            calib_data = np.load(calibration_file)
            self.camera_matrix = calib_data['camera_matrix']
            self.dist_coeffs = calib_data['dist_coeffs']
            print(f"✅ 相机标定参数加载成功: {calibration_file}")
        except Exception as e:
            print(f"❌ 加载相机标定参数失败: {e}")
    
    def detect_markers_blob(self, image, debug=False):
        """
        使用Blob检测方法检测标记点
        
        Args:
            image: 输入图像
            debug: 是否显示调试信息
            
        Returns:
            markers: 检测到的标记点坐标列表 [(x, y), ...]
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 图像预处理 - 直方图均衡化
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # 使用优化后的参数配置
        config = OPTIMIZED_PARAMS

        # 设置Blob检测参数 - 使用优化后的配置
        params = cv2.SimpleBlobDetector_Params()

        # 阈值设置
        params.minThreshold = config['min_threshold']
        params.maxThreshold = config['max_threshold']
        params.thresholdStep = config['threshold_step']

        # 颜色过滤 - 检测深色标记点
        params.filterByColor = config['filter_by_color']
        params.blobColor = config['blob_color']

        # 面积过滤 - 使用优化后参数
        params.filterByArea = True
        params.minArea = config['min_area']      # 100
        params.maxArea = config['max_area']      # 200 (优化后)

        # 圆度过滤 - 使用优化后参数
        params.filterByCircularity = True
        params.minCircularity = config['min_circularity']  # 0.8 (优化后)

        # 凸度过滤 - 确保形状规整
        params.filterByConvexity = True
        params.minConvexity = config['min_convexity']      # 0.85

        # 惯性过滤 - 椭圆度要求
        params.filterByInertia = True
        params.minInertiaRatio = config['min_inertia_ratio']  # 0.3
        
        # 创建检测器并执行检测
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(enhanced)
        
        # 转换为坐标列表
        markers = [(kp.pt[0], kp.pt[1]) for kp in keypoints]
        
        if debug:
            print(f"检测到 {len(markers)} 个标记点")
            for i, (x, y) in enumerate(markers):
                print(f"  标记点{i+1}: ({x:.1f}, {y:.1f})")
        
        return markers

    def select_three_key_points(self, markers):
        """
        从检测到的标记点中选择三个关键点A、B、C

        算法：
        1. 找到最长边的两个端点作为候选A、B
        2. 找到与A、B距离最短的第三个点C
        3. 连接最长边和最短边的点为A，另一个为B

        Args:
            markers: 检测到的标记点列表 [(x, y), ...]

        Returns:
            dict: {'A': (x, y), 'B': (x, y), 'C': (x, y)} 或 None
        """
        if len(markers) < 3:
            print(f"标记点数量不足，需要至少3个点，当前只有{len(markers)}个")
            return None

        # 计算所有点对之间的距离
        distances = []
        for i in range(len(markers)):
            for j in range(i + 1, len(markers)):
                p1, p2 = markers[i], markers[j]
                dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                distances.append({
                    'distance': dist,
                    'point1_idx': i,
                    'point2_idx': j,
                    'point1': p1,
                    'point2': p2
                })

        # 按距离排序，找到最长边
        distances.sort(key=lambda x: x['distance'], reverse=True)
        longest_edge = distances[0]

        # 最长边的两个端点作为候选A、B
        candidate_a_idx = longest_edge['point1_idx']
        candidate_b_idx = longest_edge['point2_idx']
        candidate_a = longest_edge['point1']
        candidate_b = longest_edge['point2']

        print(f"最长边距离: {longest_edge['distance']:.1f} 像素")
        print(f"最长边端点: ({candidate_a[0]:.1f}, {candidate_a[1]:.1f}) - ({candidate_b[0]:.1f}, {candidate_b[1]:.1f})")

        # 找到其他点中与A、B距离最短的点作为C
        other_points = []
        for i, marker in enumerate(markers):
            if i != candidate_a_idx and i != candidate_b_idx:
                other_points.append((i, marker))

        if not other_points:
            print("没有足够的其他点来选择C点")
            return None

        # 计算其他点到A、B的最短距离
        min_distance_to_ab = float('inf')
        point_c_idx = -1
        point_c = None

        for idx, point in other_points:
            # 计算到A的距离
            dist_to_a = np.sqrt((point[0] - candidate_a[0])**2 + (point[1] - candidate_a[1])**2)
            # 计算到B的距离
            dist_to_b = np.sqrt((point[0] - candidate_b[0])**2 + (point[1] - candidate_b[1])**2)
            # 取较短的距离
            min_dist = min(dist_to_a, dist_to_b)

            if min_dist < min_distance_to_ab:
                min_distance_to_ab = min_dist
                point_c_idx = idx
                point_c = point

        print(f"选择的C点: ({point_c[0]:.1f}, {point_c[1]:.1f}), 到AB最短距离: {min_distance_to_ab:.1f}")

        # 确定A、B的顺序：连接最长边和最短边的点为A
        # 计算C到A和C到B的距离
        dist_c_to_a = np.sqrt((point_c[0] - candidate_a[0])**2 + (point_c[1] - candidate_a[1])**2)
        dist_c_to_b = np.sqrt((point_c[0] - candidate_b[0])**2 + (point_c[1] - candidate_b[1])**2)

        # 连接最长边和最短边的点为A
        if dist_c_to_a < dist_c_to_b:
            # C更接近candidate_a，所以candidate_a是A
            point_a = candidate_a
            point_b = candidate_b
            print(f"A点 (连接最长边和最短边): ({point_a[0]:.1f}, {point_a[1]:.1f})")
            print(f"B点: ({point_b[0]:.1f}, {point_b[1]:.1f})")
        else:
            # C更接近candidate_b，所以candidate_b是A
            point_a = candidate_b
            point_b = candidate_a
            print(f"A点 (连接最长边和最短边): ({point_a[0]:.1f}, {point_a[1]:.1f})")
            print(f"B点: ({point_b[0]:.1f}, {point_b[1]:.1f})")

        return {
            'A': point_a,
            'B': point_b,
            'C': point_c
        }

    def undistort_image(self, image):
        """
        图像去畸变
        
        Args:
            image: 输入图像
            
        Returns:
            undistorted_image: 去畸变后的图像
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            print("警告: 未加载相机标定参数，跳过去畸变")
            return image
        
        return cv2.undistort(image, self.camera_matrix, self.dist_coeffs)
    
    def detect_and_visualize(self, image_path, save_result=False, show_abc_points=True):
        """
        检测标记点并可视化结果

        Args:
            image_path: 图像文件路径
            save_result: 是否保存结果图像
            show_abc_points: 是否显示A、B、C三个关键点

        Returns:
            markers: 检测到的标记点坐标
            result_image: 标注了标记点的结果图像
            abc_points: A、B、C三个关键点 (如果show_abc_points=True)
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 去畸变（如果有标定参数）
        if self.camera_matrix is not None:
            image = self.undistort_image(image)
        
        # 检测标记点
        markers = self.detect_markers_blob(image, debug=True)
        
        # 创建结果图像
        result_image = image.copy()
        abc_points = None

        # 选择A、B、C三个关键点
        if show_abc_points and len(markers) >= 3:
            abc_points = self.select_three_key_points(markers)

            if abc_points:
                # 先绘制所有检测点（灰色）
                for i, (x, y) in enumerate(markers):
                    cv2.circle(result_image, (int(x), int(y)), 8, (128, 128, 128), 2)
                    cv2.putText(result_image, f"{i+1}", (int(x)+12, int(y)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)

                # 绘制A、B、C三个关键点（红色，更大）
                colors = {'A': (0, 0, 255), 'B': (0, 0, 255), 'C': (0, 0, 255)}  # 红色
                for label, (x, y) in abc_points.items():
                    # 绘制大红圈
                    cv2.circle(result_image, (int(x), int(y)), 15, colors[label], 4)
                    # 添加A、B、C标签
                    cv2.putText(result_image, label, (int(x)+20, int(y)),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, colors[label], 3)

                # 绘制连接线
                # A-B线（最长边）
                cv2.line(result_image,
                        (int(abc_points['A'][0]), int(abc_points['A'][1])),
                        (int(abc_points['B'][0]), int(abc_points['B'][1])),
                        (255, 0, 0), 2)  # 蓝色线

                # A-C线或B-C线（最短边）
                dist_ac = np.sqrt((abc_points['A'][0] - abc_points['C'][0])**2 +
                                 (abc_points['A'][1] - abc_points['C'][1])**2)
                dist_bc = np.sqrt((abc_points['B'][0] - abc_points['C'][0])**2 +
                                 (abc_points['B'][1] - abc_points['C'][1])**2)

                if dist_ac < dist_bc:
                    # A-C是最短边
                    cv2.line(result_image,
                            (int(abc_points['A'][0]), int(abc_points['A'][1])),
                            (int(abc_points['C'][0]), int(abc_points['C'][1])),
                            (0, 255, 255), 2)  # 黄色线
                else:
                    # B-C是最短边
                    cv2.line(result_image,
                            (int(abc_points['B'][0]), int(abc_points['B'][1])),
                            (int(abc_points['C'][0]), int(abc_points['C'][1])),
                            (0, 255, 255), 2)  # 黄色线

                # 分行显示信息文本，避免截断
                info_text1 = f"Detected: {len(markers)} markers"
                info_text2 = f"A({abc_points['A'][0]:.0f},{abc_points['A'][1]:.0f}) B({abc_points['B'][0]:.0f},{abc_points['B'][1]:.0f}) C({abc_points['C'][0]:.0f},{abc_points['C'][1]:.0f})"
            else:
                # 如果无法选择ABC点，使用普通绘制
                for i, (x, y) in enumerate(markers):
                    cv2.circle(result_image, (int(x), int(y)), 12, (0, 255, 0), 3)
                    cv2.putText(result_image, f"{i+1}", (int(x)+15, int(y)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                info_text = f"Detected: {len(markers)} markers (无法选择ABC点)"
        else:
            # 普通绘制模式
            for i, (x, y) in enumerate(markers):
                cv2.circle(result_image, (int(x), int(y)), 12, (0, 255, 0), 3)
                cv2.putText(result_image, f"{i+1}", (int(x)+15, int(y)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            info_text = f"Detected: {len(markers)} markers"

        # 添加信息文本
        if show_abc_points and abc_points:
            # 分两行显示，避免文本过长被截断
            # 第一行：检测数量
            cv2.putText(result_image, info_text1, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
            cv2.putText(result_image, info_text1, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # 第二行：ABC坐标
            cv2.putText(result_image, info_text2, (10, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
            cv2.putText(result_image, info_text2, (10, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)  # 红色文字
        else:
            # 单行显示
            cv2.putText(result_image, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
            cv2.putText(result_image, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # 保存结果（如果需要）
        if save_result:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            result_path = f"detection_result_{base_name}.jpg"
            cv2.imwrite(result_path, result_image)
            print(f"结果已保存到: {result_path}")
        
        if show_abc_points:
            return markers, result_image, abc_points
        else:
            return markers, result_image

def main():
    """主函数 - 演示用法"""
    print("🔍 标记点检测器")
    print("=" * 40)
    
    # 创建检测器
    detector = MarkerDetector('camera_calibration.npz')
    
    # 查找图像文件
    image_folder = 'tip_images'
    if not os.path.exists(image_folder):
        print(f"❌ 图像文件夹不存在: {image_folder}")
        return
    
    image_files = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        image_files.extend(glob.glob(os.path.join(image_folder, ext)))
    
    if not image_files:
        print(f"❌ 在 {image_folder} 中未找到图像文件")
        return
    
    print(f"📸 找到 {len(image_files)} 张图像")
    
    # 处理第一张图像作为演示
    test_image = image_files[0]
    print(f"\n🔍 测试图像: {os.path.basename(test_image)}")
    
    try:
        result = detector.detect_and_visualize(test_image, save_result=True)
        if len(result) == 3:
            markers, result_image, abc_points = result
            print(f"选择的ABC点: {abc_points}")
        else:
            markers, result_image = result
        
        # 显示结果
        cv2.imshow('Marker Detection Result', result_image)
        print("\n按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"❌ 处理图像时出错: {e}")

if __name__ == "__main__":
    import glob
    main()
