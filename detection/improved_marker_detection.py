#!/usr/bin/env python3
"""
改进的标记点检测工具

针对您遇到的误检测问题，提供更精确的标记点检测和参数调试功能。

使用方法:
python improved_marker_detection.py

作者: AI Assistant
日期: 2025-01-18
"""

import cv2
import numpy as np
import os
import glob
from tool_tip_3d_localization import ToolTip3DLocalizer

class ImprovedMarkerDetector:
    def __init__(self, calibration_file='camera_calibration.npz'):
        """初始化改进的标记点检测器"""
        self.localizer = ToolTip3DLocalizer(calibration_file)
        
        # 改进的检测参数
        self.improved_params = {
            # 基本形状筛选
            'min_area': 80,              # 增加最小面积，过滤小噪点
            'max_area': 800,             # 减少最大面积，避免大区域误检
            'min_circularity': 0.6,      # 提高圆度要求
            'max_aspect_ratio': 1.5,     # 添加长宽比限制
            
            # 阈值参数
            'adaptive_thresh_block_size': 15,  # 增加块大小
            'adaptive_thresh_c': 3,            # 调整常数
            
            # 形态学参数
            'kernel_size': (2, 2),       # 减小核大小，保持细节
            'open_iterations': 2,        # 增加开运算，去除噪点
            'close_iterations': 1,
            
            # 新增筛选条件
            'min_solidity': 0.8,         # 实心度（轮廓面积/凸包面积）
            'max_extent': 0.9,           # 范围度（轮廓面积/边界矩形面积）
            'min_extent': 0.5,
            
            # 颜色筛选（针对深色标记点）
            'use_color_filter': True,
            'max_brightness': 100,       # 最大亮度值（0-255）
            
            # 空间筛选
            'use_spatial_filter': True,
            'min_distance_between_markers': 30,  # 标记点之间最小距离
        }
    
    def detect_markers_improved(self, image, debug=False):
        """改进的标记点检测"""
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 颜色预筛选（如果启用）
        if self.improved_params['use_color_filter']:
            # 创建亮度掩码，只保留较暗的区域
            brightness_mask = gray < self.improved_params['max_brightness']
            gray_filtered = gray.copy()
            gray_filtered[~brightness_mask] = 255  # 将亮区域设为白色
        else:
            gray_filtered = gray
        
        # 自适应阈值
        thresh = cv2.adaptiveThreshold(
            gray_filtered, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
            self.improved_params['adaptive_thresh_block_size'],
            self.improved_params['adaptive_thresh_c']
        )
        
        # 形态学操作
        kernel = np.ones(self.improved_params['kernel_size'], np.uint8)
        thresh_clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel,
                                      iterations=self.improved_params['open_iterations'])
        thresh_clean = cv2.morphologyEx(thresh_clean, cv2.MORPH_CLOSE, kernel,
                                      iterations=self.improved_params['close_iterations'])
        
        # 查找轮廓
        contours, _ = cv2.findContours(thresh_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 筛选标记点
        candidates = []
        debug_info = []
        
        for i, contour in enumerate(contours):
            # 基本面积筛选
            area = cv2.contourArea(contour)
            if not (self.improved_params['min_area'] < area < self.improved_params['max_area']):
                continue
            
            # 计算各种几何特征
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            # 圆度
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < self.improved_params['min_circularity']:
                continue
            
            # 长宽比
            rect = cv2.minAreaRect(contour)
            width, height = rect[1]
            if width == 0 or height == 0:
                continue
            aspect_ratio = max(width, height) / min(width, height)
            if aspect_ratio > self.improved_params['max_aspect_ratio']:
                continue
            
            # 实心度（轮廓面积/凸包面积）
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue
            solidity = area / hull_area
            if solidity < self.improved_params['min_solidity']:
                continue
            
            # 范围度（轮廓面积/边界矩形面积）
            x, y, w, h = cv2.boundingRect(contour)
            rect_area = w * h
            if rect_area == 0:
                continue
            extent = area / rect_area
            if not (self.improved_params['min_extent'] < extent < self.improved_params['max_extent']):
                continue
            
            # 计算质心
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # 检查亮度（在原始灰度图中）
            if self.improved_params['use_color_filter']:
                local_brightness = gray[max(0, cy-5):min(gray.shape[0], cy+5), 
                                      max(0, cx-5):min(gray.shape[1], cx+5)]
                avg_brightness = np.mean(local_brightness)
                if avg_brightness > self.improved_params['max_brightness']:
                    continue
            
            candidates.append({
                'center': [cx, cy],
                'area': area,
                'circularity': circularity,
                'aspect_ratio': aspect_ratio,
                'solidity': solidity,
                'extent': extent,
                'contour': contour
            })
            
            debug_info.append(f"候选点{len(candidates)}: 面积={area:.0f}, 圆度={circularity:.2f}, "
                            f"长宽比={aspect_ratio:.2f}, 实心度={solidity:.2f}")
        
        # 空间筛选：移除距离太近的点
        if self.improved_params['use_spatial_filter'] and len(candidates) > 1:
            candidates = self.spatial_filter(candidates)
        
        # 转换为numpy数组
        markers = np.array([c['center'] for c in candidates], dtype=np.float32)
        
        if debug:
            print(f"改进检测结果: {len(markers)} 个标记点")
            for info in debug_info:
                print(f"  {info}")
            
            # 显示调试图像
            self.show_debug_images(image, gray, thresh, thresh_clean, candidates)
        
        return markers, candidates
    
    def spatial_filter(self, candidates):
        """空间筛选：移除距离太近的标记点"""
        if len(candidates) <= 1:
            return candidates
        
        # 按面积排序，优先保留大的标记点
        candidates.sort(key=lambda x: x['area'], reverse=True)
        
        filtered = []
        min_dist = self.improved_params['min_distance_between_markers']
        
        for candidate in candidates:
            cx, cy = candidate['center']
            
            # 检查与已选择点的距离
            too_close = False
            for selected in filtered:
                sx, sy = selected['center']
                distance = np.sqrt((cx - sx)**2 + (cy - sy)**2)
                if distance < min_dist:
                    too_close = True
                    break
            
            if not too_close:
                filtered.append(candidate)
        
        return filtered
    
    def show_debug_images(self, original, gray, thresh, thresh_clean, candidates):
        """显示调试图像"""
        # 创建候选点图像
        candidates_img = original.copy()
        for i, candidate in enumerate(candidates):
            cx, cy = candidate['center']
            cv2.circle(candidates_img, (cx, cy), 8, (0, 255, 0), 2)
            cv2.putText(candidates_img, f"{i+1}", (cx+10, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 显示图像
        cv2.imshow('1. 原图', cv2.resize(original, (400, 300)))
        cv2.imshow('2. 灰度图', cv2.resize(gray, (400, 300)))
        cv2.imshow('3. 阈值图', cv2.resize(thresh, (400, 300)))
        cv2.imshow('4. 形态学处理', cv2.resize(thresh_clean, (400, 300)))
        cv2.imshow('5. 改进检测结果', cv2.resize(candidates_img, (400, 300)))
        
        print("按任意键继续...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def test_single_image(self, image_path, debug=True):
        """测试单张图像"""
        print(f"\n测试图像: {os.path.basename(image_path)}")
        
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图像: {image_path}")
            return
        
        # 去畸变
        undistorted_img = cv2.undistort(img, self.localizer.camera_matrix, self.localizer.dist_coeffs)
        
        # 原始检测
        print("\n=== 原始检测方法 ===")
        original_markers = self.localizer.detect_markers(undistorted_img, debug=False)
        print(f"原始方法检测到: {len(original_markers)} 个标记点")
        
        # 改进检测
        print("\n=== 改进检测方法 ===")
        improved_markers, candidates = self.detect_markers_improved(undistorted_img, debug=debug)
        print(f"改进方法检测到: {len(improved_markers)} 个标记点")
        
        # 比较结果
        self.compare_results(undistorted_img, original_markers, improved_markers, candidates)
    
    def compare_results(self, image, original_markers, improved_markers, candidates):
        """比较检测结果"""
        # 创建对比图像
        comparison_img = np.hstack([image.copy(), image.copy()])
        
        # 左侧：原始检测结果
        for i, marker in enumerate(original_markers):
            cv2.circle(comparison_img, (int(marker[0]), int(marker[1])), 8, (0, 255, 0), 2)
            cv2.putText(comparison_img, f"O{i+1}", (int(marker[0])+10, int(marker[1])), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 右侧：改进检测结果
        offset_x = image.shape[1]
        for i, marker in enumerate(improved_markers):
            cv2.circle(comparison_img, (int(marker[0]) + offset_x, int(marker[1])), 8, (0, 0, 255), 2)
            cv2.putText(comparison_img, f"I{i+1}", (int(marker[0]) + offset_x + 10, int(marker[1])), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 添加标题
        cv2.putText(comparison_img, "Original Detection", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison_img, "Improved Detection", (offset_x + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow('检测结果对比', comparison_img)
        print("\n按任意键继续...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def interactive_parameter_tuning(self, image_path):
        """交互式参数调试"""
        print(f"\n=== 交互式参数调试 ===")
        print(f"图像: {os.path.basename(image_path)}")
        print("使用说明:")
        print("  调整参数后按回车查看效果")
        print("  输入 'q' 退出")
        print("  输入 'save' 保存当前参数")

        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图像: {image_path}")
            return

        undistorted_img = cv2.undistort(img, self.localizer.camera_matrix, self.localizer.dist_coeffs)

        while True:
            print(f"\n当前参数:")
            print(f"1. min_area: {self.improved_params['min_area']}")
            print(f"2. max_area: {self.improved_params['max_area']}")
            print(f"3. min_circularity: {self.improved_params['min_circularity']}")
            print(f"4. max_brightness: {self.improved_params['max_brightness']}")
            print(f"5. min_distance_between_markers: {self.improved_params['min_distance_between_markers']}")

            # 检测并显示结果
            markers, candidates = self.detect_markers_improved(undistorted_img, debug=False)

            # 创建结果图像
            result_img = undistorted_img.copy()
            for i, marker in enumerate(markers):
                cv2.circle(result_img, (int(marker[0]), int(marker[1])), 10, (0, 0, 255), -1)
                cv2.putText(result_img, f"{i+1}", (int(marker[0])+12, int(marker[1])),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('参数调试结果', result_img)
            cv2.waitKey(1)  # 非阻塞显示

            print(f"\n检测结果: {len(markers)} 个标记点")

            # 获取用户输入
            user_input = input("\n输入参数编号和新值 (如: 1 100) 或 'q' 退出: ").strip()

            if user_input.lower() == 'q':
                break
            elif user_input.lower() == 'save':
                self.save_parameters()
                continue

            try:
                parts = user_input.split()
                if len(parts) == 2:
                    param_idx = int(parts[0])
                    new_value = float(parts[1])

                    if param_idx == 1:
                        self.improved_params['min_area'] = int(new_value)
                    elif param_idx == 2:
                        self.improved_params['max_area'] = int(new_value)
                    elif param_idx == 3:
                        self.improved_params['min_circularity'] = new_value
                    elif param_idx == 4:
                        self.improved_params['max_brightness'] = int(new_value)
                    elif param_idx == 5:
                        self.improved_params['min_distance_between_markers'] = int(new_value)
                    else:
                        print("无效的参数编号")
                else:
                    print("输入格式错误，请使用: 参数编号 新值")
            except ValueError:
                print("输入格式错误")

        cv2.destroyAllWindows()

    def save_parameters(self):
        """保存优化后的参数到配置文件"""
        config_content = f"""# 优化后的标记点检测参数
# 由改进检测工具生成

IMPROVED_MARKER_DETECTION_PARAMS = {{
    'min_area': {self.improved_params['min_area']},
    'max_area': {self.improved_params['max_area']},
    'min_circularity': {self.improved_params['min_circularity']:.2f},
    'max_aspect_ratio': {self.improved_params['max_aspect_ratio']:.2f},
    'adaptive_thresh_block_size': {self.improved_params['adaptive_thresh_block_size']},
    'adaptive_thresh_c': {self.improved_params['adaptive_thresh_c']},
    'kernel_size': {self.improved_params['kernel_size']},
    'open_iterations': {self.improved_params['open_iterations']},
    'close_iterations': {self.improved_params['close_iterations']},
    'min_solidity': {self.improved_params['min_solidity']:.2f},
    'max_extent': {self.improved_params['max_extent']:.2f},
    'min_extent': {self.improved_params['min_extent']:.2f},
    'use_color_filter': {self.improved_params['use_color_filter']},
    'max_brightness': {self.improved_params['max_brightness']},
    'use_spatial_filter': {self.improved_params['use_spatial_filter']},
    'min_distance_between_markers': {self.improved_params['min_distance_between_markers']},
}}
"""

        with open('optimized_params.py', 'w', encoding='utf-8') as f:
            f.write(config_content)

        print("参数已保存到 optimized_params.py")

def main():
    """主函数"""
    print("=== 改进的标记点检测工具 ===")

    detector = ImprovedMarkerDetector()

    # 获取图像文件
    image_files = sorted(glob.glob("tip_images/*.jpg"))
    if not image_files:
        image_files = sorted(glob.glob("tip_images/*.png"))

    if not image_files:
        print("未找到图像文件，请检查 tip_images 文件夹")
        return

    print(f"找到 {len(image_files)} 张图像")
    print("\n选择测试模式:")
    print("1. 测试单张图像（详细调试）")
    print("2. 测试所有图像（快速对比）")
    print("3. 交互式参数调试")

    choice = input("请输入选择 (1, 2 或 3): ").strip()

    if choice == '1':
        # 选择图像
        print("\n可用图像:")
        for i, img_path in enumerate(image_files[:10]):  # 只显示前10张
            print(f"{i+1}. {os.path.basename(img_path)}")

        try:
            img_idx = int(input(f"请选择图像 (1-{min(10, len(image_files))}): ")) - 1
            if 0 <= img_idx < len(image_files):
                detector.test_single_image(image_files[img_idx], debug=True)
            else:
                print("无效选择")
        except ValueError:
            print("无效输入")

    elif choice == '2':
        # 测试所有图像
        print("\n开始测试所有图像...")
        original_success = 0
        improved_success = 0

        for i, img_path in enumerate(image_files):
            print(f"\n处理 {i+1}/{len(image_files)}: {os.path.basename(img_path)}")

            img = cv2.imread(img_path)
            if img is None:
                continue

            undistorted_img = cv2.undistort(img, detector.localizer.camera_matrix,
                                          detector.localizer.dist_coeffs)

            # 原始检测
            original_markers = detector.localizer.detect_markers(undistorted_img, debug=False)
            original_count = len(original_markers)

            # 改进检测
            improved_markers, _ = detector.detect_markers_improved(undistorted_img, debug=False)
            improved_count = len(improved_markers)

            # 统计成功率（假设需要5-6个标记点）
            if 5 <= original_count <= 8:
                original_success += 1
            if 5 <= improved_count <= 8:
                improved_success += 1

            print(f"  原始: {original_count} 个, 改进: {improved_count} 个")

        print(f"\n=== 测试结果汇总 ===")
        print(f"总图像数量: {len(image_files)}")
        print(f"原始方法合理检测: {original_success} 张 ({original_success/len(image_files)*100:.1f}%)")
        print(f"改进方法合理检测: {improved_success} 张 ({improved_success/len(image_files)*100:.1f}%)")

    elif choice == '3':
        # 交互式参数调试
        print("\n可用图像:")
        for i, img_path in enumerate(image_files[:10]):
            print(f"{i+1}. {os.path.basename(img_path)}")

        try:
            img_idx = int(input(f"请选择图像 (1-{min(10, len(image_files))}): ")) - 1
            if 0 <= img_idx < len(image_files):
                detector.interactive_parameter_tuning(image_files[img_idx])
            else:
                print("无效选择")
        except ValueError:
            print("无效输入")

if __name__ == "__main__":
    main()
