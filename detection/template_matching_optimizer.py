#!/usr/bin/env python3
"""
模板匹配参数优化工具

专门用于优化模板匹配检测的参数，提供实时调试界面

使用方法:
python template_matching_optimizer.py

作者: AI Assistant
日期: 2025-01-18
"""

import cv2
import numpy as np
import os
import glob
from tool_tip_3d_localization import ToolTip3DLocalizer

class TemplateMatchingOptimizer:
    def __init__(self, calibration_file='camera_calibration.npz'):
        """初始化优化器"""
        self.localizer = ToolTip3DLocalizer(calibration_file)
        
        # 可调参数
        self.params = {
            'template_sizes': [4, 5, 6, 7, 8, 9, 10, 12],
            'match_threshold': 0.65,
            'min_distance': 25,
            'clahe_clip_limit': 2.0,
            'clahe_tile_size': 8,
            'use_gaussian_blur': False,
            'blur_kernel_size': 3,
            'template_type': 'filled_circle',  # 'filled_circle', 'ring', 'cross'
        }
        
        self.current_image = None
        self.current_enhanced = None
        self.current_markers = []
    
    def create_template(self, radius, template_type='filled_circle'):
        """创建不同类型的模板"""
        size = radius * 2 + 1
        template = np.zeros((size, size), dtype=np.uint8)
        
        if template_type == 'filled_circle':
            cv2.circle(template, (radius, radius), radius, 255, -1)
        elif template_type == 'ring':
            cv2.circle(template, (radius, radius), radius, 255, 2)
        elif template_type == 'cross':
            cv2.line(template, (radius-radius//2, radius), (radius+radius//2, radius), 255, 2)
            cv2.line(template, (radius, radius-radius//2), (radius, radius+radius//2), 255, 2)
        
        return template
    
    def preprocess_image(self, image):
        """图像预处理"""
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 直方图均衡化
        clahe = cv2.createCLAHE(
            clipLimit=self.params['clahe_clip_limit'], 
            tileGridSize=(self.params['clahe_tile_size'], self.params['clahe_tile_size'])
        )
        enhanced = clahe.apply(gray)
        
        # 可选的高斯模糊
        if self.params['use_gaussian_blur']:
            kernel_size = self.params['blur_kernel_size']
            if kernel_size % 2 == 0:
                kernel_size += 1
            enhanced = cv2.GaussianBlur(enhanced, (kernel_size, kernel_size), 0)
        
        return gray, enhanced
    
    def detect_with_template_matching(self, image):
        """使用当前参数进行模板匹配检测"""
        gray, enhanced = self.preprocess_image(image)
        self.current_enhanced = enhanced
        
        all_matches = []
        
        for template_size in self.params['template_sizes']:
            # 创建模板
            template = self.create_template(template_size, self.params['template_type'])
            
            # 模板匹配
            result = cv2.matchTemplate(enhanced, template, cv2.TM_CCOEFF_NORMED)
            
            # 查找匹配位置
            locations = np.where(result >= self.params['match_threshold'])
            
            for pt in zip(*locations[::-1]):
                x, y = pt[0] + template_size, pt[1] + template_size
                confidence = result[pt[1], pt[0]]
                all_matches.append([x, y, template_size, confidence])
        
        # 去除重复检测
        if len(all_matches) > 0:
            all_matches = np.array(all_matches)
            markers = self.remove_duplicate_detections(all_matches)
            self.current_markers = markers[:, :2].astype(np.float32)
        else:
            self.current_markers = np.array([], dtype=np.float32).reshape(0, 2)
        
        return self.current_markers
    
    def remove_duplicate_detections(self, markers):
        """移除重复检测"""
        if len(markers) == 0:
            return markers
        
        # 按置信度排序
        sorted_indices = np.argsort(markers[:, 3])[::-1]
        sorted_markers = markers[sorted_indices]
        
        filtered = []
        for marker in sorted_markers:
            x, y = marker[0], marker[1]
            
            # 检查与已选择标记点的距离
            too_close = False
            for selected in filtered:
                sx, sy = selected[0], selected[1]
                distance = np.sqrt((x - sx)**2 + (y - sy)**2)
                if distance < self.params['min_distance']:
                    too_close = True
                    break
            
            if not too_close:
                filtered.append(marker)
        
        return np.array(filtered)
    
    def create_result_visualization(self):
        """创建结果可视化"""
        if self.current_image is None:
            return None
        
        # 创建结果图像
        result_img = self.current_image.copy()
        
        # 绘制检测到的标记点
        for i, marker in enumerate(self.current_markers):
            x, y = int(marker[0]), int(marker[1])
            cv2.circle(result_img, (x, y), 10, (0, 0, 255), 2)
            cv2.putText(result_img, f"{i+1}", (x+12, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 添加参数信息
        info_text = [
            f"Markers: {len(self.current_markers)}",
            f"Threshold: {self.params['match_threshold']:.2f}",
            f"Min Distance: {self.params['min_distance']}",
            f"Template Sizes: {len(self.params['template_sizes'])}",
            f"CLAHE Clip: {self.params['clahe_clip_limit']:.1f}"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(result_img, text, (10, 30 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(result_img, text, (10, 30 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        return result_img
    
    def interactive_optimization(self, image_path):
        """交互式参数优化"""
        print(f"\n=== 模板匹配参数优化 ===")
        print(f"图像: {os.path.basename(image_path)}")
        print("\n控制说明:")
        print("  1-8: 调整匹配阈值 (0.1-0.8)")
        print("  q/w: 减少/增加最小距离")
        print("  a/s: 减少/增加CLAHE剪切限制")
        print("  z/x: 切换模板类型")
        print("  b: 切换高斯模糊")
        print("  r: 重置参数")
        print("  ESC: 退出")
        print("  SPACE: 保存当前参数")
        
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图像: {image_path}")
            return
        
        # 去畸变
        self.current_image = cv2.undistort(img, self.localizer.camera_matrix, self.localizer.dist_coeffs)
        
        # 创建窗口
        cv2.namedWindow('Template Matching Optimization', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Enhanced Image', cv2.WINDOW_NORMAL)
        
        template_types = ['filled_circle', 'ring', 'cross']
        template_type_idx = 0
        
        while True:
            # 检测标记点
            self.detect_with_template_matching(self.current_image)
            
            # 创建可视化
            result_img = self.create_result_visualization()
            
            # 显示图像
            cv2.imshow('Template Matching Optimization', result_img)
            cv2.imshow('Enhanced Image', self.current_enhanced)
            
            # 等待按键
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == ord(' '):  # SPACE - 保存参数
                self.save_optimized_parameters()
                print("参数已保存!")
            elif key == ord('r'):  # 重置参数
                self.reset_parameters()
                print("参数已重置!")
            elif key == ord('b'):  # 切换高斯模糊
                self.params['use_gaussian_blur'] = not self.params['use_gaussian_blur']
                print(f"高斯模糊: {'开启' if self.params['use_gaussian_blur'] else '关闭'}")
            elif key == ord('z'):  # 切换模板类型
                template_type_idx = (template_type_idx - 1) % len(template_types)
                self.params['template_type'] = template_types[template_type_idx]
                print(f"模板类型: {self.params['template_type']}")
            elif key == ord('x'):  # 切换模板类型
                template_type_idx = (template_type_idx + 1) % len(template_types)
                self.params['template_type'] = template_types[template_type_idx]
                print(f"模板类型: {self.params['template_type']}")
            elif key == ord('q'):  # 减少最小距离
                self.params['min_distance'] = max(10, self.params['min_distance'] - 5)
                print(f"最小距离: {self.params['min_distance']}")
            elif key == ord('w'):  # 增加最小距离
                self.params['min_distance'] = min(50, self.params['min_distance'] + 5)
                print(f"最小距离: {self.params['min_distance']}")
            elif key == ord('a'):  # 减少CLAHE剪切限制
                self.params['clahe_clip_limit'] = max(0.5, self.params['clahe_clip_limit'] - 0.5)
                print(f"CLAHE剪切限制: {self.params['clahe_clip_limit']}")
            elif key == ord('s'):  # 增加CLAHE剪切限制
                self.params['clahe_clip_limit'] = min(5.0, self.params['clahe_clip_limit'] + 0.5)
                print(f"CLAHE剪切限制: {self.params['clahe_clip_limit']}")
            elif ord('1') <= key <= ord('8'):  # 调整匹配阈值
                threshold_value = (key - ord('0')) * 0.1
                self.params['match_threshold'] = threshold_value
                print(f"匹配阈值: {self.params['match_threshold']:.1f}")
            
            # 显示当前参数
            if key != 255:  # 如果有按键按下
                print(f"\n当前检测结果: {len(self.current_markers)} 个标记点")
        
        cv2.destroyAllWindows()
    
    def reset_parameters(self):
        """重置参数到默认值"""
        self.params = {
            'template_sizes': [4, 5, 6, 7, 8, 9, 10, 12],
            'match_threshold': 0.65,
            'min_distance': 25,
            'clahe_clip_limit': 2.0,
            'clahe_tile_size': 8,
            'use_gaussian_blur': False,
            'blur_kernel_size': 3,
            'template_type': 'filled_circle',
        }
    
    def save_optimized_parameters(self):
        """保存优化后的参数"""
        config_content = f"""# 优化后的模板匹配参数
# 由模板匹配优化工具生成

OPTIMIZED_TEMPLATE_MATCHING_PARAMS = {{
    'template_sizes': {self.params['template_sizes']},
    'match_threshold': {self.params['match_threshold']:.2f},
    'min_distance': {self.params['min_distance']},
    'clahe_clip_limit': {self.params['clahe_clip_limit']:.1f},
    'clahe_tile_size': {self.params['clahe_tile_size']},
    'use_gaussian_blur': {self.params['use_gaussian_blur']},
    'blur_kernel_size': {self.params['blur_kernel_size']},
    'template_type': '{self.params['template_type']}',
}}

# 使用方法：
# 在 tool_tip_3d_localization.py 中导入这些参数
# from optimized_template_params import OPTIMIZED_TEMPLATE_MATCHING_PARAMS
"""
        
        with open('optimized_template_params.py', 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        print("优化参数已保存到 optimized_template_params.py")
    
    def batch_test(self, image_folder="tip_images"):
        """批量测试优化参数"""
        image_files = sorted(glob.glob(f"{image_folder}/*.jpg"))
        if not image_files:
            image_files = sorted(glob.glob(f"{image_folder}/*.png"))
        
        if not image_files:
            print(f"未找到图像文件在 {image_folder} 文件夹")
            return
        
        print(f"\n=== 批量测试 {len(image_files)} 张图像 ===")
        
        results = []
        for i, img_path in enumerate(image_files):
            print(f"处理 {i+1}/{len(image_files)}: {os.path.basename(img_path)}")
            
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            undistorted_img = cv2.undistort(img, self.localizer.camera_matrix, self.localizer.dist_coeffs)
            markers = self.detect_with_template_matching(undistorted_img)
            
            marker_count = len(markers)
            results.append(marker_count)
            print(f"  检测到 {marker_count} 个标记点")
        
        # 统计结果
        if results:
            avg_markers = np.mean(results)
            std_markers = np.std(results)
            success_rate = sum(1 for r in results if 4 <= r <= 8) / len(results) * 100
            
            print(f"\n=== 批量测试结果 ===")
            print(f"平均标记点数量: {avg_markers:.1f} ± {std_markers:.1f}")
            print(f"合理检测率 (4-8个标记点): {success_rate:.1f}%")
            print(f"最少: {min(results)}, 最多: {max(results)}")

def main():
    """主函数"""
    print("=== 模板匹配参数优化工具 ===")
    
    optimizer = TemplateMatchingOptimizer()
    
    # 获取图像文件
    image_files = sorted(glob.glob("tip_images/*.jpg"))
    if not image_files:
        image_files = sorted(glob.glob("tip_images/*.png"))
    
    if not image_files:
        print("未找到图像文件，请检查 tip_images 文件夹")
        return
    
    print(f"找到 {len(image_files)} 张图像")
    print("\n选择模式:")
    print("1. 交互式参数优化")
    print("2. 批量测试当前参数")
    
    choice = input("请输入选择 (1 或 2): ").strip()
    
    if choice == '1':
        # 选择图像进行交互式优化
        print("\n可用图像:")
        for i, img_path in enumerate(image_files[:10]):
            print(f"{i+1}. {os.path.basename(img_path)}")
        
        try:
            img_idx = int(input(f"请选择图像 (1-{min(10, len(image_files))}): ")) - 1
            if 0 <= img_idx < len(image_files):
                optimizer.interactive_optimization(image_files[img_idx])
            else:
                print("无效选择")
        except ValueError:
            print("无效输入")
    
    elif choice == '2':
        # 批量测试
        optimizer.batch_test()

if __name__ == "__main__":
    main()
