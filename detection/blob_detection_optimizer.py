#!/usr/bin/env python3
"""
Blob检测参数优化工具

专门用于优化Blob检测的参数，提供实时调试界面

使用方法:
python blob_detection_optimizer.py

作者: AI Assistant
日期: 2025-01-18
"""

import cv2
import numpy as np
import os
import glob
from marker_detector import MarkerDetector

class BlobDetectionOptimizer:
    def __init__(self, calibration_file='../calibration/camera_calibration.npz'):
        """初始化优化器"""
        self.detector = MarkerDetector(calibration_file)

        # 可调参数
        self.params = {
            'min_area': 100,
            'max_area': 2000,
            'min_circularity': 0.6,
            'min_convexity': 0.85,
            'min_inertia_ratio': 0.3,
            'clahe_clip_limit': 2.0,
            'clahe_tile_size': 8,
            'use_gaussian_blur': False,
            'blur_kernel_size': 3,
        }
        
        self.current_image = None
        self.current_gray = None
        self.current_markers = []
    
    def detect_with_blob(self, image):
        """使用当前参数进行Blob检测"""
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        self.current_gray = gray
        
        # 图像预处理
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
        
        # 设置Blob检测参数
        blob_params = cv2.SimpleBlobDetector_Params()
        
        # 按颜色过滤
        blob_params.filterByColor = True
        blob_params.blobColor = 0  # 检测深色blob
        
        # 按面积过滤
        blob_params.filterByArea = True
        blob_params.minArea = self.params['min_area']
        blob_params.maxArea = self.params['max_area']
        
        # 按圆度过滤
        blob_params.filterByCircularity = True
        blob_params.minCircularity = self.params['min_circularity']
        
        # 按凸性过滤
        blob_params.filterByConvexity = True
        blob_params.minConvexity = self.params['min_convexity']
        
        # 按惯性过滤
        blob_params.filterByInertia = True
        blob_params.minInertiaRatio = self.params['min_inertia_ratio']
        
        # 创建检测器
        detector = cv2.SimpleBlobDetector_create(blob_params)
        
        # 执行检测
        keypoints = detector.detect(enhanced)
        
        # 转换为标记点格式
        markers = []
        for kp in keypoints:
            markers.append([kp.pt[0], kp.pt[1], kp.size])  # 包含大小信息
        
        self.current_markers = np.array(markers, dtype=np.float32)
        return self.current_markers
    
    def create_result_visualization(self):
        """创建结果可视化"""
        if self.current_image is None:
            return None
        
        # 创建结果图像
        result_img = self.current_image.copy()
        
        # 绘制检测到的标记点
        for i, marker in enumerate(self.current_markers):
            x, y = int(marker[0]), int(marker[1])
            size = int(marker[2]) if len(marker) > 2 else 8
            
            # 绘制圆圈，大小根据检测到的blob大小调整
            cv2.circle(result_img, (x, y), max(size//2, 5), (0, 255, 0), 2)
            cv2.putText(result_img, f"{i+1}", (x+12, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 添加参数信息
        info_text = [
            f"Markers: {len(self.current_markers)}",
            f"Area: {self.params['min_area']}-{self.params['max_area']}",
            f"Circularity: {self.params['min_circularity']:.1f}",
            f"Convexity: {self.params['min_convexity']:.1f}",
            f"Inertia: {self.params['min_inertia_ratio']:.1f}",
            f"CLAHE: {self.params['clahe_clip_limit']:.1f}"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(result_img, text, (10, 30 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(result_img, text, (10, 30 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        return result_img
    
    def interactive_optimization(self, image_path):
        """交互式参数优化"""
        print(f"\n=== Blob检测参数优化 ===")
        print(f"图像: {os.path.basename(image_path)}")
        print("\n控制说明:")
        print("  q/w: 减少/增加最小面积")
        print("  a/s: 减少/增加最大面积")
        print("  z/x: 减少/增加最小圆度")
        print("  e/r: 减少/增加最小凸性")
        print("  t/y: 减少/增加最小惯性比")
        print("  u/i: 减少/增加CLAHE剪切限制")
        print("  b: 切换高斯模糊")
        print("  g: 重置参数")
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
        cv2.namedWindow('Blob Detection Optimization', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Processed Image', cv2.WINDOW_NORMAL)
        
        while True:
            # 检测标记点
            self.detect_with_blob(self.current_image)
            
            # 创建可视化
            result_img = self.create_result_visualization()
            
            # 显示处理后的图像
            clahe = cv2.createCLAHE(
                clipLimit=self.params['clahe_clip_limit'], 
                tileGridSize=(self.params['clahe_tile_size'], self.params['clahe_tile_size'])
            )
            processed = clahe.apply(self.current_gray)
            
            # 显示图像
            cv2.imshow('Blob Detection Optimization', result_img)
            cv2.imshow('Processed Image', processed)
            
            # 等待按键
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == ord(' '):  # SPACE - 保存参数
                self.save_optimized_parameters()
                print("参数已保存!")
            elif key == ord('g'):  # 重置参数
                self.reset_parameters()
                print("参数已重置!")
            elif key == ord('b'):  # 切换高斯模糊
                self.params['use_gaussian_blur'] = not self.params['use_gaussian_blur']
                print(f"高斯模糊: {'开启' if self.params['use_gaussian_blur'] else '关闭'}")
            elif key == ord('q'):  # 减少最小面积
                self.params['min_area'] = max(5, self.params['min_area'] - 5)
                print(f"最小面积: {self.params['min_area']}")
            elif key == ord('w'):  # 增加最小面积
                self.params['min_area'] = min(100, self.params['min_area'] + 5)
                print(f"最小面积: {self.params['min_area']}")
            elif key == ord('a'):  # 减少最大面积
                self.params['max_area'] = max(50, self.params['max_area'] - 20)
                print(f"最大面积: {self.params['max_area']}")
            elif key == ord('s'):  # 增加最大面积
                self.params['max_area'] = min(500, self.params['max_area'] + 20)
                print(f"最大面积: {self.params['max_area']}")
            elif key == ord('z'):  # 减少最小圆度
                self.params['min_circularity'] = max(0.1, self.params['min_circularity'] - 0.1)
                print(f"最小圆度: {self.params['min_circularity']:.1f}")
            elif key == ord('x'):  # 增加最小圆度
                self.params['min_circularity'] = min(0.9, self.params['min_circularity'] + 0.1)
                print(f"最小圆度: {self.params['min_circularity']:.1f}")
            elif key == ord('e'):  # 减少最小凸性
                self.params['min_convexity'] = max(0.1, self.params['min_convexity'] - 0.1)
                print(f"最小凸性: {self.params['min_convexity']:.1f}")
            elif key == ord('r'):  # 增加最小凸性
                self.params['min_convexity'] = min(0.9, self.params['min_convexity'] + 0.1)
                print(f"最小凸性: {self.params['min_convexity']:.1f}")
            elif key == ord('t'):  # 减少最小惯性比
                self.params['min_inertia_ratio'] = max(0.1, self.params['min_inertia_ratio'] - 0.1)
                print(f"最小惯性比: {self.params['min_inertia_ratio']:.1f}")
            elif key == ord('y'):  # 增加最小惯性比
                self.params['min_inertia_ratio'] = min(0.9, self.params['min_inertia_ratio'] + 0.1)
                print(f"最小惯性比: {self.params['min_inertia_ratio']:.1f}")
            elif key == ord('u'):  # 减少CLAHE剪切限制
                self.params['clahe_clip_limit'] = max(0.5, self.params['clahe_clip_limit'] - 0.5)
                print(f"CLAHE剪切限制: {self.params['clahe_clip_limit']:.1f}")
            elif key == ord('i'):  # 增加CLAHE剪切限制
                self.params['clahe_clip_limit'] = min(5.0, self.params['clahe_clip_limit'] + 0.5)
                print(f"CLAHE剪切限制: {self.params['clahe_clip_limit']:.1f}")
            
            # 显示当前参数
            if key != 255:  # 如果有按键按下
                print(f"\n当前检测结果: {len(self.current_markers)} 个标记点")
        
        cv2.destroyAllWindows()
    
    def reset_parameters(self):
        """重置参数到默认值"""
        self.params = {
            'min_area': 10,
            'max_area': 200,
            'min_circularity': 0.3,
            'min_convexity': 0.5,
            'min_inertia_ratio': 0.3,
            'clahe_clip_limit': 2.0,
            'clahe_tile_size': 8,
            'use_gaussian_blur': False,
            'blur_kernel_size': 3,
        }
    
    def save_optimized_parameters(self):
        """保存优化后的参数"""
        config_content = f"""# 优化后的Blob检测参数
# 由Blob检测优化工具生成

OPTIMIZED_BLOB_PARAMS = {{
    'min_area': {self.params['min_area']},
    'max_area': {self.params['max_area']},
    'min_circularity': {self.params['min_circularity']:.1f},
    'min_convexity': {self.params['min_convexity']:.1f},
    'min_inertia_ratio': {self.params['min_inertia_ratio']:.1f},
    'clahe_clip_limit': {self.params['clahe_clip_limit']:.1f},
    'clahe_tile_size': {self.params['clahe_tile_size']},
    'use_gaussian_blur': {self.params['use_gaussian_blur']},
    'blur_kernel_size': {self.params['blur_kernel_size']},
}}

# 使用方法：
# 在 tool_tip_3d_localization.py 中导入这些参数
# from optimized_blob_params import OPTIMIZED_BLOB_PARAMS
"""
        
        with open('optimized_blob_params.py', 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        print("优化参数已保存到 optimized_blob_params.py")
    
    def batch_test(self, image_folder="../tip_images"):
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
            markers = self.detect_with_blob(undistorted_img)
            
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
    print("=== Blob检测参数优化工具 ===")
    
    optimizer = BlobDetectionOptimizer()
    
    # 获取图像文件 (从detection文件夹运行，需要上级目录)
    image_files = sorted(glob.glob("../tip_images/*.jpg"))
    if not image_files:
        image_files = sorted(glob.glob("../tip_images/*.png"))

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
