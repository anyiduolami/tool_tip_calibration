#!/usr/bin/env python3
"""
简化的Blob检测参数优化工具

提供简单的参数调整和测试功能

作者: AI Assistant
日期: 2025-01-18
"""

import cv2
import numpy as np
import os
import glob
import sys

# 添加父目录到路径以便导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detection.marker_detector import MarkerDetector

class SimpleOptimizer:
    def __init__(self):
        """初始化优化器"""
        # 获取项目根目录
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 创建检测器
        calib_path = os.path.join(self.project_root, 'calibration', 'camera_calibration.npz')
        self.detector = MarkerDetector(calib_path)
        
        # 当前参数 (优化后的最佳参数)
        self.current_params = {
            'min_area': 100,
            'max_area': 200,        # 优化后: 200
            'min_circularity': 0.8, # 优化后: 0.8
            'min_convexity': 0.85,
            'min_inertia_ratio': 0.3
        }
    
    def get_image_files(self):
        """获取图像文件列表"""
        image_folder = os.path.join(self.project_root, 'tip_images')
        
        image_files = []
        for ext in ['*.jpg', '*.png', '*.jpeg']:
            pattern = os.path.join(image_folder, ext)
            image_files.extend(glob.glob(pattern))
        
        return sorted(image_files)
    
    def detect_with_params(self, image, params):
        """使用指定参数进行检测"""
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 图像预处理
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # 设置Blob检测参数
        blob_params = cv2.SimpleBlobDetector_Params()
        
        # 阈值设置
        blob_params.minThreshold = 40
        blob_params.maxThreshold = 160
        blob_params.thresholdStep = 5
        
        # 颜色过滤
        blob_params.filterByColor = True
        blob_params.blobColor = 0
        
        # 面积过滤
        blob_params.filterByArea = True
        blob_params.minArea = params['min_area']
        blob_params.maxArea = params['max_area']
        
        # 圆度过滤
        blob_params.filterByCircularity = True
        blob_params.minCircularity = params['min_circularity']
        
        # 凸度过滤
        blob_params.filterByConvexity = True
        blob_params.minConvexity = params['min_convexity']
        
        # 惯性过滤
        blob_params.filterByInertia = True
        blob_params.minInertiaRatio = params['min_inertia_ratio']
        
        # 创建检测器并执行检测
        detector = cv2.SimpleBlobDetector_create(blob_params)
        keypoints = detector.detect(enhanced)
        
        # 转换为坐标列表
        markers = [(kp.pt[0], kp.pt[1]) for kp in keypoints]
        return markers
    
    def test_single_image(self, image_path):
        """测试单张图像"""
        print(f"\n测试图像: {os.path.basename(image_path)}")
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print("无法读取图像")
            return
        
        # 去畸变
        if self.detector.camera_matrix is not None:
            image = self.detector.undistort_image(image)
        
        # 检测标记点
        markers = self.detect_with_params(image, self.current_params)
        
        print(f"当前参数检测结果: {len(markers)} 个标记点")
        for i, (x, y) in enumerate(markers):
            print(f"  标记点{i+1}: ({x:.1f}, {y:.1f})")
        
        # 创建结果图像
        result_image = image.copy()
        for i, (x, y) in enumerate(markers):
            cv2.circle(result_image, (int(x), int(y)), 12, (0, 255, 0), 3)
            cv2.putText(result_image, f"{i+1}", (int(x)+15, int(y)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # 添加参数信息
        info_text = f"Detected: {len(markers)} markers"
        cv2.putText(result_image, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        cv2.putText(result_image, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 显示结果
        cv2.imshow('Parameter Test Result', result_image)
        print("按任意键继续...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def batch_test(self):
        """批量测试当前参数"""
        image_files = self.get_image_files()
        if not image_files:
            print("未找到图像文件")
            return
        
        print(f"\n批量测试 {len(image_files)} 张图像...")
        
        results = []
        for i, image_path in enumerate(image_files):
            print(f"处理 {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
            
            image = cv2.imread(image_path)
            if image is None:
                continue
            
            # 去畸变
            if self.detector.camera_matrix is not None:
                image = self.detector.undistort_image(image)
            
            # 检测标记点
            markers = self.detect_with_params(image, self.current_params)
            results.append(len(markers))
        
        # 统计结果
        if results:
            print(f"\n📊 批量测试结果:")
            print(f"平均标记点数: {np.mean(results):.1f}")
            print(f"标记点数量范围: {min(results)} - {max(results)}")
            
            # 分布统计
            print(f"\n标记点数量分布:")
            for count in range(max(results) + 1):
                num_images = sum(1 for r in results if r == count)
                if num_images > 0:
                    print(f"  {count} 个标记点: {num_images} 张图像")
    
    def adjust_parameters(self):
        """调整参数"""
        print(f"\n当前参数:")
        for key, value in self.current_params.items():
            print(f"  {key}: {value}")
        
        print(f"\n选择要调整的参数:")
        print("1. min_area (最小面积)")
        print("2. max_area (最大面积)")
        print("3. min_circularity (最小圆度)")
        print("4. min_convexity (最小凸度)")
        print("5. min_inertia_ratio (最小惯性率)")
        print("6. 返回主菜单")
        
        try:
            choice = int(input("请选择 (1-6): "))
            
            if choice == 1:
                new_value = float(input(f"输入新的最小面积 (当前: {self.current_params['min_area']}): "))
                self.current_params['min_area'] = new_value
            elif choice == 2:
                new_value = float(input(f"输入新的最大面积 (当前: {self.current_params['max_area']}): "))
                self.current_params['max_area'] = new_value
            elif choice == 3:
                new_value = float(input(f"输入新的最小圆度 (当前: {self.current_params['min_circularity']}): "))
                self.current_params['min_circularity'] = max(0.0, min(1.0, new_value))
            elif choice == 4:
                new_value = float(input(f"输入新的最小凸度 (当前: {self.current_params['min_convexity']}): "))
                self.current_params['min_convexity'] = max(0.0, min(1.0, new_value))
            elif choice == 5:
                new_value = float(input(f"输入新的最小惯性率 (当前: {self.current_params['min_inertia_ratio']}): "))
                self.current_params['min_inertia_ratio'] = max(0.0, min(1.0, new_value))
            elif choice == 6:
                return
            
            print("参数已更新!")
            
        except ValueError:
            print("无效输入")

def main():
    """主函数"""
    print("🔧 简化参数优化工具")
    print("=" * 40)
    
    optimizer = SimpleOptimizer()
    
    # 检查图像文件
    image_files = optimizer.get_image_files()
    if not image_files:
        print("❌ 未找到图像文件，请检查 tip_images 文件夹")
        return
    
    print(f"✅ 找到 {len(image_files)} 张图像")
    
    while True:
        print(f"\n选择功能:")
        print("1. 🖼️  测试单张图像")
        print("2. 📊 批量测试")
        print("3. ⚙️  调整参数")
        print("4. 📋 显示当前参数")
        print("5. ❌ 退出")
        
        choice = input("请输入选择 (1-5): ").strip()
        
        if choice == '1':
            # 选择图像
            print("\n可用图像:")
            for i, img_path in enumerate(image_files[:10]):
                print(f"{i+1}. {os.path.basename(img_path)}")
            
            try:
                img_idx = int(input(f"请选择图像 (1-{min(10, len(image_files))}): ")) - 1
                if 0 <= img_idx < len(image_files):
                    optimizer.test_single_image(image_files[img_idx])
                else:
                    print("无效选择")
            except ValueError:
                print("无效输入")
        
        elif choice == '2':
            optimizer.batch_test()
        
        elif choice == '3':
            optimizer.adjust_parameters()
        
        elif choice == '4':
            print(f"\n当前参数:")
            for key, value in optimizer.current_params.items():
                print(f"  {key}: {value}")
        
        elif choice == '5':
            print("\n👋 再见!")
            break
        
        else:
            print("无效选择")

if __name__ == "__main__":
    main()
