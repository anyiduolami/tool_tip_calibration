#!/usr/bin/env python3
"""
专门针对金属器械的标记点检测工具

针对金属器械上的小孔/凹陷标记点进行优化检测

使用方法:
python specialized_marker_detector.py

作者: AI Assistant
日期: 2025-01-18
"""

import cv2
import numpy as np
import os
import glob

class SpecializedMarkerDetector:
    def __init__(self):
        """初始化专用检测器"""
        # 多种检测策略的参数
        self.strategies = {
            'dark_holes': {
                'name': '深色孔洞检测',
                'use_gaussian_blur': True,
                'blur_kernel': (5, 5),
                'threshold_method': 'adaptive',
                'adaptive_block_size': 21,
                'adaptive_c': 8,
                'min_area': 20,
                'max_area': 300,
                'min_circularity': 0.4,
                'use_morphology': True,
                'morph_kernel': (3, 3),
                'morph_iterations': 1
            },
            'edge_circles': {
                'name': '边缘圆形检测',
                'use_gaussian_blur': True,
                'blur_kernel': (3, 3),
                'canny_low': 50,
                'canny_high': 150,
                'hough_dp': 1,
                'hough_min_dist': 30,
                'hough_param1': 50,
                'hough_param2': 30,
                'min_radius': 3,
                'max_radius': 15
            },
            'template_matching': {
                'name': '模板匹配检测',
                'template_sizes': [8, 10, 12, 15],
                'match_threshold': 0.6,
                'use_normalized': True
            },
            'blob_detection': {
                'name': 'Blob检测',
                'min_threshold': 10,
                'max_threshold': 200,
                'threshold_step': 10,
                'min_area': 15,
                'max_area': 300,
                'min_circularity': 0.3,
                'min_convexity': 0.5,
                'min_inertia_ratio': 0.2
            }
        }
        
        self.current_strategy = 'dark_holes'
    
    def preprocess_image(self, image):
        """图像预处理"""
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 直方图均衡化，增强对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        return gray, enhanced
    
    def detect_dark_holes(self, image, params):
        """检测深色孔洞"""
        gray, enhanced = self.preprocess_image(image)
        
        # 高斯模糊
        if params['use_gaussian_blur']:
            blurred = cv2.GaussianBlur(enhanced, params['blur_kernel'], 0)
        else:
            blurred = enhanced
        
        # 自适应阈值
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
            params['adaptive_block_size'], params['adaptive_c']
        )
        
        # 形态学操作
        if params['use_morphology']:
            kernel = np.ones(params['morph_kernel'], np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, 
                                    iterations=params['morph_iterations'])
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, 
                                    iterations=params['morph_iterations'])
        
        # 查找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        markers = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if params['min_area'] < area < params['max_area']:
                # 计算圆度
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > params['min_circularity']:
                        # 计算质心
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            markers.append([cx, cy, area, circularity])
        
        return np.array(markers), thresh
    
    def detect_edge_circles(self, image, params):
        """使用霍夫圆变换检测圆形"""
        gray, enhanced = self.preprocess_image(image)
        
        # 高斯模糊
        if params['use_gaussian_blur']:
            blurred = cv2.GaussianBlur(enhanced, params['blur_kernel'], 0)
        else:
            blurred = enhanced
        
        # Canny边缘检测
        edges = cv2.Canny(blurred, params['canny_low'], params['canny_high'])
        
        # 霍夫圆变换
        circles = cv2.HoughCircles(
            edges, cv2.HOUGH_GRADIENT, params['hough_dp'], params['hough_min_dist'],
            param1=params['hough_param1'], param2=params['hough_param2'],
            minRadius=params['min_radius'], maxRadius=params['max_radius']
        )
        
        markers = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                # 计算圆的面积作为特征
                area = np.pi * r * r
                markers.append([x, y, area, 1.0])  # 圆度设为1.0
        
        return np.array(markers), edges
    
    def detect_blob_features(self, image, params):
        """使用Blob检测器"""
        gray, enhanced = self.preprocess_image(image)
        
        # 设置Blob检测器参数
        detector_params = cv2.SimpleBlobDetector_Params()
        
        # 阈值参数
        detector_params.minThreshold = params['min_threshold']
        detector_params.maxThreshold = params['max_threshold']
        detector_params.thresholdStep = params['threshold_step']
        
        # 面积筛选
        detector_params.filterByArea = True
        detector_params.minArea = params['min_area']
        detector_params.maxArea = params['max_area']
        
        # 圆度筛选
        detector_params.filterByCircularity = True
        detector_params.minCircularity = params['min_circularity']
        
        # 凸性筛选
        detector_params.filterByConvexity = True
        detector_params.minConvexity = params['min_convexity']
        
        # 惯性比筛选
        detector_params.filterByInertia = True
        detector_params.minInertiaRatio = params['min_inertia_ratio']
        
        # 创建检测器
        detector = cv2.SimpleBlobDetector_create(detector_params)
        
        # 检测关键点
        keypoints = detector.detect(enhanced)
        
        markers = []
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            size = kp.size
            area = np.pi * (size/2) * (size/2)
            markers.append([x, y, area, 1.0])
        
        # 创建可视化图像
        debug_img = cv2.drawKeypoints(enhanced, keypoints, np.array([]), (0,0,255), 
                                    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        return np.array(markers), debug_img
    
    def create_circular_template(self, radius):
        """创建圆形模板"""
        size = radius * 2 + 1
        template = np.zeros((size, size), dtype=np.uint8)
        cv2.circle(template, (radius, radius), radius, 255, -1)
        return template
    
    def detect_template_matching(self, image, params):
        """模板匹配检测"""
        gray, enhanced = self.preprocess_image(image)
        
        all_matches = []
        
        for template_size in params['template_sizes']:
            # 创建圆形模板
            template = self.create_circular_template(template_size)
            
            # 模板匹配
            if params['use_normalized']:
                result = cv2.matchTemplate(enhanced, template, cv2.TM_CCOEFF_NORMED)
            else:
                result = cv2.matchTemplate(enhanced, template, cv2.TM_CCOEFF)
            
            # 查找匹配位置
            locations = np.where(result >= params['match_threshold'])
            
            for pt in zip(*locations[::-1]):
                x, y = pt[0] + template_size, pt[1] + template_size
                confidence = result[pt[1], pt[0]]
                area = np.pi * template_size * template_size
                all_matches.append([x, y, area, confidence])
        
        # 去除重复检测
        if len(all_matches) > 0:
            markers = self.remove_duplicate_detections(np.array(all_matches))
        else:
            markers = np.array([])
        
        return markers, result if len(params['template_sizes']) > 0 else enhanced
    
    def remove_duplicate_detections(self, markers, min_distance=20):
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
                if distance < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                filtered.append(marker)
        
        return np.array(filtered)
    
    def detect_markers(self, image, strategy=None, debug=False):
        """使用指定策略检测标记点"""
        if strategy is None:
            strategy = self.current_strategy
        
        params = self.strategies[strategy]
        
        if strategy == 'dark_holes':
            markers, debug_img = self.detect_dark_holes(image, params)
        elif strategy == 'edge_circles':
            markers, debug_img = self.detect_edge_circles(image, params)
        elif strategy == 'blob_detection':
            markers, debug_img = self.detect_blob_features(image, params)
        elif strategy == 'template_matching':
            markers, debug_img = self.detect_template_matching(image, params)
        else:
            raise ValueError(f"未知的检测策略: {strategy}")
        
        if debug:
            print(f"策略 '{params['name']}' 检测到 {len(markers)} 个标记点")
            
            # 显示结果
            result_img = image.copy()
            for i, marker in enumerate(markers):
                x, y = int(marker[0]), int(marker[1])
                cv2.circle(result_img, (x, y), 8, (0, 0, 255), 2)
                cv2.putText(result_img, f"{i+1}", (x+10, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            cv2.imshow(f'原图 - {params["name"]}', cv2.resize(result_img, (400, 300)))
            cv2.imshow(f'处理结果 - {params["name"]}', cv2.resize(debug_img, (400, 300)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return markers[:, :2] if len(markers) > 0 else np.array([])  # 只返回坐标
    
    def compare_all_strategies(self, image_path):
        """比较所有检测策略"""
        print(f"\n=== 比较所有检测策略 ===")
        print(f"图像: {os.path.basename(image_path)}")
        
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图像: {image_path}")
            return
        
        results = {}
        
        for strategy_name in self.strategies.keys():
            print(f"\n测试策略: {self.strategies[strategy_name]['name']}")
            markers = self.detect_markers(img, strategy_name, debug=False)
            results[strategy_name] = markers
            print(f"检测到 {len(markers)} 个标记点")
        
        # 创建对比图像
        self.create_comparison_visualization(img, results)
        
        return results
    
    def create_comparison_visualization(self, image, results):
        """创建对比可视化"""
        num_strategies = len(results)
        cols = 2
        rows = (num_strategies + 1) // 2
        
        fig_width = image.shape[1] * cols // 2
        fig_height = image.shape[0] * rows // 2
        
        comparison_img = np.ones((fig_height, fig_width, 3), dtype=np.uint8) * 255
        
        strategy_names = list(results.keys())
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
        
        for i, strategy in enumerate(strategy_names):
            row = i // cols
            col = i % cols
            
            # 计算位置
            start_y = row * (image.shape[0] // 2)
            end_y = start_y + (image.shape[0] // 2)
            start_x = col * (image.shape[1] // 2)
            end_x = start_x + (image.shape[1] // 2)
            
            # 调整图像大小
            resized_img = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
            
            # 绘制标记点
            markers = results[strategy]
            for j, marker in enumerate(markers):
                x, y = int(marker[0] // 2), int(marker[1] // 2)
                cv2.circle(resized_img, (x, y), 4, colors[i % len(colors)], 2)
                cv2.putText(resized_img, f"{j+1}", (x+6, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[i % len(colors)], 1)
            
            # 添加标题
            strategy_name = self.strategies[strategy]['name']
            cv2.putText(resized_img, f"{strategy_name} ({len(markers)})", 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(resized_img, f"{strategy_name} ({len(markers)})", 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i % len(colors)], 1)
            
            # 放置到对比图像中
            comparison_img[start_y:end_y, start_x:end_x] = resized_img
        
        cv2.imshow('检测策略对比', comparison_img)
        print("\n按任意键继续...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    """主函数"""
    print("=== 专用标记点检测工具 ===")
    
    detector = SpecializedMarkerDetector()
    
    # 获取图像文件
    image_files = sorted(glob.glob("tip_images/*.jpg"))
    if not image_files:
        image_files = sorted(glob.glob("tip_images/*.png"))
    
    if not image_files:
        print("未找到图像文件，请检查 tip_images 文件夹")
        return
    
    print(f"找到 {len(image_files)} 张图像")
    print("\n选择测试模式:")
    print("1. 单张图像 - 比较所有策略")
    print("2. 单张图像 - 测试特定策略")
    print("3. 批量测试最佳策略")
    
    choice = input("请输入选择 (1, 2 或 3): ").strip()
    
    if choice == '1':
        # 选择图像
        print("\n可用图像:")
        for i, img_path in enumerate(image_files[:10]):
            print(f"{i+1}. {os.path.basename(img_path)}")
        
        try:
            img_idx = int(input(f"请选择图像 (1-{min(10, len(image_files))}): ")) - 1
            if 0 <= img_idx < len(image_files):
                detector.compare_all_strategies(image_files[img_idx])
            else:
                print("无效选择")
        except ValueError:
            print("无效输入")
    
    elif choice == '2':
        # 选择策略和图像
        print("\n可用策略:")
        strategies = list(detector.strategies.keys())
        for i, strategy in enumerate(strategies):
            print(f"{i+1}. {detector.strategies[strategy]['name']}")
        
        try:
            strategy_idx = int(input(f"请选择策略 (1-{len(strategies)}): ")) - 1
            if 0 <= strategy_idx < len(strategies):
                selected_strategy = strategies[strategy_idx]
                
                print("\n可用图像:")
                for i, img_path in enumerate(image_files[:10]):
                    print(f"{i+1}. {os.path.basename(img_path)}")
                
                img_idx = int(input(f"请选择图像 (1-{min(10, len(image_files))}): ")) - 1
                if 0 <= img_idx < len(image_files):
                    img = cv2.imread(image_files[img_idx])
                    detector.detect_markers(img, selected_strategy, debug=True)
                else:
                    print("无效选择")
            else:
                print("无效选择")
        except ValueError:
            print("无效输入")

if __name__ == "__main__":
    main()
