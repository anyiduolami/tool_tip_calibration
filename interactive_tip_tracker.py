"""
Interactive Tool Tip Tracker
实现摄像头拍照、图片选择、点检测和尖端位置求解的交互式程序
"""

import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import time

class InteractiveTipTracker:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Interactive Tool Tip Tracker")
        self.root.geometry("1200x800")
        
        # 初始化变量
        self.camera = None
        self.is_capturing = False
        self.captured_images = []
        self.detected_points = None
        self.tip_position = None
        self.current_frame = None

        # 两点测距相关变量
        self.start_image = None
        self.end_image = None
        self.start_points = []
        self.end_points = []
        self.start_tip_position = None
        self.end_tip_position = None
        self.max_capture_count = 2  # 最多拍摄2张照片
        self.capture_count = 0  # 当前拍摄计数
        self.show_dual_images = False  # 是否显示并排图像
        
        # 加载标定数据
        self.load_calibration_data()
        
        # 初始化检测参数（可以自定义调整）
        self.detection_params = {
            # Blob检测参数
            'min_threshold': 50,
            'max_threshold': 200,
            'threshold_step': 10,
            'min_area': 100,
            'max_area': 5000,
            'min_circularity': 0.7,
            'max_circularity': 1.0,
            'min_convexity': 0.4,
            'max_convexity': 1.0,
            'min_inertia_ratio': 0.4,
            'max_inertia_ratio': 1.0,

            # 图像预处理参数
            'gaussian_blur_size': 5,
            'morphology_kernel_size': 3,
            'clahe_clip_limit': 2.0,
            'clahe_tile_size': 8,

            # 颜色检测参数
            'blob_color': 0,  # 0=暗色blob, 255=亮色blob
        }
        
        # 创建UI界面
        self.create_ui()
        
        # 创建images文件夹
        os.makedirs('images', exist_ok=True)
        
    def load_calibration_data(self):
        """加载相机标定和工具标定数据"""
        try:
            # 加载相机标定数据
            calib_data = np.load('calibration/camera_calibration.npz')
            self.camera_matrix = calib_data['camera_matrix']
            self.dist_coeffs = calib_data['dist_coeffs']
            print("✅ 相机标定数据加载成功")
            
            # 加载工具标定结果
            tool_data = np.load('tool_tip_calibration_results.npz')
            self.tip_world = tool_data['tip_world']
            self.tip_tool = tool_data['tip_tool']
            print("✅ 工具标定数据加载成功")
            print(f"   Tip_w: {self.tip_world}")
            print(f"   Tip_t: {self.tip_tool}")
            
        except Exception as e:
            print(f"❌ 加载标定数据失败: {e}")
            messagebox.showerror("错误", f"加载标定数据失败: {e}")
            
    def create_ui(self):
        """创建用户界面"""
        # 设置更大的窗口尺寸
        self.root.geometry("1400x900")

        # 主框架
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置网格权重 - 调整比例让图像显示区域更大
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=0)  # 左侧控制面板固定宽度
        main_frame.columnconfigure(1, weight=1)  # 右侧图像显示区域自适应
        main_frame.rowconfigure(0, weight=1)

        # 左侧控制面板 - 调整宽度和样式
        control_frame = ttk.LabelFrame(main_frame, text="🎛️ 控制面板", padding="15")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 15))
        control_frame.configure(width=350)  # 固定宽度
        
        # 摄像头控制 - 美化样式
        camera_frame = ttk.LabelFrame(control_frame, text="📷 摄像头控制", padding="10")
        camera_frame.pack(fill=tk.X, pady=(0, 15))

        # 摄像头信息显示
        camera_info_frame = ttk.Frame(camera_frame)
        camera_info_frame.pack(fill=tk.X, pady=(0, 8))

        info_label = ttk.Label(camera_info_frame, text="📹 使用外接摄像头 (索引: 1)",
                              font=("Arial", 9), foreground="gray")
        info_label.pack()

        # 按钮样式优化
        self.start_camera_btn = ttk.Button(camera_frame, text="🟢 启动摄像头", command=self.start_camera)
        self.start_camera_btn.pack(fill=tk.X, pady=3, ipady=5)

        self.stop_camera_btn = ttk.Button(camera_frame, text="🔴 停止摄像头", command=self.stop_camera, state=tk.DISABLED)
        self.stop_camera_btn.pack(fill=tk.X, pady=3, ipady=5)

        # 拍照按钮（备选方案）
        self.capture_btn = ttk.Button(camera_frame, text="📸 拍照 (空格键)", command=self.manual_capture, state=tk.DISABLED)
        self.capture_btn.pack(fill=tk.X, pady=3, ipady=5)



        # 操作说明 - 简化并美化
        instruction_frame = ttk.LabelFrame(control_frame, text="📋 操作说明", padding="10")
        instruction_frame.pack(fill=tk.X, pady=(0, 15))

        instructions = [
            "1️⃣ 启动摄像头开始拍摄",
            "2️⃣ 空格键拍照(2张)",
            "3️⃣ Enter键显示两张图片",
            "4️⃣ 检测标记点并计算距离"
        ]

        for instruction in instructions:
            label = ttk.Label(instruction_frame, text=instruction, font=("Arial", 9))
            label.pack(anchor=tk.W, pady=1)

        # 图片处理 - 美化按钮
        image_frame = ttk.LabelFrame(control_frame, text="🖼️ 图片处理", padding="10")
        image_frame.pack(fill=tk.X, pady=(0, 15))

        self.detect_points_btn = ttk.Button(image_frame, text="🎯 检测标记点", command=self.detect_points, state=tk.DISABLED)
        self.detect_points_btn.pack(fill=tk.X, pady=3, ipady=5)

        self.calculate_tip_btn = ttk.Button(image_frame, text="� 计算两点距离", command=self.calculate_tip_position, state=tk.DISABLED)
        self.calculate_tip_btn.pack(fill=tk.X, pady=3, ipady=5)

        # 结果显示 - 扩大显示区域
        result_frame = ttk.LabelFrame(control_frame, text="📊 结果显示", padding="10")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))

        # 创建滚动文本框
        text_frame = ttk.Frame(result_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)

        # 文本框和滚动条
        self.result_text = tk.Text(text_frame, height=20, width=40,
                                  font=("Consolas", 9), wrap=tk.WORD,
                                  bg="#f8f9fa", fg="#212529",
                                  selectbackground="#007bff",
                                  relief=tk.FLAT, borderwidth=1)

        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)

        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 添加清空按钮
        clear_btn = ttk.Button(result_frame, text="🗑️ 清空结果", command=self.clear_results)
        clear_btn.pack(fill=tk.X, pady=(5, 0))
        
        # 右侧显示区域 - 优化布局
        display_frame = ttk.LabelFrame(main_frame, text="🖼️ 图像显示", padding="15")
        display_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        display_frame.columnconfigure(0, weight=1)
        display_frame.rowconfigure(0, weight=1)

        # 图像显示标签 - 添加边框和背景
        self.image_label = ttk.Label(display_frame, text="📷 图像将在这里显示\n\n请选择图片或启动摄像头开始",
                                    font=("Arial", 12), foreground="gray",
                                    anchor=tk.CENTER, justify=tk.CENTER)
        self.image_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)

        # 状态栏 - 美化样式
        self.status_var = tk.StringVar()
        self.status_var.set("🟢 系统就绪")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var,
                              relief=tk.SUNKEN, font=("Arial", 9),
                              background="#e9ecef", foreground="#495057")
        status_bar.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(15, 0))

        # 绑定键盘事件
        self.root.bind('<KeyPress>', self.on_key_press)
        self.root.focus_set()  # 确保窗口可以接收键盘事件

    def clear_results(self):
        """清空结果显示"""
        self.result_text.delete(1.0, tk.END)
        self.status_var.set("🗑️ 结果已清空")
        




    def on_key_press(self, event):
        """处理键盘按键事件"""
        if self.is_capturing and self.current_frame is not None:
            if event.keysym == 'space':  # 空格键拍照
                self.capture_image(self.current_frame)
            elif event.keysym == 'Return':  # Enter键
                if self.capture_count == 2:
                    # 拍摄完两张照片后，按Enter显示并排图像
                    self.show_captured_images()
                else:
                    # 否则停止摄像头
                    self.stop_camera()

    def show_captured_images(self):
        """显示拍摄的两张图片并停止摄像头"""
        if self.capture_count == 2 and self.start_image and self.end_image:
            # 设置标志阻止摄像头覆盖显示
            self.show_dual_images = True

            # 停止摄像头
            self.stop_camera()

            # 启用检测按钮
            self.detect_points_btn.config(state=tk.NORMAL)

            # 显示并排图像
            self.display_dual_images()

            self.update_result_text("📷 已显示拍摄的两张图片\n")
            self.status_var.set("✅ 图片已显示，可以开始检测标记点")
        else:
            self.update_result_text("⚠️ 请先拍摄两张照片\n")

    def manual_capture(self):
        """手动拍照（按钮触发）"""
        if self.is_capturing and self.current_frame is not None:
            self.capture_image(self.current_frame)

    def start_camera(self):
        """启动外接摄像头（索引1）"""
        try:
            # 直接使用索引1作为外接摄像头
            camera_index = 0
            self.status_var.set("正在启动外接摄像头...")

            self.camera = cv2.VideoCapture(camera_index)
            if not self.camera.isOpened():
                raise Exception(f"无法打开外接摄像头 (索引: {camera_index})，请确保外接摄像头已连接")

            # 设置摄像头参数以获得更好的图像质量
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.camera.set(cv2.CAP_PROP_FPS, 30)

            # 获取实际设置的参数
            width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.camera.get(cv2.CAP_PROP_FPS))

            # 重置拍摄计数和图像
            self.capture_count = 0
            self.captured_images = []
            self.start_image = None
            self.end_image = None
            self.start_points = []
            self.end_points = []
            self.start_tip_position = None
            self.end_tip_position = None
            self.show_dual_images = False  # 重置显示标志

            self.is_capturing = True
            self.start_camera_btn.config(state=tk.DISABLED)
            self.stop_camera_btn.config(state=tk.NORMAL)
            self.capture_btn.config(state=tk.NORMAL)
            self.detect_points_btn.config(state=tk.DISABLED)

            # 启动摄像头线程
            self.camera_thread = threading.Thread(target=self.camera_loop)
            self.camera_thread.daemon = True
            self.camera_thread.start()

            self.status_var.set(f"外接摄像头已启动 (索引:{camera_index}, {width}x{height}@{fps}fps) - 按空格键拍照，按Enter键结束")
            self.update_result_text(f"✅ 成功启动外接摄像头 (索引: {camera_index})\n")
            self.update_result_text(f"   分辨率: {width}x{height}, 帧率: {fps}fps\n")

        except Exception as e:
            messagebox.showerror("错误", f"启动外接摄像头失败: {e}")
            self.status_var.set("外接摄像头启动失败")
            
    def stop_camera(self):
        """停止摄像头"""
        self.is_capturing = False
        if self.camera:
            self.camera.release()
            self.camera = None

        self.start_camera_btn.config(state=tk.NORMAL)
        self.stop_camera_btn.config(state=tk.DISABLED)
        self.capture_btn.config(state=tk.DISABLED)
        self.status_var.set(f"摄像头已停止 - 共拍摄 {len(self.captured_images)} 张图片")
        
    def camera_loop(self):
        """摄像头循环"""
        while self.is_capturing and self.camera:
            ret, frame = self.camera.read()
            if ret:
                # 存储当前帧用于拍照
                self.current_frame = frame.copy()

                # 只有在不需要显示并排图像时才显示摄像头画面
                if not self.show_dual_images:
                    # 在画面上添加拍照计数字幕
                    frame_with_overlay = self.add_capture_overlay(frame)
                    self.display_image(frame_with_overlay)

                # 短暂延时
                time.sleep(0.033)  # 约30fps
            else:
                break

    def add_capture_overlay(self, frame):
        """在摄像头画面上添加拍照计数字幕"""
        overlay_frame = frame.copy()

        # 获取画面尺寸
        _, width = overlay_frame.shape[:2]

        # 在右上角显示拍照计数
        count_text = f"{self.capture_count}/2"
        if self.capture_count < 2:
            color = (0, 255, 0)  # 绿色
        else:
            color = (255, 0, 0)  # 蓝色

        cv2.putText(overlay_frame, count_text, (width - 100, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
        cv2.putText(overlay_frame, count_text, (width - 100, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        # 只有在拍摄完两张照片后，在左上角显示Enter提示
        if self.capture_count == 2:
            enter_text = "Press ENTER to view images"
            cv2.putText(overlay_frame, enter_text, (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
            cv2.putText(overlay_frame, enter_text, (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        return overlay_frame
        
    def capture_image(self, frame):
        """拍摄图片 - 最多拍摄2张照片"""
        if self.capture_count >= self.max_capture_count:
            self.update_result_text("已达到最大拍摄数量(2张)，请先处理当前图片\n")
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"images/captured_{timestamp}.jpg"

        cv2.imwrite(filename, frame)
        self.captured_images.append(filename)
        self.capture_count += 1

        # 根据拍摄顺序保存为起始点或终点图像
        if self.capture_count == 1:
            self.start_image = filename
            self.update_result_text(f"📍 起始点图像: {filename}\n")
            self.status_var.set("📷 已拍摄起始点，请拍摄终点")
        elif self.capture_count == 2:
            self.end_image = filename
            self.update_result_text(f"🎯 终点图像: {filename}\n")
            self.status_var.set("✅ 已拍摄两张图片，按Enter键查看图像")

            # 不自动停止摄像头，继续显示带字幕的画面
            # 用户需要按Enter键来查看并排图像

        self.update_result_text(f"已拍摄 {self.capture_count}/{self.max_capture_count} 张图片\n")
        
    def display_image(self, cv_image):
        """在UI中显示图像"""
        # 转换颜色空间
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # 调整图像大小以适应显示区域
        height, width = rgb_image.shape[:2]
        max_width, max_height = 800, 600
        
        if width > max_width or height > max_height:
            scale = min(max_width/width, max_height/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            rgb_image = cv2.resize(rgb_image, (new_width, new_height))
        
        # 转换为PIL图像
        pil_image = Image.fromarray(rgb_image)
        photo = ImageTk.PhotoImage(pil_image)
        
        # 更新显示
        self.image_label.configure(image=photo)
        self.image_label.image = photo  # 保持引用

    def display_dual_images(self):
        """显示起始点和终点两张图片并排"""
        if not self.start_image or not self.end_image:
            return

        try:
            # 读取两张图片
            start_img = cv2.imread(self.start_image)
            end_img = cv2.imread(self.end_image)

            if start_img is None or end_img is None:
                self.update_result_text("❌ 无法读取图片文件\n")
                return

            # 转换颜色空间
            start_rgb = cv2.cvtColor(start_img, cv2.COLOR_BGR2RGB)
            end_rgb = cv2.cvtColor(end_img, cv2.COLOR_BGR2RGB)

            # 调整图片大小使其一致 - 增大显示尺寸
            target_height = 400  # 增加高度
            start_h, start_w = start_rgb.shape[:2]
            end_h, end_w = end_rgb.shape[:2]

            # 按比例缩放
            start_scale = target_height / start_h
            end_scale = target_height / end_h

            start_new_w = int(start_w * start_scale)
            end_new_w = int(end_w * end_scale)

            start_resized = cv2.resize(start_rgb, (start_new_w, target_height))
            end_resized = cv2.resize(end_rgb, (end_new_w, target_height))

            # 创建并排图像
            gap = 30  # 增加间隔
            total_width = start_new_w + end_new_w + gap
            combined_img = np.ones((target_height, total_width, 3), dtype=np.uint8) * 240  # 浅灰色背景

            # 放置图片
            combined_img[:, :start_new_w] = start_resized
            combined_img[:, start_new_w+gap:start_new_w+gap+end_new_w] = end_resized

            # 添加标签 - 更大更清晰的文字
            cv2.putText(combined_img, "START POINT", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.putText(combined_img, "END POINT", (start_new_w+gap+10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            # 添加分隔线
            line_x = start_new_w + gap // 2
            cv2.line(combined_img, (line_x, 0), (line_x, target_height), (100, 100, 100), 2)

            # 转换为PIL图像并显示
            pil_image = Image.fromarray(combined_img)
            photo = ImageTk.PhotoImage(pil_image)

            self.image_label.configure(image=photo)
            self.image_label.image = photo

            self.update_result_text("📷 已显示起始点和终点图像\n")

        except Exception as e:
            self.update_result_text(f"❌ 显示图片失败: {e}\n")
        

    def preprocess_image(self, image):
        """图像预处理"""
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 应用CLAHE增强对比度
        clahe = cv2.createCLAHE(
            clipLimit=self.detection_params['clahe_clip_limit'],
            tileGridSize=(self.detection_params['clahe_tile_size'], self.detection_params['clahe_tile_size'])
        )
        enhanced = clahe.apply(gray)

        # 高斯模糊
        if self.detection_params['gaussian_blur_size'] > 0:
            blurred = cv2.GaussianBlur(enhanced,
                                     (self.detection_params['gaussian_blur_size'],
                                      self.detection_params['gaussian_blur_size']), 0)
        else:
            blurred = enhanced

        # 形态学操作
        kernel_size = self.detection_params['morphology_kernel_size']
        if kernel_size > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            # 开运算去除噪声
            processed = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
            # 闭运算填充空洞
            processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
        else:
            processed = blurred

        return processed

    def detect_blobs(self, image):
        """使用SimpleBlobDetector检测blob，并计算每个blob的圆度"""
        # 设置blob检测参数
        params = cv2.SimpleBlobDetector_Params()

        # 阈值参数
        params.minThreshold = self.detection_params['min_threshold']
        params.maxThreshold = self.detection_params['max_threshold']
        params.thresholdStep = self.detection_params['threshold_step']

        # 面积过滤
        params.filterByArea = True
        params.minArea = self.detection_params['min_area']
        params.maxArea = self.detection_params['max_area']

        # 圆形度过滤
        params.filterByCircularity = True
        params.minCircularity = self.detection_params['min_circularity']
        params.maxCircularity = self.detection_params['max_circularity']

        # 凸性过滤
        params.filterByConvexity = True
        params.minConvexity = self.detection_params['min_convexity']
        params.maxConvexity = self.detection_params['max_convexity']

        # 惯性比过滤
        params.filterByInertia = True
        params.minInertiaRatio = self.detection_params['min_inertia_ratio']
        params.maxInertiaRatio = self.detection_params['max_inertia_ratio']

        # 颜色过滤
        params.filterByColor = True
        params.blobColor = self.detection_params['blob_color']

        # 创建检测器
        detector = cv2.SimpleBlobDetector_create(params)

        # 检测关键点
        keypoints = detector.detect(image)

        # 计算每个blob的详细特征，包括圆度
        points_with_features = []

        for kp in keypoints:
            # 获取blob的基本信息
            x, y = kp.pt
            size = kp.size

            # 使用亚像素精度优化点位置
            refined_point = self.refine_point_subpixel(image, x, y, size)

            # 计算圆度：使用轮廓分析
            circularity = self.calculate_blob_circularity(image, refined_point[0], refined_point[1], size)

            # 计算像素面积：使用轮廓分析
            pixel_area = self.calculate_blob_pixel_area(image, refined_point[0], refined_point[1], size)

            points_with_features.append({
                'point': refined_point,
                'size': size,
                'circularity': circularity,
                'pixel_area': pixel_area,
                'response': kp.response  # blob检测器的响应强度
            })

        return points_with_features

    def refine_point_subpixel(self, image, x, y, size):
        """
        使用亚像素精度优化点位置

        Args:
            image: 输入图像
            x, y: 初始点坐标
            size: blob大小

        Returns:
            list: 优化后的点坐标 [x, y]
        """
        try:
            # 转换为灰度图像
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # 定义搜索窗口大小
            win_size = max(5, int(size / 4))

            # 使用cornerSubPix进行亚像素精度优化
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

            # 初始点坐标
            corners = np.array([[x, y]], dtype=np.float32)

            # 亚像素精度优化
            refined_corners = cv2.cornerSubPix(
                gray,
                corners,
                (win_size, win_size),
                (-1, -1),
                criteria
            )

            refined_x, refined_y = refined_corners[0]

            # 检查优化结果是否合理（不应该偏移太远）
            max_offset = size / 2
            if abs(refined_x - x) > max_offset or abs(refined_y - y) > max_offset:
                # 如果偏移过大，使用原始坐标
                return [x, y]

            return [float(refined_x), float(refined_y)]

        except Exception:
            # 如果优化失败，返回原始坐标
            return [x, y]

    def calculate_blob_circularity(self, image, x, y, size):
        """计算blob的圆度"""
        try:
            # 创建一个围绕blob的小区域
            radius = int(size / 2) + 5
            x_int, y_int = int(x), int(y)

            # 确保区域在图像范围内 

            x1 = max(0, x_int - radius)
            y1 = max(0, y_int - radius)
            x2 = min(image.shape[1], x_int + radius)
            y2 = min(image.shape[0], y_int + radius)

            if x2 <= x1 or y2 <= y1:
                return 0.0

            # 提取ROI
            roi = image[y1:y2, x1:x2]

            # 二值化
            if self.detection_params['blob_color'] == 0:  # 暗色blob
                _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            else:  # 亮色blob
                _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # 查找轮廓
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return 0.0

            # 找到最大的轮廓（应该是我们的blob）
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            if area < 10:  # 面积太小
                return 0.0

            # 计算周长
            perimeter = cv2.arcLength(largest_contour, True)

            if perimeter == 0:
                return 0.0

            # 计算圆度：4π*面积/周长²
            circularity = 4 * np.pi * area / (perimeter * perimeter)

            # 圆度值应该在0-1之间，完美的圆为1
            return min(1.0, circularity)

        except Exception:
            # 如果计算失败，返回默认值
            return 0.5

    def calculate_blob_pixel_area(self, image, x, y, size):
        """计算blob的像素面积"""
        try:
            # 创建一个围绕blob的小区域
            radius = int(size / 2) + 5
            x_int, y_int = int(x), int(y)

            # 确保区域在图像范围内
            x1 = max(0, x_int - radius)
            y1 = max(0, y_int - radius)
            x2 = min(image.shape[1], x_int + radius)
            y2 = min(image.shape[0], y_int + radius)

            if x2 <= x1 or y2 <= y1:
                return 0

            # 提取ROI
            roi = image[y1:y2, x1:x2]

            # 二值化
            if self.detection_params['blob_color'] == 0:  # 暗色blob
                _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            else:  # 亮色blob
                _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # 查找轮廓
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return 0

            # 找到最大的轮廓（应该是我们的blob）
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            return int(area)

        except Exception:
            # 如果计算失败，返回默认值
            return 0

    def calculate_area_similarity_score(self, areas):
        """
        计算面积相似性得分

        Args:
            areas: 面积列表

        Returns:
            float: 相似性得分，值越高表示面积越相似
        """
        if len(areas) < 2:
            return 1.0

        areas = np.array(areas)
        mean_area = np.mean(areas)

        if mean_area == 0:
            return 0.0

        # 计算变异系数（标准差/均值），值越小表示越相似
        cv = np.std(areas) / mean_area

        # 转换为相似性得分（0-1之间，1表示完全相似）
        similarity_score = np.exp(-cv * 2)  # 使用指数函数，cv=0时得分为1

        return similarity_score

    def detect_area_outliers(self, markers_with_features):
        """
        检测面积异常值

        Args:
            markers_with_features: 带特征的标记点列表

        Returns:
            tuple: (正常点列表, 异常点列表)
        """
        if len(markers_with_features) < 3:
            return markers_with_features, []

        areas = np.array([m['pixel_area'] for m in markers_with_features])

        # 使用四分位数方法检测异常值
        Q1 = np.percentile(areas, 25)
        Q3 = np.percentile(areas, 75)
        IQR = Q3 - Q1

        # 异常值定义：超出 Q1-1.5*IQR 或 Q3+1.5*IQR
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        normal_markers = []
        outlier_markers = []

        for marker in markers_with_features:
            area = marker['pixel_area']
            if lower_bound <= area <= upper_bound:
                normal_markers.append(marker)
            else:
                outlier_markers.append(marker)

        if outlier_markers:
            self.update_result_text(f"面积异常值检测:\n")
            self.update_result_text(f"  正常范围: {lower_bound:.0f} - {upper_bound:.0f} 像素\n")
            self.update_result_text(f"  正常点: {len(normal_markers)}个\n")
            self.update_result_text(f"  异常点: {len(outlier_markers)}个\n")
            for marker in outlier_markers:
                self.update_result_text(f"    异常点: 位置=({marker['point'][0]:.1f}, {marker['point'][1]:.1f}), "
                                      f"面积={marker['pixel_area']}像素 (超出正常范围)\n")

        return normal_markers, outlier_markers

    def select_best_points_by_triple_criteria(self, markers_with_features, target_count=5):
        """
        基于三重标准选择最佳ABCDE点：
        1. 面积筛选：直接排除面积异常的点
        2. 圆度评估：评估点的圆形程度
        3. 蓝图匹配：评估与器械坐标蓝图的几何相似性

        Args:
            markers_with_features: 带特征的标记点列表
            target_count: 目标点数量

        Returns:
            list: 选择的最佳点列表
        """
        if len(markers_with_features) <= target_count:
            return markers_with_features

        self.update_result_text(f"开始三重标准选择最佳{target_count}个点...\n")
        self.update_result_text(f"标准1: 面积筛选 → 标准2: 圆度评估 → 标准3: 蓝图匹配\n")

        # 第一步：面积筛选 - 直接排除面积异常的点
        self.update_result_text(f"\n=== 第一步：面积筛选 ===\n")
        normal_markers, outlier_markers = self.detect_area_outliers(markers_with_features)

        if len(outlier_markers) > 0:
            self.update_result_text(f"排除{len(outlier_markers)}个面积异常点:\n")
            for marker in outlier_markers:
                self.update_result_text(f"  ❌ 位置=({marker['point'][0]:.1f}, {marker['point'][1]:.1f}), "
                                      f"面积={marker['pixel_area']}像素 (异常)\n")

        if len(normal_markers) < target_count:
            self.update_result_text(f"⚠️ 面积正常的点不够({len(normal_markers)}个)，需要包含部分面积异常点\n")

            # 按面积异常程度排序，选择异常程度最小的点
            if normal_markers:
                normal_areas = [m['pixel_area'] for m in normal_markers]
                median_area = np.median(normal_areas)
            else:
                all_areas = [m['pixel_area'] for m in markers_with_features]
                median_area = np.median(all_areas)

            # 计算每个异常点与中位数的差异，选择差异较小的
            outlier_markers_sorted = sorted(outlier_markers,
                                          key=lambda x: abs(x['pixel_area'] - median_area))

            needed_outliers = target_count - len(normal_markers)
            candidate_markers = normal_markers + outlier_markers_sorted[:needed_outliers]

            self.update_result_text(f"添加{needed_outliers}个异常程度最小的异常点:\n")
            for marker in outlier_markers_sorted[:needed_outliers]:
                self.update_result_text(f"  ✅ 位置=({marker['point'][0]:.1f}, {marker['point'][1]:.1f}), "
                                      f"面积={marker['pixel_area']}像素 (异常程度较小)\n")
        else:
            candidate_markers = normal_markers
            self.update_result_text(f"✅ 使用{len(normal_markers)}个面积正常的点进行后续筛选\n")

        # 第二步：从候选点中，基于圆度和蓝图匹配综合选择
        return self.select_by_circularity_and_blueprint_matching(candidate_markers, target_count)

    def select_by_circularity_and_blueprint_matching(self, candidate_markers, target_count=5):
        """
        基于圆度和蓝图匹配综合选择最佳点

        Args:
            candidate_markers: 候选标记点列表（已通过面积筛选）
            target_count: 目标点数量

        Returns:
            list: 选择的最佳点列表
        """
        self.update_result_text(f"\n=== 第二步：圆度和蓝图匹配综合评估 ===\n")

        if len(candidate_markers) <= target_count:
            return candidate_markers

        # 从候选点中选择所有可能的5点组合，进行综合评估
        from itertools import combinations

        # 限制组合数量，避免计算过慢
        if len(candidate_markers) > 12:
            # 按圆度排序，选择前12个进行组合分析
            candidate_markers = sorted(candidate_markers, key=lambda x: x['circularity'], reverse=True)[:12]
            self.update_result_text(f"候选点过多，按圆度排序后选择前12个进行组合分析\n")

        self.update_result_text(f"开始分析{len(candidate_markers)}个候选点的{target_count}点组合...\n")

        best_score = -1
        best_combination = None
        combination_count = 0

        # 遍历所有可能的5点组合
        for combination in combinations(candidate_markers, target_count):
            combination_count += 1

            # 提取点坐标
            points = [marker['point'] for marker in combination]

            # 计算三个评估指标
            circularity_score = self.calculate_circularity_score(combination)
            blueprint_score = self.calculate_blueprint_matching_score(points)

            # 综合评分：圆度权重0.5，蓝图匹配权重0.5
            combined_score = 0.6 * circularity_score + 0.4 * blueprint_score

            if combined_score > best_score:
                best_score = combined_score
                best_combination = combination

        self.update_result_text(f"分析了{combination_count}种组合，最佳组合得分: {best_score:.3f}\n")

        if best_combination:
            # 显示最佳组合的详细信息
            points = [marker['point'] for marker in best_combination]
            circularity_score = self.calculate_circularity_score(best_combination)
            blueprint_score = self.calculate_blueprint_matching_score(points)

            self.update_result_text(f"最佳组合详情:\n")
            self.update_result_text(f"  圆度评分: {circularity_score:.3f}\n")
            self.update_result_text(f"  蓝图匹配评分: {blueprint_score:.3f}\n")
            self.update_result_text(f"  综合得分: {best_score:.3f}\n")

            return list(best_combination)
        else:
            # 如果没有找到合适的组合，回退到按圆度选择
            self.update_result_text(f"未找到合适组合，回退到按圆度选择\n")
            sorted_markers = sorted(candidate_markers, key=lambda x: x['circularity'], reverse=True)
            return sorted_markers[:target_count]

    def calculate_circularity_score(self, markers):
        """
        计算一组点的圆度评分

        Args:
            markers: 标记点列表

        Returns:
            float: 圆度评分 (0-1)
        """
        circularities = [marker['circularity'] for marker in markers]

        # 平均圆度作为基础分数
        avg_circularity = np.mean(circularities)

        # 圆度一致性奖励：圆度越一致，奖励越高
        circularity_std = np.std(circularities)
        consistency_bonus = np.exp(-circularity_std * 3)  # 标准差越小，奖励越高

        # 综合圆度评分
        circularity_score = 0.7 * avg_circularity + 0.3 * consistency_bonus

        return min(1.0, circularity_score)

    def calculate_blueprint_matching_score(self, points):
        """
        计算点集与器械坐标蓝图的几何相似性
        基于线段比例相似性：检查候选点组成的图形各线段与蓝图模型的线段是否成比例

        Args:
            points: 点坐标列表 [[x1,y1], [x2,y2], ...]

        Returns:
            float: 蓝图匹配评分 (0-1)
        """
        if len(points) != 5:
            return 0.0

        # 器械坐标蓝图（来自工具标定的真实ABCDE坐标）
        # 使用tool_tip_calibration.py中定义的实际工具坐标
        ideal_blueprint = np.array([
            [0.0, 0.0],           # A点为原点
            [87.5, 0.0],          # B点在X轴上
            [33.6, 21.1],         # C点坐标
            [25.3, -33.6],        # D点坐标
            [48.9, -57.5]         # E点坐标
        ], dtype=np.float32)

        try:
            # 计算检测点集的所有线段长度
            detected_distances = self.calculate_all_distances(points)

            # 计算理想蓝图的所有线段长度
            ideal_distances = self.calculate_all_distances(ideal_blueprint.tolist())

            # 计算线段比例相似性
            similarity_score = self.calculate_distance_ratio_similarity(detected_distances, ideal_distances)

            self.update_result_text(f"    蓝图匹配分析: 线段比例相似性得分={similarity_score:.3f}\n")

            return similarity_score

        except Exception as e:
            self.update_result_text(f"    蓝图匹配计算失败: {e}\n")
            return 0.0

    def calculate_all_distances(self, points):
        """
        计算点集中所有点对之间的距离

        Args:
            points: 点坐标列表 [[x1,y1], [x2,y2], ...]

        Returns:
            list: 所有点对距离的列表
        """
        distances = []
        points_array = np.array(points, dtype=np.float32)

        for i in range(len(points_array)):
            for j in range(i+1, len(points_array)):
                dist = np.linalg.norm(points_array[i] - points_array[j])
                distances.append(dist)

        return distances

    def calculate_distance_ratio_similarity(self, detected_distances, ideal_distances):
        """
        计算两组距离的比例相似性

        Args:
            detected_distances: 检测点的距离列表
            ideal_distances: 理想蓝图的距离列表

        Returns:
            float: 相似性得分 (0-1)
        """
        if len(detected_distances) != len(ideal_distances):
            return 0.0

        # 将距离列表排序，以便比较对应的线段
        detected_sorted = sorted(detected_distances)
        ideal_sorted = sorted(ideal_distances)

        # 计算比例因子（使用最长的线段作为基准）
        if ideal_sorted[-1] == 0:
            return 0.0

        scale_factor = detected_sorted[-1] / ideal_sorted[-1]

        # 将理想距离按比例缩放
        scaled_ideal = [d * scale_factor for d in ideal_sorted]

        # 计算相对误差
        relative_errors = []
        for detected, scaled in zip(detected_sorted, scaled_ideal):
            if scaled > 0:
                relative_error = abs(detected - scaled) / scaled
                relative_errors.append(relative_error)

        if not relative_errors:
            return 0.0

        # 计算平均相对误差
        avg_relative_error = np.mean(relative_errors)

        # 转换为相似性得分：误差越小，得分越高
        # 使用指数函数，当平均相对误差为0时得分为1，误差增大时得分快速下降
        similarity_score = np.exp(-avg_relative_error * 5)  # 5是调节参数，控制得分下降速度

        return similarity_score

    def undistort_points(self, points):
        """
        使用相机标定参数对点进行去畸变处理

        注意：此方法仅用于显示对比，实际的PnP求解中，
        solvePnP会内部自动处理畸变，避免重复去畸变。

        Args:
            points: 原始点坐标列表 [[x1,y1], [x2,y2], ...]

        Returns:
            list: 去畸变后的点坐标列表 [[x1,y1], [x2,y2], ...]
        """
        if not points:
            return points

        # 转换为numpy数组格式，OpenCV要求的格式
        points_array = np.array(points, dtype=np.float32).reshape(-1, 1, 2)

        # 使用cv2.undistortPoints进行去畸变
        # 注意：cv2.undistortPoints返回的是归一化坐标，需要转换回像素坐标
        undistorted_normalized = cv2.undistortPoints(
            points_array,
            self.camera_matrix,
            self.dist_coeffs,
            P=self.camera_matrix  # 使用P参数直接得到像素坐标
        )

        # 转换回列表格式
        undistorted_points = undistorted_normalized.reshape(-1, 2).tolist()

        # 显示去畸变前后的对比（仅用于显示，实际PnP求解由solvePnP内部处理畸变）
        self.update_result_text("去畸变效果对比（仅供参考，实际由solvePnP内部处理）:\n")
        for i, (orig, undist) in enumerate(zip(points, undistorted_points)):
            diff_x = undist[0] - orig[0]
            diff_y = undist[1] - orig[1]
            diff_magnitude = np.sqrt(diff_x**2 + diff_y**2)
            self.update_result_text(f"  点{i+1}: ({orig[0]:.1f},{orig[1]:.1f}) → ({undist[0]:.1f},{undist[1]:.1f}) 偏移:{diff_magnitude:.2f}像素\n")

        return undistorted_points

    def select_best_5_points(self, markers):
        """
        基于蓝图坐标相似性匹配ABCDE点

        策略：
        1. 使用已知的ABCDE蓝图坐标作为参考模板
        2. 计算检测点与蓝图的形状相似性
        3. 找到最佳匹配的排列组合
        4. 基于几何形状匹配，不依赖最长边

        Args:
            markers: 检测到的标记点列表 [[x1,y1], [x2,y2], ...]

        Returns:
            dict: 选择的ABCDE点 {'A': (x,y), 'B': (x,y), 'C': (x,y), 'D': (x,y), 'E': (x,y)}
        """
        if len(markers) != 5:
            raise ValueError(f"需要恰好5个标记点，但检测到{len(markers)}个")

        markers = np.array(markers)
        self.update_result_text(f"开始基于蓝图相似性匹配ABCDE点...\n")
        for i, marker in enumerate(markers):
            self.update_result_text(f"  检测点{i+1}: ({marker[0]:.1f}, {marker[1]:.1f})\n")

        # ABCDE蓝图坐标（器械坐标系，单位：mm）
        blueprint_3d = np.array([
            [0.0, 0.0, 0.0],      # A点：原点
            [87.5, 0.0, 0.0],     # B点：X轴上
            [33.6, 21.1, 0.0],    # C点
            [25.3, -33.6, 0.0],   # D点
            [48.9, -57.5, 0.0]    # E点
        ], dtype=np.float32)

        # 提取2D蓝图坐标（忽略Z坐标，因为都是0）
        blueprint_2d = blueprint_3d[:, :2]

        self.update_result_text(f"蓝图坐标参考:\n")
        labels = ['A', 'B', 'C', 'D', 'E']
        for i, (label, point) in enumerate(zip(labels, blueprint_2d)):
            self.update_result_text(f"  {label}点蓝图: ({point[0]:.1f}, {point[1]:.1f})\n")

        # 尝试所有可能的排列组合，找到最佳匹配
        from itertools import permutations

        best_score = float('inf')
        best_assignment = None
        best_transform = None

        self.update_result_text(f"开始尝试所有排列组合匹配...\n")

        # 遍历所有可能的点分配
        for perm in permutations(range(5)):
            # 当前排列：markers[perm[i]] 对应 blueprint_2d[i]
            current_markers = markers[list(perm)]

            # 计算从蓝图到检测点的相似变换
            transform_score, transform_params = self.calculate_similarity_transform(blueprint_2d, current_markers)

            if transform_score < best_score:
                best_score = transform_score
                best_assignment = perm
                best_transform = transform_params

        self.update_result_text(f"最佳匹配得分: {best_score:.3f}\n")
        self.update_result_text(f"变换参数: 缩放={best_transform['scale']:.3f}, 旋转={best_transform['rotation']:.1f}°, 平移=({best_transform['translation'][0]:.1f}, {best_transform['translation'][1]:.1f})\n")

        # 根据最佳分配创建结果
        result_points = {}
        for i, label in enumerate(labels):
            marker_idx = best_assignment[i]
            point = markers[marker_idx]
            result_points[label] = (float(point[0]), float(point[1]))
            self.update_result_text(f"{label}点匹配: 检测点{marker_idx+1} ({point[0]:.1f}, {point[1]:.1f})\n")

        return result_points

    def calculate_similarity_transform(self, template_points, detected_points):
        """
        计算从模板点到检测点的相似变换，并返回匹配得分

        相似变换包括：缩放、旋转、平移

        Args:
            template_points: 模板点坐标 (5x2)
            detected_points: 检测点坐标 (5x2)

        Returns:
            tuple: (匹配得分, 变换参数字典)
        """
        try:
            # 计算质心
            template_center = np.mean(template_points, axis=0)
            detected_center = np.mean(detected_points, axis=0)

            # 中心化
            template_centered = template_points - template_center
            detected_centered = detected_points - detected_center

            # 计算缩放因子（使用RMS距离比）
            template_rms = np.sqrt(np.mean(np.sum(template_centered**2, axis=1)))
            detected_rms = np.sqrt(np.mean(np.sum(detected_centered**2, axis=1)))

            if template_rms == 0 or detected_rms == 0:
                return float('inf'), {}

            scale = detected_rms / template_rms

            # 缩放模板点
            template_scaled = template_centered * scale

            # 计算旋转角度（使用Procrustes分析）
            # H = template_scaled.T @ detected_centered
            H = np.dot(template_scaled.T, detected_centered)
            U, _, Vt = np.linalg.svd(H)
            R = np.dot(Vt.T, U.T)

            # 确保是旋转矩阵（行列式为1）
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = np.dot(Vt.T, U.T)

            # 计算旋转角度
            rotation_angle = np.arctan2(R[1, 0], R[0, 0]) * 180 / np.pi

            # 应用变换到模板点
            template_transformed = np.dot(template_scaled, R.T) + detected_center

            # 计算匹配误差（RMS距离）
            distances = np.sqrt(np.sum((template_transformed - detected_points)**2, axis=1))
            rms_error = np.sqrt(np.mean(distances**2))

            transform_params = {
                'scale': scale,
                'rotation': rotation_angle,
                'translation': detected_center - template_center * scale,
                'rms_error': rms_error
            }

            return rms_error, transform_params

        except Exception:
            return float('inf'), {}

    def detect_points(self):
        """分别检测两张图片的标记点"""
        if not self.start_image or not self.end_image:
            messagebox.showwarning("警告", "请先拍摄两张图片（起始点和终点）")
            return

        try:
            self.status_var.set("正在检测两张图片的标记点...")
            self.update_result_text("🔍 开始分别检测起始点和终点图片的标记点...\n")

            # 检测起始点图片
            self.update_result_text("\n📍 检测起始点图片标记点:\n")
            self.start_points = self.detect_single_image_points(self.start_image, "起始点")

            # 检测终点图片
            self.update_result_text("\n🎯 检测终点图片标记点:\n")
            self.end_points = self.detect_single_image_points(self.end_image, "终点")

            if self.start_points is not None and self.end_points is not None:
                self.calculate_tip_btn.config(state=tk.NORMAL)
                self.update_result_text("\n✅ 两张图片的标记点检测完成！\n")
                self.status_var.set("✅ 标记点检测完成，可以计算距离")

                # 显示带有标记点的并排图像
                self.display_dual_images_with_points()
            else:
                raise Exception("标记点检测失败")

        except Exception as e:
            messagebox.showerror("错误", f"检测标记点失败: {e}")
            self.status_var.set("检测失败")
            self.update_result_text(f"❌ 检测失败: {e}\n")

    def detect_single_image_points(self, image_path, image_type):
        """检测单张图片的标记点"""
        try:
            # 加载图片
            image = cv2.imread(image_path)
            if image is None:
                raise Exception(f"无法加载{image_type}图片")

            self.update_result_text(f"正在检测{image_type}图片标记点...\n")

            # 图像预处理
            processed_image = self.preprocess_image(image)
            self.update_result_text(f"{image_type}图像预处理完成\n")

            # 检测blob（返回带特征的点）
            markers_with_features = self.detect_blobs(processed_image)
            self.update_result_text(f"{image_type}初步检测到 {len(markers_with_features)} 个候选点\n")

            # 显示所有检测到的点的详细信息
            if markers_with_features:
                self.update_result_text(f"\n{image_type}检测到的点的详细信息:\n")
                self.update_result_text(f"{'序号':<4} {'位置':<15} {'圆度':<8} {'面积':<8} {'响应强度':<10}\n")
                self.update_result_text(f"{'-'*4} {'-'*15} {'-'*8} {'-'*8} {'-'*10}\n")

                # 按圆度降序排序显示
                sorted_markers = sorted(markers_with_features, key=lambda x: x['circularity'], reverse=True)
                for i, marker in enumerate(sorted_markers):
                    pos_str = f"({marker['point'][0]:.1f},{marker['point'][1]:.1f})"
                    self.update_result_text(f"{i+1:<4} {pos_str:<15} {marker['circularity']:.3f}{'':4} {marker['pixel_area']:<8} {marker['response']:.3f}\n")

                # 统计信息
                areas = [m['pixel_area'] for m in markers_with_features]
                circularities = [m['circularity'] for m in markers_with_features]

                self.update_result_text(f"\n{image_type}统计信息:\n")
                self.update_result_text(f"  圆度范围: {min(circularities):.3f} - {max(circularities):.3f}\n")
                self.update_result_text(f"  圆度平均: {np.mean(circularities):.3f} ± {np.std(circularities):.3f}\n")
                self.update_result_text(f"  面积范围: {min(areas)} - {max(areas)} 像素\n")
                self.update_result_text(f"  面积平均: {np.mean(areas):.1f} ± {np.std(areas):.1f} 像素\n")
                self.update_result_text(f"  面积变异系数: {np.std(areas)/np.mean(areas):.3f}\n")
            else:
                self.update_result_text(f"{image_type}未检测到任何候选点\n")

            if len(markers_with_features) >= 5:
                # 使用三重标准选择最佳的5个点（面积筛选 + 圆度评估 + 蓝图匹配）
                selected_markers = self.select_best_points_by_triple_criteria(markers_with_features, 5)

                self.update_result_text(f"{image_type}最终选择的5个点:\n")
                for i, marker in enumerate(selected_markers):
                    self.update_result_text(f"  选中点{i+1}: 圆度={marker['circularity']:.3f}, "
                                          f"面积={marker['pixel_area']}像素, "
                                          f"位置=({marker['point'][0]:.1f}, {marker['point'][1]:.1f})\n")

                # 提取点坐标用于后续处理
                markers = [marker['point'] for marker in selected_markers]

                # 直接使用原始点进行智能ABCDE排序（不进行手动去畸变）
                # solvePnP会在内部自动处理畸变，避免重复去畸变
                abcde_points = self.select_best_5_points(markers)
                self.update_result_text(f"{image_type}ABCDE点排序完成\n")

                # 按ABCDE顺序排列点（原始畸变点，solvePnP会内部去畸变）
                detected_points = np.array([
                    abcde_points['A'],
                    abcde_points['B'],
                    abcde_points['C'],
                    abcde_points['D'],
                    abcde_points['E']
                ], dtype=np.float32)

                self.update_result_text(f"✅ {image_type}成功检测并排序5个标记点 (ABCDE)\n")
                labels = ['A', 'B', 'C', 'D', 'E']
                for point, label in zip(detected_points, labels):
                    self.update_result_text(f"  {label}点: ({point[0]:.1f}, {point[1]:.1f})\n")

                # 保存标注后的图像到文件（用于调试）
                result_image = self.draw_detected_points_on_image(
                    image, markers_with_features, selected_markers, detected_points, image_type
                )

                return detected_points
            else:
                raise Exception(f"{image_type}只检测到 {len(markers_with_features)} 个点，需要至少5个点")

        except Exception as e:
            self.update_result_text(f"❌ {image_type}检测失败: {e}\n")
            return None

    def draw_detected_points_on_image(self, image, markers_with_features, selected_markers, detected_points, image_type):
        """在图像上绘制检测到的标记点（用于调试）"""
        result_image = image.copy()

        # 在图像顶部显示候选点总数和图像类型
        header_text = f"{image_type.upper()} - Total Candidates: {len(markers_with_features)}"
        cv2.putText(result_image, header_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)  # 白色背景
        cv2.putText(result_image, header_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)      # 红色文字

        # 绘制所有候选点（浅灰色，较小）并显示详细信息
        for i, marker in enumerate(markers_with_features):
            point = marker['point']
            x, y = int(point[0]), int(point[1])

            # 绘制候选点
            cv2.circle(result_image, (x, y), 6, (200, 200, 200), 2)
            cv2.circle(result_image, (x, y), 2, (150, 150, 150), -1)

            # 显示点的序号（白色背景，黑色文字）
            cv2.putText(result_image, f"{i+1}", (x+10, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(result_image, f"{i+1}", (x+10, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # 显示圆度值（蓝色）
            circularity_text = f"C:{marker['circularity']:.3f}"
            cv2.putText(result_image, circularity_text, (x+10, y+8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
            cv2.putText(result_image, circularity_text, (x+10, y+8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            # 显示面积值（绿色）
            area_text = f"A:{marker['pixel_area']}"
            cv2.putText(result_image, area_text, (x+10, y+25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
            cv2.putText(result_image, area_text, (x+10, y+25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # 绘制选中的5个候选点（红色，更大）
        for i, marker in enumerate(selected_markers):
            point = marker['point']
            x, y = int(point[0]), int(point[1])

            # 绘制选中标记（红色圆圈）
            cv2.circle(result_image, (x, y), 10, (0, 0, 255), 3)
            cv2.circle(result_image, (x, y), 3, (0, 0, 255), -1)

            # 显示"SELECTED"标记（红色）
            cv2.putText(result_image, "SELECTED", (x-30, y-40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(result_image, "SELECTED", (x-30, y-40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 绘制ABCDE点（不同颜色，最突出）
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]  # 蓝绿红黄紫
        labels = ['A', 'B', 'C', 'D', 'E']

        for point, color, label in zip(detected_points, colors, labels):
            x, y = int(point[0]), int(point[1])

            # 绘制最终选择的点（大圆圈）
            cv2.circle(result_image, (x, y), 15, color, -1)
            cv2.circle(result_image, (x, y), 18, (255, 255, 255), 3)  # 白色边框

            # 显示ABCDE标签（大字体）
            cv2.putText(result_image, label, (x+25, y+8),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            cv2.putText(result_image, label, (x+25, y+8),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

        # 在右上角显示检测完成状态（使用更短的文字）
        status_text = "DETECTED"
        text_width = 120  # 估算文字宽度
        cv2.putText(result_image, status_text,
                   (result_image.shape[1] - text_width, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
        cv2.putText(result_image, status_text,
                   (result_image.shape[1] - text_width, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 保存标注后的图像
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        # 使用英文名称避免编码问题
        image_type_en = "start" if "起始" in image_type else "end"
        output_filename = f"images/detected_{image_type_en}_{timestamp}.jpg"
        cv2.imwrite(output_filename, result_image)
        self.update_result_text(f"  标注图像已保存: {output_filename}\n")

        return result_image

    def display_dual_images_with_points(self):
        """显示带有标记点的起始点和终点图像并排"""
        if not self.start_image or not self.end_image:
            return

        try:
            # 读取两张图片
            start_img = cv2.imread(self.start_image)
            end_img = cv2.imread(self.end_image)

            if start_img is None or end_img is None:
                self.update_result_text("❌ 无法读取图片文件\n")
                return

            # 在图像上绘制标记点
            start_with_points = self.draw_points_on_single_image(start_img, self.start_points, "START")
            end_with_points = self.draw_points_on_single_image(end_img, self.end_points, "END")

            # 转换颜色空间
            start_rgb = cv2.cvtColor(start_with_points, cv2.COLOR_BGR2RGB)
            end_rgb = cv2.cvtColor(end_with_points, cv2.COLOR_BGR2RGB)

            # 调整图片大小使其一致
            target_height = 400
            start_h, start_w = start_rgb.shape[:2]
            end_h, end_w = end_rgb.shape[:2]

            # 按比例缩放
            start_scale = target_height / start_h
            end_scale = target_height / end_h

            start_new_w = int(start_w * start_scale)
            end_new_w = int(end_w * end_scale)

            start_resized = cv2.resize(start_rgb, (start_new_w, target_height))
            end_resized = cv2.resize(end_rgb, (end_new_w, target_height))

            # 创建并排图像
            gap = 30
            total_width = start_new_w + end_new_w + gap
            combined_img = np.ones((target_height, total_width, 3), dtype=np.uint8) * 240

            # 放置图片
            combined_img[:, :start_new_w] = start_resized
            combined_img[:, start_new_w+gap:start_new_w+gap+end_new_w] = end_resized

            # 添加标签
            cv2.putText(combined_img, "START POINT", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.putText(combined_img, "END POINT", (start_new_w+gap+10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            # 添加分隔线
            line_x = start_new_w + gap // 2
            cv2.line(combined_img, (line_x, 0), (line_x, target_height), (100, 100, 100), 2)

            # 转换为PIL图像并显示
            pil_image = Image.fromarray(combined_img)
            photo = ImageTk.PhotoImage(pil_image)

            self.image_label.configure(image=photo)
            self.image_label.image = photo

            self.update_result_text("📷 已显示带有标记点的起始点和终点图像\n")

        except Exception as e:
            self.update_result_text(f"❌ 显示图片失败: {e}\n")

    def draw_points_on_single_image(self, image, detected_points, image_type):
        """在单张图像上绘制ABCDE标记点"""
        result_image = image.copy()

        if detected_points is None:
            return result_image

        # 绘制ABCDE点（不同颜色，更突出）
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]  # 蓝绿红黄紫
        labels = ['A', 'B', 'C', 'D', 'E']

        for point, color, label in zip(detected_points, colors, labels):
            # 绘制大圆点
            cv2.circle(result_image, (int(point[0]), int(point[1])), 15, color, -1)
            cv2.circle(result_image, (int(point[0]), int(point[1])), 18, (255, 255, 255), 3)  # 白色边框

            # 绘制标签
            cv2.putText(result_image, label, (int(point[0])+25, int(point[1])+8),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(result_image, label, (int(point[0])+25, int(point[1])+8),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 1)  # 白色边框

        # 不添加图像类型标签，避免与并排图像的标签重叠

        return result_image
            
    def update_result_text(self, text):
        """更新结果显示文本"""
        # 检查result_text控件是否已创建
        if hasattr(self, 'result_text') and self.result_text:
            self.result_text.insert(tk.END, text)
            self.result_text.see(tk.END)
            self.root.update_idletasks()
        else:
            # 如果控件还没创建，打印到控制台
            print(text.strip())

    def calculate_tip_position(self):
        """计算两个尖端位置的欧式距离"""
        if self.start_points is None or self.end_points is None:
            messagebox.showwarning("警告", "请先检测两张图片的标记点")
            return

        try:
            self.status_var.set("正在计算两点间距离...")
            self.update_result_text("\n🔍 开始计算起始点和终点的尖端位置...\n")

            # 器械坐标系下的5个标记点坐标（从tool_tip_calibration.py中获取）
            points_3d_tool = np.array([
                [0.0, 0.0, 0.0],      # A点：原点
                [87.5, 0.0, 0.0],     # B点：X轴上
                [33.6, 21.1, 0.0],    # C点
                [25.3, -33.6, 0.0],   # D点
                [48.9, -57.5, 0.0]    # E点
            ], dtype=np.float32)

            # 计算起始点的尖端位置
            self.update_result_text("📍 计算起始点尖端位置...\n")
            self.start_tip_position = self.calculate_single_tip_position(
                self.start_points, points_3d_tool, "起始点"
            )

            # 计算终点的尖端位置
            self.update_result_text("🎯 计算终点尖端位置...\n")
            self.end_tip_position = self.calculate_single_tip_position(
                self.end_points, points_3d_tool, "终点"
            )

            if self.start_tip_position is not None and self.end_tip_position is not None:
                # 计算欧式距离
                distance = np.linalg.norm(self.end_tip_position - self.start_tip_position)

                # 显示结果
                self.update_result_text("\n" + "="*50 + "\n")
                self.update_result_text("📏 两点间距离计算结果\n")
                self.update_result_text("="*50 + "\n")
                self.update_result_text(f"📍 起始点尖端坐标: [{self.start_tip_position[0]:.3f}, {self.start_tip_position[1]:.3f}, {self.start_tip_position[2]:.3f}] mm\n")
                self.update_result_text(f"🎯 终点尖端坐标:   [{self.end_tip_position[0]:.3f}, {self.end_tip_position[1]:.3f}, {self.end_tip_position[2]:.3f}] mm\n")
                self.update_result_text(f"📐 两点间欧式距离: {distance:.3f} mm\n")
                self.update_result_text("="*50 + "\n")

                self.status_var.set(f"✅ 两点间距离: {distance:.3f} mm")
            else:
                raise Exception("无法计算尖端位置")

        except Exception as e:
            messagebox.showerror("错误", f"计算距离失败: {e}")
            self.status_var.set("计算失败")
            self.update_result_text(f"❌ 计算失败: {e}\n")

    def calculate_single_tip_position(self, detected_points, points_3d_tool, point_type):
        """计算单个图片的尖端位置"""
        try:
            # 使用PnP求解相机姿态
            success, rvec, tvec = cv2.solvePnP(
                points_3d_tool, detected_points,
                self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_EPNP
            )

            if not success:
                raise Exception(f"{point_type}PnP求解失败")

            # 使用LM精化提高精度
            rvec_refined, tvec_refined = cv2.solvePnPRefineLM(
                points_3d_tool, detected_points,
                self.camera_matrix, self.dist_coeffs,
                rvec, tvec
            )

            # 计算重投影误差
            projected_points, _ = cv2.projectPoints(
                points_3d_tool, rvec_refined, tvec_refined,
                self.camera_matrix, self.dist_coeffs
            )
            projected_points = projected_points.reshape(-1, 2)
            individual_errors = np.sqrt(np.sum((projected_points - detected_points)**2, axis=1))
            reprojection_error = np.mean(individual_errors)

            # 将器械坐标系下的尖端坐标转换到世界坐标系
            R, _ = cv2.Rodrigues(rvec_refined)
            T = tvec_refined.flatten()

            # 使用标定得到的器械坐标系下的尖端位置
            tip_world_calculated = R @ self.tip_tool + T

            self.update_result_text(f"  {point_type}重投影误差: {reprojection_error:.3f} 像素\n")
            self.update_result_text(f"  {point_type}尖端坐标: [{tip_world_calculated[0]:.3f}, {tip_world_calculated[1]:.3f}, {tip_world_calculated[2]:.3f}] mm\n")

            return tip_world_calculated

        except Exception as e:
            self.update_result_text(f"❌ {point_type}计算失败: {e}\n")
            return None

    def analyze_reprojection_error(self, individual_errors, projected_points, detected_points):
        """
        分析重投影误差，找出问题点

        Args:
            individual_errors: 每个点的重投影误差
            projected_points: 投影点坐标
            detected_points: 检测点坐标
        """
        self.update_result_text("\n=== 重投影误差详细分析 ===\n")

        # 显示每个点的误差
        for i, error in enumerate(individual_errors):
            detected = detected_points[i]
            projected = projected_points[i]
            diff_x = projected[0] - detected[0]
            diff_y = projected[1] - detected[1]

            status = "✅ 良好" if error < 3.0 else "⚠️ 偏大" if error < 5.0 else "❌ 过大"
            self.update_result_text(f"点{i+1}: 误差={error:.2f}像素 {status}\n")
            self.update_result_text(f"  检测坐标: ({detected[0]:.1f}, {detected[1]:.1f})\n")
            self.update_result_text(f"  投影坐标: ({projected[0]:.1f}, {projected[1]:.1f})\n")
            self.update_result_text(f"  偏移: dx={diff_x:.2f}, dy={diff_y:.2f}\n")

        # 找出误差最大的点
        max_error_idx = np.argmax(individual_errors)
        self.update_result_text(f"\n误差最大的点: 点{max_error_idx+1}, 误差={individual_errors[max_error_idx]:.2f}像素\n")

        # 提供可能的原因和建议
        self.update_result_text("\n可能的原因:\n")
        if np.max(individual_errors) > 5.0:
            self.update_result_text("1. 点检测不准确 - 尝试改进检测算法或手动调整点位置\n")
            self.update_result_text("2. 点匹配错误 - 检查ABCDE点的对应关系是否正确\n")
            self.update_result_text("3. 相机标定不准确 - 重新进行相机标定\n")
            self.update_result_text("4. 工具标定坐标不准确 - 检查3D坐标是否正确\n")

    def optimize_pose_estimation(self, points_3d, points_2d, initial_rvec, initial_tvec):
        """
        优化位姿估计，尝试减少重投影误差

        Args:
            points_3d: 3D点坐标
            points_2d: 2D点坐标
            initial_rvec: 初始旋转向量
            initial_tvec: 初始平移向量

        Returns:
            tuple: (优化后的rvec, 优化后的tvec, 优化后的误差)
        """
        try:
            # 尝试不同的PnP算法
            methods = [
                (cv2.SOLVEPNP_ITERATIVE, "迭代法"),
                (cv2.SOLVEPNP_EPNP, "EPnP"),
                (cv2.SOLVEPNP_P3P, "P3P"),
                (cv2.SOLVEPNP_AP3P, "AP3P"),
                (cv2.SOLVEPNP_IPPE, "IPPE"),
                (cv2.SOLVEPNP_IPPE_SQUARE, "IPPE_SQUARE")
            ]

            best_error = float('inf')
            best_rvec = initial_rvec
            best_tvec = initial_tvec
            best_method = "原始方法"

            for method_flag, method_name in methods:
                try:
                    # 使用不同的方法求解PnP
                    success, rvec, tvec = cv2.solvePnP(
                        points_3d, points_2d,
                        self.camera_matrix, self.dist_coeffs,
                        flags=method_flag
                    )

                    if not success:
                        continue

                    # 使用LM精化
                    rvec_refined, tvec_refined = cv2.solvePnPRefineLM(
                        points_3d, points_2d,
                        self.camera_matrix, self.dist_coeffs,
                        rvec, tvec
                    )

                    # 计算重投影误差
                    projected_points, _ = cv2.projectPoints(
                        points_3d, rvec_refined, tvec_refined,
                        self.camera_matrix, self.dist_coeffs
                    )
                    projected_points = projected_points.reshape(-1, 2)

                    error = np.mean(np.sqrt(np.sum((projected_points - points_2d)**2, axis=1)))

                    self.update_result_text(f"  {method_name}: 误差={error:.3f}像素\n")

                    if error < best_error:
                        best_error = error
                        best_rvec = rvec_refined
                        best_tvec = tvec_refined
                        best_method = method_name

                except Exception:
                    continue

            self.update_result_text(f"最佳方法: {best_method}, 误差={best_error:.3f}像素\n")
            return best_rvec, best_tvec, best_error

        except Exception as e:
            self.update_result_text(f"优化失败: {e}\n")
            return initial_rvec, initial_tvec, float('inf')

    def draw_coordinate_system(self, image, rvec, tvec):
        """在图像上绘制坐标系和尖端位置"""
        result_image = image.copy()

        # 绘制检测到的点
        for i, point in enumerate(self.detected_points):
            cv2.circle(result_image, (int(point[0]), int(point[1])), 8, (0, 255, 0), -1)
            cv2.putText(result_image, f"P{i+1}", (int(point[0])+10, int(point[1])),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 绘制器械坐标系原点（A点）
        origin_3d = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        origin_2d, _ = cv2.projectPoints(origin_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        origin_2d = origin_2d.reshape(-1, 2)[0]

        # 绘制坐标轴
        axis_length = 50.0
        axes_3d = np.array([
            [0.0, 0.0, 0.0],      # 原点
            [axis_length, 0.0, 0.0],  # X轴
            [0.0, axis_length, 0.0],  # Y轴
            [0.0, 0.0, axis_length]   # Z轴
        ], dtype=np.float32)

        axes_2d, _ = cv2.projectPoints(axes_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        axes_2d = axes_2d.reshape(-1, 2)

        # 绘制坐标轴
        origin = tuple(axes_2d[0].astype(int))
        x_axis = tuple(axes_2d[1].astype(int))
        y_axis = tuple(axes_2d[2].astype(int))
        z_axis = tuple(axes_2d[3].astype(int))

        cv2.line(result_image, origin, x_axis, (0, 0, 255), 3)  # X轴：红色
        cv2.line(result_image, origin, y_axis, (0, 255, 0), 3)  # Y轴：绿色
        cv2.line(result_image, origin, z_axis, (255, 0, 0), 3)  # Z轴：蓝色

        # 标注坐标轴
        cv2.putText(result_image, "X", x_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(result_image, "Y", y_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(result_image, "Z", z_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # 绘制尖端位置
        tip_3d = np.array([self.tip_tool], dtype=np.float32)
        tip_2d, _ = cv2.projectPoints(tip_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        tip_2d = tip_2d.reshape(-1, 2)[0]

        cv2.circle(result_image, tuple(tip_2d.astype(int)), 12, (255, 255, 0), -1)  # 尖端：黄色
        cv2.putText(result_image, "TIP", (int(tip_2d[0])+15, int(tip_2d[1])),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        return result_image

    def run(self):
        """运行程序"""
        self.root.mainloop()


if __name__ == "__main__":
    # 创建并运行交互式尖端跟踪器
    app = InteractiveTipTracker()
    app.run()
