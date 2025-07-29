import cv2
import numpy as np
import os

def debug_chessboard_detection(image_path):
    """调试棋盘格检测，显示图像和尝试不同的参数"""
    
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"图像尺寸: {img.shape}")
    print(f"图像路径: {image_path}")
    
    # 常见的棋盘格尺寸
    common_sizes = [
        (9, 6), (8, 6), (7, 5), (6, 4), (10, 7), (11, 8),
        (6, 9), (6, 8), (5, 7), (4, 6), (7, 10), (8, 11),
        (9, 7), (7, 9), (8, 5), (5, 8), (10, 6), (6, 10),
        (5, 4), (4, 3), (3, 3), (12, 9), (13, 10)
    ]
    
    # 不同的检测标志
    flags_list = [
        ("默认", None),
        ("自适应阈值", cv2.CALIB_CB_ADAPTIVE_THRESH),
        ("图像归一化", cv2.CALIB_CB_NORMALIZE_IMAGE),
        ("自适应+归一化", cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE),
        ("自适应+滤波", cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FILTER_QUADS),
    ]
    
    print("\n开始检测棋盘格...")
    found_any = False
    
    for size in common_sizes:
        for flag_name, flags in flags_list:
            ret, corners = cv2.findChessboardCorners(gray, size, flags)
            if ret:
                print(f"✓ 找到棋盘格! 尺寸: {size}, 参数: {flag_name}")
                
                # 绘制角点
                img_copy = img.copy()
                cv2.drawChessboardCorners(img_copy, size, corners, ret)
                
                # 保存结果
                output_name = f"detected_{size[0]}x{size[1]}_{flag_name.replace('+', '_')}.jpg"
                cv2.imwrite(output_name, img_copy)
                print(f"  结果已保存到: {output_name}")
                found_any = True
                break
        if found_any:
            break
    
    if not found_any:
        print("✗ 未能检测到任何棋盘格模式")
        print("\n可能的原因:")
        print("1. 图像中没有标准的棋盘格")
        print("2. 棋盘格被部分遮挡或变形")
        print("3. 图像质量不佳（模糊、光照不均等）")
        print("4. 棋盘格尺寸不在常见范围内")
        
        # 保存原图的灰度版本用于检查
        cv2.imwrite("debug_gray.jpg", gray)
        print("已保存灰度图像到 debug_gray.jpg 用于检查")

def main():
    # 检查calib文件夹中的第一张图片
    calib_folder = "calib"
    
    if not os.path.exists(calib_folder):
        print(f"文件夹 {calib_folder} 不存在")
        return
    
    # 获取第一张图片
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    import glob
    
    image_files = []
    for ext in image_extensions:
        files = glob.glob(os.path.join(calib_folder, ext))
        files.extend(glob.glob(os.path.join(calib_folder, ext.upper())))
        # 过滤掉以 'corners_' 开头的文件
        files = [f for f in files if not os.path.basename(f).startswith('corners_')]
        image_files.extend(files)

    # 去除重复文件（Windows系统不区分大小写可能导致重复）
    image_files = list(set(image_files))
    
    if not image_files:
        print(f"在 {calib_folder} 文件夹中未找到图片文件")
        return
    
    print(f"找到 {len(image_files)} 张图片，将调试第一张:")
    debug_chessboard_detection(image_files[0])

if __name__ == "__main__":
    main()
