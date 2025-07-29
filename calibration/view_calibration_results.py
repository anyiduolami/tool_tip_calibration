import numpy as np
import cv2
import os

def load_and_display_calibration(npz_file='camera_calibration.npz'):
    """加载并显示相机标定结果"""
    
    if not os.path.exists(npz_file):
        print(f"文件 {npz_file} 不存在")
        return None
    
    # 加载 .npz 文件
    data = np.load(npz_file)
    
    print("="*60)
    print("相机标定结果详细信息")
    print("="*60)
    
    # 显示文件中包含的所有数组
    print(f"文件: {npz_file}")
    print(f"包含的数组: {list(data.keys())}")
    print()
    
    # 提取各个参数
    camera_matrix = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']

    # 兼容不同的键名
    if 'reprojection_error' in data:
        reprojection_error = data['reprojection_error']
    elif 'mean_error' in data:
        reprojection_error = data['mean_error']
    else:
        reprojection_error = 0.0
        print("   ⚠ 未找到重投影误差数据")

    if 'image_shape' in data:
        image_shape = data['image_shape']
    else:
        image_shape = (640, 480)  # 默认值
        print("   ⚠ 未找到图像尺寸数据，使用默认值")
    
    # 显示相机内参矩阵
    print("1. 相机内参矩阵 (Camera Matrix):")
    print("   这个3x3矩阵包含了相机的内部参数")
    print(camera_matrix)
    print()
    
    # 解释内参矩阵的各个元素
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    print("   内参矩阵解释:")
    print(f"   - fx (水平焦距): {fx:.2f} 像素")
    print(f"   - fy (垂直焦距): {fy:.2f} 像素")
    print(f"   - cx (主点x坐标): {cx:.2f} 像素")
    print(f"   - cy (主点y坐标): {cy:.2f} 像素")
    print()
    
    # 显示畸变系数
    print("2. 畸变系数 (Distortion Coefficients):")
    print("   这些系数用于校正镜头畸变")
    print(f"   {dist_coeffs.flatten()}")
    print()
    print("   畸变系数解释:")
    print(f"   - k1 (径向畸变1): {dist_coeffs[0][0]:.6f}")
    print(f"   - k2 (径向畸变2): {dist_coeffs[0][1]:.6f}")
    print(f"   - p1 (切向畸变1): {dist_coeffs[0][2]:.6f}")
    print(f"   - p2 (切向畸变2): {dist_coeffs[0][3]:.6f}")
    print(f"   - k3 (径向畸变3): {dist_coeffs[0][4]:.6f}")
    print()
    
    # 显示其他信息
    print("3. 其他标定信息:")
    print(f"   - 重投影误差: {reprojection_error:.4f} 像素")
    print(f"   - 图像尺寸: {image_shape} (宽x高)")
    print()
    
    # 计算视场角
    width, height = image_shape
    fov_x = 2 * np.arctan(width / (2 * fx)) * 180 / np.pi
    fov_y = 2 * np.arctan(height / (2 * fy)) * 180 / np.pi
    
    print("4. 计算得出的相机参数:")
    print(f"   - 水平视场角: {fov_x:.1f}°")
    print(f"   - 垂直视场角: {fov_y:.1f}°")
    print(f"   - 像素纵横比: {fx/fy:.4f}")
    print()
    
    # 评估标定质量
    print("5. 标定质量评估:")
    if reprojection_error < 0.5:
        quality = "优秀"
    elif reprojection_error < 1.0:
        quality = "良好"
    elif reprojection_error < 2.0:
        quality = "一般"
    else:
        quality = "较差"
    
    print(f"   重投影误差 {reprojection_error:.4f} 像素 - 标定质量: {quality}")
    
    if abs(fx - fy) / max(fx, fy) < 0.01:
        print("   ✓ 焦距比例正常 (fx ≈ fy)")
    else:
        print("   ⚠ 焦距比例异常，可能存在问题")
    
    if abs(cx - width/2) < width*0.1 and abs(cy - height/2) < height*0.1:
        print("   ✓ 主点位置正常 (接近图像中心)")
    else:
        print("   ⚠ 主点偏离图像中心较远")
    
    print()
    
    return {
        'camera_matrix': camera_matrix,
        'dist_coeffs': dist_coeffs,
        'reprojection_error': reprojection_error,
        'image_shape': image_shape
    }

def save_calibration_to_txt(npz_file='camera_calibration.npz', txt_file='camera_calibration.txt'):
    """将标定结果保存为可读的文本文件"""
    
    data = np.load(npz_file)
    
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("相机标定结果\n")
        f.write("="*50 + "\n\n")
        
        f.write("相机内参矩阵:\n")
        np.savetxt(f, data['camera_matrix'], fmt='%.6f')
        f.write("\n")
        
        f.write("畸变系数:\n")
        np.savetxt(f, data['dist_coeffs'], fmt='%.6f')
        f.write("\n")
        
        # 兼容不同的键名
        if 'reprojection_error' in data:
            f.write(f"重投影误差: {data['reprojection_error']:.6f}\n")
        elif 'mean_error' in data:
            f.write(f"重投影误差: {data['mean_error']:.6f}\n")

        if 'image_shape' in data:
            f.write(f"图像尺寸: {data['image_shape']}\n")
        else:
            f.write(f"图像尺寸: (640, 480)\n")
    
    print(f"标定结果已保存到文本文件: {txt_file}")

def main():
    # 检查标定文件是否存在
    npz_file = 'camera_calibration.npz'
    
    if not os.path.exists(npz_file):
        print(f"标定文件 {npz_file} 不存在")
        print("请先运行 camera_calibration.py 进行相机标定")
        return
    
    # 显示标定结果
    results = load_and_display_calibration(npz_file)
    
    if results:
        # 保存为文本文件
        save_calibration_to_txt(npz_file)
        
        print("\n" + "="*60)
        print("使用说明:")
        print("1. 这些参数可以用于图像去畸变")
        print("2. 可以用于3D重建、测距等应用")
        print("3. 运行 undistort_images.py 可以测试去畸变效果")
        print("="*60)

if __name__ == "__main__":
    main()
