import cv2
import numpy as np
import glob
import os

class ZhangCalibration:
    def __init__(self, chessboard_size=(9, 6), square_size=1.0):
        """
        张正友相机标定法实现
        
        Args:
            chessboard_size: 棋盘格内角点数量 (width, height)
            square_size: 棋盘格方格的实际尺寸 (单位: mm)
        """
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        
        # 准备3D点坐标 (世界坐标系)
        self.objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
        
        # 存储所有图像的3D点和2D点
        self.objpoints = []  # 3D点
        self.imgpoints = []  # 2D点
        
    def find_corners(self, image_path, debug=False):
        """在单张图像中寻找棋盘格角点"""
        img = cv2.imread(image_path)
        if img is None:
            print(f"  ✗ 无法读取图像: {image_path}")
            return False, None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 尝试多种检测参数
        flags_list = [
            None,
            cv2.CALIB_CB_ADAPTIVE_THRESH,
            cv2.CALIB_CB_NORMALIZE_IMAGE,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FILTER_QUADS,
        ]

        for flags in flags_list:
            # 寻找棋盘格角点
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, flags)

            if ret:
                if debug:
                    print(f"    成功检测到角点，使用参数: {flags}")

                # 亚像素精度优化
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                self.objpoints.append(self.objp)
                self.imgpoints.append(corners2)

                # 可视化角点检测结果
                img_with_corners = cv2.drawChessboardCorners(img, self.chessboard_size, corners2, ret)
                return True, img_with_corners

        if debug:
            print(f"    尝试了所有检测参数都未能找到 {self.chessboard_size} 的棋盘格")

        return False, img

    def detect_chessboard_size(self, image_path):
        """自动检测棋盘格尺寸"""
        img = cv2.imread(image_path)
        if img is None:
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 常见的棋盘格尺寸
        common_sizes = [
            (9, 6), (8, 6), (7, 5), (6, 4), (10, 7), (11, 8),
            (6, 9), (6, 8), (5, 7), (4, 6), (7, 10), (8, 11),
            (9, 7), (7, 9), (8, 5), (5, 8), (10, 6), (6, 10)
        ]

        print(f"正在检测棋盘格尺寸...")

        for size in common_sizes:
            ret, corners = cv2.findChessboardCorners(gray, size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ret:
                print(f"检测到棋盘格尺寸: {size}")
                return size

        print("未能自动检测到棋盘格尺寸")
        return None

    def calibrate_camera(self, calib_folder, auto_detect_size=True):
        """
        执行相机标定

        Args:
            calib_folder: 包含标定图片的文件夹路径
            auto_detect_size: 是否自动检测棋盘格尺寸
        """
        # 获取所有图片文件，但排除角点检测结果图片
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
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
            raise ValueError(f"在 {calib_folder} 文件夹中未找到图片文件")

        print(f"找到 {len(image_files)} 张标定图片")

        # 自动检测棋盘格尺寸
        if auto_detect_size:
            detected_size = self.detect_chessboard_size(image_files[0])
            if detected_size:
                print(f"自动检测到棋盘格尺寸: {detected_size}")
                self.chessboard_size = detected_size
                # 重新计算3D点坐标
                self.objp = np.zeros((detected_size[0] * detected_size[1], 3), np.float32)
                self.objp[:, :2] = np.mgrid[0:detected_size[0], 0:detected_size[1]].T.reshape(-1, 2)
                self.objp *= self.square_size
                # 清空之前的点
                self.objpoints = []
                self.imgpoints = []
            else:
                print(f"使用默认棋盘格尺寸: {self.chessboard_size}")

        print(f"当前使用的棋盘格尺寸: {self.chessboard_size}")
        
        successful_images = 0

        # 处理每张图片
        for img_path in image_files:
            print(f"处理图片: {os.path.basename(img_path)}")
            success, img_with_corners = self.find_corners(img_path, debug=True)

            if success:
                successful_images += 1
                print(f"  ✓ 成功检测到角点")

                # 保存角点检测结果
                output_path = os.path.join(calib_folder, f"corners_{os.path.basename(img_path)}")
                cv2.imwrite(output_path, img_with_corners)
            else:
                print(f"  ✗ 未能检测到角点")
        
        print(f"\n成功处理 {successful_images}/{len(image_files)} 张图片")
        
        if successful_images < 3:
            raise ValueError("至少需要3张成功检测到角点的图片进行标定")
        
        # 执行相机标定
        img_shape = cv2.imread(image_files[0]).shape[:2][::-1]  # (width, height)
        
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, img_shape, None, None
        )
        
        if not ret:
            raise RuntimeError("相机标定失败")
        
        # 计算重投影误差
        total_error = 0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(
                self.objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
            )
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        
        mean_error = total_error / len(self.objpoints)
        
        return {
            'camera_matrix': camera_matrix,
            'distortion_coefficients': dist_coeffs,
            'rotation_vectors': rvecs,
            'translation_vectors': tvecs,
            'reprojection_error': mean_error,
            'image_shape': img_shape
        }
    
    def save_calibration_results(self, results, output_file='camera_calibration.npz'):
        """保存标定结果"""
        np.savez(output_file,
                camera_matrix=results['camera_matrix'],
                dist_coeffs=results['distortion_coefficients'],
                rvecs=results['rotation_vectors'],
                tvecs=results['translation_vectors'],
                reprojection_error=results['reprojection_error'],
                mean_error=results['reprojection_error'],  # 兼容性
                image_shape=results['image_shape'])
        print(f"标定结果已保存到: {output_file}")
    
    def print_results(self, results):
        """打印标定结果"""
        print("\n" + "="*50)
        print("相机标定结果")
        print("="*50)
        print(f"重投影误差: {results['reprojection_error']:.4f} 像素")
        print(f"图像尺寸: {results['image_shape']}")
        print("\n相机内参矩阵:")
        print(results['camera_matrix'])
        print(f"\n焦距 fx: {results['camera_matrix'][0,0]:.2f}")
        print(f"焦距 fy: {results['camera_matrix'][1,1]:.2f}")
        print(f"主点 cx: {results['camera_matrix'][0,2]:.2f}")
        print(f"主点 cy: {results['camera_matrix'][1,2]:.2f}")
        print("\n畸变系数:")
        print(results['distortion_coefficients'].flatten())

def main():
    # 创建标定对象
    calibrator = ZhangCalibration(
        chessboard_size=(8, 6),  # 根据实际棋盘格调整 - 自动检测为(8,6)
        square_size=35.0 # 方格尺寸，单位mm - 恢复原始值
    )
    
    try:
        # 执行标定
        results = calibrator.calibrate_camera('calib')
        
        # 打印结果
        calibrator.print_results(results)
        
        # 保存结果
        calibrator.save_calibration_results(results)
        
    except Exception as e:
        print(f"标定过程中出现错误: {e}")

if __name__ == "__main__":
    main()