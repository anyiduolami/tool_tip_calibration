# 🔍 标记点检测系统

基于OpenCV的圆形标记点检测系统，支持相机标定和高精度标记点检测。

## 📋 功能特性

- **📷 相机标定**: 使用棋盘格进行相机内参标定和畸变校正
- **🔍 标记点检测**: 基于Blob检测算法的圆形标记点检测
- **⚙️ 参数优化**: 可调整检测参数以适应不同场景
- **📊 批量处理**: 支持单张图像和批量图像处理
- **🎨 结果可视化**: 直观显示检测结果

## 🚀 快速开始

### 方法1: 使用批处理文件
双击 `启动标记点检测.bat`

### 方法2: 使用Python脚本
```bash
python run_detection.py
```

### 方法3: 直接运行检测
```bash
python test_marker_detection.py
```

## 📁 项目结构

```
📁 标记点检测系统/
├── 📁 calibration/             # 相机标定模块
│   ├── camera_calibration.py       # 相机标定脚本
│   ├── debug_calibration.py        # 标定调试
│   ├── view_calibration_results.py # 查看标定结果
│   └── camera_calibration.npz      # 标定数据文件
├── 📁 detection/               # 标记点检测模块
│   ├── marker_detector.py          # 主要检测器
│   ├── blob_detection_optimizer.py # 参数优化器
│   └── ...                         # 其他检测算法
├── 📁 tip_images/              # 测试图像文件夹
├── 📁 calib/                   # 标定图像文件夹
├── test_marker_detection.py    # 检测测试脚本
├── run_detection.py           # 主启动脚本
└── 启动标记点检测.bat          # 批处理启动文件
```

## 🔧 使用步骤

### 1. 相机标定 (可选)
如果需要去除图像畸变:
1. 将棋盘格标定图像放入 `calib/` 文件夹
2. 运行相机标定: `python calibration/camera_calibration.py`
3. 生成 `camera_calibration.npz` 标定文件

### 2. 标记点检测
1. 将待检测图像放入 `tip_images/` 文件夹
2. 运行检测程序: `python test_marker_detection.py`
3. 选择测试模式:
   - 单张图像测试
   - 批量测试
   - 参数优化

### 3. 参数调整
如果检测效果不理想:
1. 运行参数优化器: `python detection/blob_detection_optimizer.py`
2. 调整检测参数
3. 重新测试

## ⚙️ 检测参数说明

主要参数 (在 `marker_detector.py` 中):
- `minArea`: 最小面积 (默认: 100)
- `maxArea`: 最大面积 (默认: 2000)
- `minCircularity`: 最小圆度 (默认: 0.6)
- `minConvexity`: 最小凸度 (默认: 0.85)
- `minInertiaRatio`: 最小惯性率 (默认: 0.3)

## 📊 检测结果

检测结果包括:
- 标记点坐标 (像素坐标)
- 检测数量统计
- 可视化结果图像
- 批量处理统计信息

## 🛠️ 依赖环境

- Python 3.7+
- OpenCV 4.0+
- NumPy
- (可选) Matplotlib

## 📝 更新日志

- v1.0: 初始版本，支持基本的标记点检测
- v1.1: 添加相机标定功能
- v1.2: 优化检测参数，提高精度
- v2.0: 简化系统，专注于标记点检测

## 📞 技术支持

如有问题，请检查:
1. 图像质量是否清晰
2. 标记点是否为圆形且对比度足够
3. 检测参数是否适合当前场景
4. 相机标定是否正确 (如果使用)

---
*标记点检测系统 - 专业、简洁、高效*
