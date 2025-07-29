#!/usr/bin/env python3
"""
标记点检测参数配置文件

包含经过优化的最佳检测参数

作者: AI Assistant
日期: 2025-01-18
"""

# 经过调试优化的最佳参数
OPTIMIZED_BLOB_PARAMS = {
    # 阈值参数
    'min_threshold': 40,
    'max_threshold': 160,
    'threshold_step': 5,
    
    # 面积过滤参数
    'min_area': 100,
    'max_area': 200,        # 优化后: 从2000降低到200
    
    # 形状过滤参数
    'min_circularity': 0.8, # 优化后: 从0.6提高到0.8
    'min_convexity': 0.85,
    'min_inertia_ratio': 0.3,
    
    # 颜色过滤参数
    'filter_by_color': True,
    'blob_color': 0,        # 0=黑色标记点
    
    # 图像预处理参数
    'clahe_clip_limit': 2.0,
    'clahe_tile_size': 8,
}

# 参数说明
PARAM_DESCRIPTIONS = {
    'min_area': '最小面积 - 过滤太小的噪点',
    'max_area': '最大面积 - 过滤太大的区域',
    'min_circularity': '最小圆度 - 确保检测圆形标记点 (0-1)',
    'min_convexity': '最小凸度 - 确保形状规整 (0-1)',
    'min_inertia_ratio': '最小惯性率 - 椭圆度要求 (0-1)',
    'min_threshold': '最小阈值 - 二值化起始值',
    'max_threshold': '最大阈值 - 二值化结束值',
    'threshold_step': '阈值步长 - 二值化精度',
}

# 优化历史记录
OPTIMIZATION_HISTORY = [
    {
        'date': '2025-01-18',
        'changes': {
            'max_area': {'from': 2000, 'to': 200, 'reason': '减少误检测大区域'},
            'min_circularity': {'from': 0.6, 'to': 0.8, 'reason': '提高圆形要求精度'}
        },
        'result': '检测精度提升，误检测减少'
    }
]

def get_optimized_params():
    """获取优化后的参数字典"""
    return OPTIMIZED_BLOB_PARAMS.copy()

def print_current_params():
    """打印当前参数设置"""
    print("🔧 当前优化参数:")
    print("=" * 40)
    
    for key, value in OPTIMIZED_BLOB_PARAMS.items():
        description = PARAM_DESCRIPTIONS.get(key, "")
        print(f"{key:20}: {value:8} - {description}")
    
    print("\n📈 优化历史:")
    for record in OPTIMIZATION_HISTORY:
        print(f"日期: {record['date']}")
        for param, change in record['changes'].items():
            print(f"  {param}: {change['from']} → {change['to']} ({change['reason']})")
        print(f"  结果: {record['result']}")

def save_params_to_file(params, filename="optimized_params.txt"):
    """保存参数到文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("# 标记点检测优化参数\n")
        f.write(f"# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for key, value in params.items():
            description = PARAM_DESCRIPTIONS.get(key, "")
            f.write(f"{key} = {value}  # {description}\n")
    
    print(f"参数已保存到: {filename}")

if __name__ == "__main__":
    import datetime
    print_current_params()
