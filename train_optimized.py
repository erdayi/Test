from ultralytics import YOLO
import os

def main():
    """从现有最佳权重开始新的优化训练手部关键点检测模型"""
    
    # 检查现有模型是否存在
    # existing_model_path = "runs/pose/train3/weights/best.pt"
    existing_model_path = ""
    if os.path.exists(existing_model_path):
        print(f"找到现有模型: {existing_model_path}")
        print("将从现有最佳权重开始新的训练...")
        model = YOLO(existing_model_path)
    else:
        print("未找到现有模型，将从头开始训练...")
        model = YOLO("yolo11n-pose.pt")
    
    # 优化训练参数：提高收敛速度与精度，合理的输入尺寸与学习率，更稳定的数据策略
    results = model.train(
        data="hand-keypoints.yaml",
        epochs=150,                     # 轮数提升，给模型足够收敛时间
        imgsz=640,                      # 提升输入分辨率以提高关键点定位精度
        batch=32,                       # 略小batch以稳定训练
        workers=4,                      # 开多进程加载数据（Windows已加main保护）
        device=0,                       # 指定GPU 0
        
        # 学习率优化 - 精细调优
        lr0=0.01,                      # 初始学习率过小可能欠拟合，使用默认量级
        lrf=0.01,                      # 末尾学习率比例
        momentum=0.937,
        weight_decay=0.0005,
        
        # 数据增强优化 - 保持关键点准确性
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=7.5,                   # 适度旋转增强
        translate=0.08,
        scale=0.20,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        
        # 训练策略
        save_period=10,
        patience=50,
        val=True,
        plots=True,
        
        # 性能优化
        cache=True,                    # 启用RAM缓存，加速IO
        rect=True,                     # 按长宽比采样，提升收敛稳定性
        cos_lr=True,
        close_mosaic=15,               # 训练后期关闭mosaic，利于关键点精细化
        optimizer='AdamW',             # 对关键点任务更稳一些
        seed=42,                       # 结果更可复现
        
        # 输出
        verbose=True,
        project="runs/pose",
        name="hand_keypoints_optimized_v2",
        
        resume=False
    )
    
    print("新的优化训练完成！")
    
    # 保存最终模型 - 使用更清晰的命名
    final_model_name = "hand_keypoints_optimized_v2_final.pt"
    model.save(final_model_name)
    print(f"模型已保存为: {final_model_name}")
    
    # 同时保存一个带时间戳的版本
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_name = f"hand_keypoints_optimized_v2_{timestamp}.pt"
    model.save(timestamped_name)
    print(f"带时间戳的模型已保存为: {timestamped_name}")

if __name__ == '__main__':
    main()
