from ultralytics import YOLO

def main():
    # Load a model (可改为 yolo11x-pose.pt 进行更高精度训练)
    model = YOLO("yolo11n-pose.pt")

    # Train the model with data validation
    results = model.train(
        data="hand-keypoints.yaml",
        epochs=200,                 # 更长训练配合早停提升泛化
        imgsz=640,                 # 关键点建议640或更高
        batch=24,                 # 略降batch缓解内存压力
        workers=0,                # Windows下避免多进程引发内存复制
        device=0,

        # 学习率与调度（防欠拟合/过拟合）
        lr0=0.01,                 # 初始LR（可在0.005~0.01间网格搜索）
        lrf=0.01,                 # 最终LR比例
        cos_lr=True,              # 余弦退火
        patience=50,              # 早停耐心

        # 强化通用化的数据增强（对旋转鲁棒）
        degrees=25.0,             # 旋转增强，解决旋转不准
        translate=0.10,
        scale=0.25,
        shear=2.0,
        fliplr=0.5,
        flipud=0.0,
        mosaic=1.0,               # 启用 mosaic
        mixup=0.10,               # 少量 mixup 提升泛化
        close_mosaic=25,          # 训练后期关闭 mosaic，专注细节

        # 采样与性能
        rect=True,                # 按长宽比采样（会关闭shuffle属正常提示）
        cache='disk',             # 改为磁盘缓存，避免10GB+ RAM占用

        # 输出与复现
        save_period=10,
        project="runs/pose",
        name="train_generalized",
        seed=42,
        verbose=True,
        val=True,
        resume=False
    )

if __name__ == '__main__':
    main()