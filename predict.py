from ultralytics import YOLO
import cv2
import numpy as np

def predict_hand_keypoints(image_path):
    """使用训练好的模型预测手部关键点"""
    
    # 加载训练好的模型
    model = YOLO("runs/pose/hand_keypoints_optimized_v23/weights/best.pt")
    
    # 进行预测
    results = model(image_path)
    
    # 处理结果
    for result in results:
        # 获取关键点
        if result.keypoints is not None:
            keypoints = result.keypoints.data[0]  # 第一只手的关键点
            
            # 将CUDA张量转换到CPU并转为numpy数组
            if hasattr(keypoints, 'cpu'):
                keypoints = keypoints.cpu().numpy()
            
            print(f"检测到 {len(keypoints)} 个关键点")
            
            # 手部21个关键点的名称（按MediaPipe标准）
            keypoint_names = [
                "WRIST",           # 0: 手腕
                "THUMB_CMC",       # 1: 拇指掌指关节
                "THUMB_MCP",       # 2: 拇指近端指间关节
                "THUMB_IP",        # 3: 拇指远端指间关节
                "THUMB_TIP",       # 4: 拇指指尖
                "INDEX_FINGER_MCP", # 5: 食指掌指关节
                "INDEX_FINGER_PIP", # 6: 食指近端指间关节
                "INDEX_FINGER_DIP", # 7: 食指远端指间关节
                "INDEX_FINGER_TIP", # 8: 食指指尖
                "MIDDLE_FINGER_MCP", # 9: 中指掌指关节
                "MIDDLE_FINGER_PIP", # 10: 中指近端指间关节
                "MIDDLE_FINGER_DIP", # 11: 中指远端指间关节
                "MIDDLE_FINGER_TIP", # 12: 中指指尖
                "RING_FINGER_MCP",   # 13: 无名指掌指关节
                "RING_FINGER_PIP",   # 14: 无名指近端指间关节
                "RING_FINGER_DIP",   # 15: 无名指远端指间关节
                "RING_FINGER_TIP",   # 16: 无名指指尖
                "PINKY_MCP",         # 17: 小指掌指关节
                "PINKY_PIP",         # 18: 小指近端指间关节
                "PINKY_DIP",         # 19: 小指远端指间关节
                "PINKY_TIP"          # 20: 小指指尖
            ]
            
            # 打印每个关键点的坐标和置信度
            for i, (x, y, conf) in enumerate(keypoints):
                print(f"{keypoint_names[i]:<20}: x={x:.1f}, y={y:.1f}, 置信度={conf:.3f}")
            
            return keypoints
        else:
            print("未检测到手部关键点")
            return None

def find_wrist_center(keypoints):
    """找到手腕中心点（通常是所有手指的起点）"""
    # 确保keypoints是numpy数组
    if not isinstance(keypoints, np.ndarray):
        keypoints = np.array(keypoints)
    
    # 找到Y坐标最大的点（最下方的点）
    wrist_idx = np.argmax([kp[1] for kp in keypoints])
    return wrist_idx

def find_finger_tips(keypoints):
    """找到每个手指的指尖"""
    # 确保keypoints是numpy数组
    if not isinstance(keypoints, np.ndarray):
        keypoints = np.array(keypoints)
    
    # 找到Y坐标最小的几个点（最上方的点）
    y_coords = [kp[1] for kp in keypoints]
    tip_indices = np.argsort(y_coords)[:5]  # 取Y坐标最小的5个点
    return tip_indices

def smart_connect_keypoints(keypoints):
    """智能连接关键点，基于距离和位置关系"""
    connections = []
    
    # 确保keypoints是numpy数组
    if not isinstance(keypoints, np.ndarray):
        keypoints = np.array(keypoints)
    
    # 找到手腕中心
    wrist_idx = find_wrist_center(keypoints)
    print(f"检测到手腕中心点: {wrist_idx}")
    
    # 找到指尖
    tip_indices = find_finger_tips(keypoints)
    print(f"检测到指尖点: {tip_indices}")
    
    # 为每个指尖找到到手腕的路径
    for tip_idx in tip_indices:
        if tip_idx == wrist_idx:
            continue
            
        # 从指尖开始，逐步找到到手腕的路径
        current_idx = tip_idx
        path = [current_idx]
        
        while current_idx != wrist_idx:
            # 找到距离当前点最近且Y坐标更大的点
            current_pos = keypoints[current_idx][:2]
            candidates = []
            
            for i, kp in enumerate(keypoints):
                if i != current_idx and i not in path:
                    # 计算距离
                    dist = np.linalg.norm(np.array(current_pos) - np.array(kp[:2]))
                    # 优先选择Y坐标更大的点（更靠近手腕）
                    y_priority = kp[1] - current_pos[1]
                    
                    if y_priority > 0:  # 只考虑更靠近手腕的点
                        candidates.append((i, dist, y_priority))
            
            if not candidates:
                break
                
            # 选择最佳候选点（距离近且Y坐标大）
            candidates.sort(key=lambda x: (x[1], -x[2]))
            next_idx = candidates[0][0]
            
            if next_idx not in path:
                path.append(next_idx)
                current_idx = next_idx
            else:
                break
        
        # 添加路径连接
        for i in range(len(path) - 1):
            connections.append((path[i], path[i+1]))
    
    return connections

def visualize_keypoints(image_path, keypoints):
    """可视化关键点"""
    if keypoints is None:
        return
    
    # 确保keypoints是numpy数组
    if not isinstance(keypoints, np.ndarray):
        keypoints = np.array(keypoints)
    
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图片: {image_path}")
        return
    
    # 获取原始图片尺寸
    original_height, original_width = image.shape[:2]
    print(f"原始图片尺寸: {original_width}x{original_height}")
    
    # 设置目标显示尺寸（保持宽高比）
    target_width = 800  # 目标宽度
    target_height = int(target_width * original_height / original_width)
    
    # 如果高度太大，则以高度为基准
    if target_height > 600:
        target_height = 600
        target_width = int(target_height * original_width / original_height)
    
    print(f"缩放后尺寸: {target_width}x{target_height}")
    
    # 等比例缩放图片
    image_resized = cv2.resize(image, (target_width, target_height))
    
    # 计算缩放比例
    scale_x = target_width / original_width
    scale_y = target_height / original_height
    
    # 绘制关键点（需要根据缩放比例调整坐标）
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > 0.3:  # 降低置信度阈值，显示更多关键点
            # 根据置信度调整颜色
            if conf > 0.8:
                color = (0, 255, 0)  # 绿色：高置信度
            elif conf > 0.5:
                color = (0, 255, 255)  # 黄色：中等置信度
            else:
                color = (0, 165, 255)  # 橙色：低置信度
            
            # 缩放坐标
            scaled_x = int(x * scale_x)
            scaled_y = int(y * scale_y)
            
            # 确保坐标在图片范围内
            scaled_x = max(0, min(target_width - 1, scaled_x))
            scaled_y = max(0, min(target_height - 1, scaled_y))
            
            # 绘制关键点圆圈
            cv2.circle(image_resized, (scaled_x, scaled_y), 6, color, -1)
            
            # 绘制关键点编号
            cv2.putText(image_resized, str(i), (scaled_x + 8, scaled_y - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # 暂时不画连接线，只显示关键点
    print("只显示关键点，未绘制连接线")
    
    # 保存结果
    output_path = "hand_keypoints_result.jpg"
    cv2.imwrite(output_path, image_resized)
    print(f"可视化结果已保存到: {output_path}")
    
    # 显示图片
    cv2.imshow("Hand Keypoints (Scaled)", image_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def debug_keypoints(keypoints):
    """调试关键点数据"""
    print("\n=== 关键点调试信息 ===")
    print(f"关键点总数: {len(keypoints)}")
    
    # 确保keypoints是numpy数组
    if not isinstance(keypoints, np.ndarray):
        keypoints = np.array(keypoints)
    
    # 检查坐标范围
    x_coords = [kp[0] for kp in keypoints]
    y_coords = [kp[1] for kp in keypoints]
    conf_scores = [kp[2] for kp in keypoints]
    
    print(f"X坐标范围: {min(x_coords):.1f} - {max(x_coords):.1f}")
    print(f"Y坐标范围: {min(y_coords):.1f} - {max(y_coords):.1f}")
    print(f"置信度范围: {min(conf_scores):.3f} - {max(conf_scores):.3f}")
    
    # 检查是否有异常值
    for i, (x, y, conf) in enumerate(keypoints):
        if x < 0 or y < 0 or conf < 0 or conf > 1:
            print(f"警告: 关键点 {i} 有异常值: x={x}, y={y}, conf={conf}")

if __name__ == "__main__":
    # 使用训练集中实际存在的图片
    image_path = "datasets/hand-keypoints/images/train/IMG_00001070.jpg"  # 使用存在的图片
    # image_path = "D:/dataset/roi/rpg/img/00150.jpg"  # 请修改为实际图片路径
    # image_path = "D:/dataset/roi/Shanghai/500w/71/b_20250422101522019143.png"  
    print("开始预测手部关键点...")
    keypoints = predict_hand_keypoints(image_path)
    
    if keypoints is not None:
        print("\n关键点预测完成！")
        
        # 调试关键点数据
        debug_keypoints(keypoints)
        
        # 可视化
        visualize_keypoints(image_path, keypoints)
    else:
        print("预测失败，请检查图片或模型")
