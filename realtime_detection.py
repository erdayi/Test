import cv2
import numpy as np
from ultralytics import YOLO
import time

class RealtimeHandKeypointDetector:
    def __init__(self, model_path="runs/pose/train3/weights/best.pt"):
        """初始化实时手部关键点检测器"""
        self.model = YOLO(model_path)
        self.cap = None
        
        # 手部21个关键点的名称
        self.keypoint_names = [
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
        
        # 手部骨架连接 - 使用正确的MediaPipe标准
        self.connections = [
            # 拇指：手腕 -> 拇指掌指 -> 近端指间 -> 远端指间 -> 指尖
            (0, 1), (1, 2), (2, 3), (3, 4),
            
            # 食指：手腕 -> 食指掌指 -> 近端指间 -> 远端指间 -> 指尖
            (0, 5), (5, 6), (6, 7), (7, 8),
            
            # 中指：手腕 -> 中指掌指 -> 近端指间 -> 远端指间 -> 指尖
            (0, 9), (9, 10), (10, 11), (11, 12),
            
            # 无名指：手腕 -> 无名指掌指 -> 近端指间 -> 远端指间 -> 指尖
            (0, 13), (13, 14), (14, 15), (15, 16),
            
            # 小指：手腕 -> 小指掌指 -> 近端指间 -> 远端指间 -> 指尖
            (0, 17), (17, 18), (18, 19), (19, 20),
        ]
        
        print("手部关键点检测器初始化完成！")
        print("按 'q' 退出，按 's' 保存当前帧")
    
    def start_camera(self, camera_id=0):
        """启动摄像头"""
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            print(f"无法打开摄像头 {camera_id}")
            return False
        
        print(f"摄像头 {camera_id} 启动成功")
        return True
    
    def detect_keypoints(self, frame):
        """检测手部关键点"""
        try:
            # 使用模型进行预测
            results = self.model(frame, verbose=False)  # 关闭详细输出
            
            for result in results:
                if result.keypoints is not None:
                    keypoints = result.keypoints.data[0]  # 第一只手的关键点
                    
                    # 将CUDA张量转换到CPU并转为numpy数组
                    if hasattr(keypoints, 'cpu'):
                        keypoints = keypoints.cpu().numpy()
                    
                    return keypoints
            
            return None
        except Exception as e:
            print(f"检测过程中出错: {e}")
            return None
    
    def draw_keypoints(self, frame, keypoints):
        """在帧上绘制关键点和骨架"""
        if keypoints is None:
            return frame
        
        # 绘制关键点
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > 0.3:  # 只绘制置信度高的关键点
                # 根据置信度调整颜色
                if conf > 0.8:
                    color = (0, 255, 0)  # 绿色：高置信度
                elif conf > 0.5:
                    color = (0, 255, 255)  # 黄色：中等置信度
                else:
                    color = (0, 165, 255)  # 橙色：低置信度
                
                # 绘制关键点圆圈
                cv2.circle(frame, (int(x), int(y)), 6, color, -1)
                
                # 绘制关键点编号
                cv2.putText(frame, str(i), (int(x)+8, int(y)-8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        # 绘制骨架连接线
        for start_idx, end_idx in self.connections:
            if (keypoints[start_idx][2] > 0.3 and keypoints[end_idx][2] > 0.3):
                start_point = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                end_point = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                cv2.line(frame, start_point, end_point, (0, 0, 255), 2)
        
        return frame
    
    def draw_info_panel(self, frame, fps, keypoints_count):
        """绘制信息面板"""
        # 绘制半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 绘制信息文本
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def run_realtime(self):
        """运行实时检测"""
        if not self.start_camera():
            return
        
        show_numbers = True  # 控制是否显示关键点编号
        frame_count = 0
        start_time = time.time()
        fps = 0  # 初始化FPS
        
        print("开始实时检测...")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("无法读取摄像头帧")
                break
            
            # 检测关键点
            keypoints = self.detect_keypoints(frame)
            
            # 绘制关键点和骨架
            if keypoints is not None:
                frame = self.draw_keypoints(frame, keypoints)
                keypoints_count = len([kp for kp in keypoints if kp[2] > 0.3])
            else:
                keypoints_count = 0
            
            # 计算FPS - 修复计算逻辑
            frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            if elapsed_time >= 1.0:  # 每秒更新一次FPS
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = current_time
            
            # 绘制信息面板
            self.draw_info_panel(frame, fps, keypoints_count)
            
            # 显示结果
            cv2.imshow("实时手部关键点检测", frame)
            
            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("用户退出")
                break
            elif key == ord('s'):
                # 保存当前帧
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"hand_detection_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"已保存图片: {filename}")
            elif key == ord('h'):
                # 切换显示/隐藏编号
                show_numbers = not show_numbers
                print(f"关键点编号显示: {'开启' if show_numbers else '关闭'}")
        
        # 清理资源
        self.cap.release()
        cv2.destroyAllWindows()
        print("实时检测结束")

def main():
    """主函数"""
    print("=== 实时手部关键点检测 ===")
    print("使用训练好的模型进行实时检测")
    
    # 创建检测器
    detector = RealtimeHandKeypointDetector()
    
    # 运行实时检测
    detector.run_realtime()

if __name__ == "__main__":
    main()
