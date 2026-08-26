import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseStamped
from scipy.spatial.transform import Rotation as R


import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# 这里我直接用系统环境的cuda跑，colcon build的时候记得不要编译这个ros包，
# 所以那些ros的参数声明我也只是装模作样写了，但是不能在ros2 run 传参，直接用python3 跑就行
# ============================================================================

class Model(nn.Module):
    def __init__(self, dim_obs=6, dim_action=2, hidden_dim=192, input_w=32, input_h=24):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2, stride=2, bias=False),  
            nn.LeakyReLU(0.05),
            nn.Conv2d(32, 64, kernel_size=3, bias=False), 
            nn.LeakyReLU(0.05),
            nn.Conv2d(64, 128, kernel_size=3, bias=False), 
            nn.LeakyReLU(0.05),
            nn.Flatten()
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_h, input_w)
            out = self.conv_net(dummy)
            self.conv_out_dim = out.shape[1]
        self.fc_visual = nn.Linear(self.conv_out_dim, 192, bias=False)
        self.fc_state = nn.Linear(dim_obs, 64)
        self.fc_state.weight.data.mul_(0.5)
        self.gru = nn.GRUCell(192 + 64, hidden_dim)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LeakyReLU(0.05),
            nn.Linear(128, dim_action),
            nn.Tanh() 
        )
        self.action_head[2].weight.data.mul_(0.01)
        self.max_v = 5.0      
        self.max_omega = 3.0 

    def reset(self):
        pass

    def forward(self, depth, state, h=None):
        B = depth.size(0)
        visual_feat = self.fc_visual(self.conv_net(depth))
        state_feat = F.leaky_relu(self.fc_state(state), 0.05)
        fusion = torch.cat([visual_feat, state_feat], dim=1)
        if h is None:
            h = torch.zeros(B, self.gru.hidden_size, device=depth.device, dtype=depth.dtype)
        h_new = self.gru(fusion, h)
        raw_action = self.action_head(h_new)
        v = (raw_action[:, 0:1] + 1.0) / 2.0 * self.max_v 
        omega = raw_action[:, 1:2] * self.max_omega
        action = torch.cat([v, omega], dim=1)
        return action, h_new
    

class WheelbotInferenceNode(Node):
    def __init__(self):
        super().__init__('wheelbot_inference_node')
        
        # --- 参数声明 ---
        self.declare_parameter('model_path', 'save/checkpoint_base00_30000.pth') 
        self.declare_parameter('device', 'cuda') 
        
        # --- 变量初始化 ---d
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.device_name = self.get_parameter('device').get_parameter_value().string_value
        self.device = torch.device(self.device_name if torch.cuda.is_available() else 'cpu')
        
        self.get_logger().info(f"Using device: {self.device}")
        
        # 状态变量
        self.current_pose = None # [x, y, yaw]
        self.current_vel = 0.0   
        self.goal_pose = [6.0, -2.0]  
        self.gru_h = None
        
        # CV Bridge
        self.bridge = CvBridge()
        self.load_model()

        # --- QoS 设置
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        # TODO: 后续改成参数声明形式，论文写完再改
        # --- ROS2 Sub ---

        # 使用fist_lio2的里程计
        # self.odom_sub = self.create_subscription(Odometry, "/Odometry", self.odom_callback, qos_sensor)  

        # 使用小车自带的里程计，轮+imu kalman融合的，感觉不太稳，先用fist_lio2的里程计，后续有时间再改
        self.odom_sub = self.create_subscription(Odometry, "/odom", self.odom_callback, qos_sensor)
     
        self.depth_sub = self.create_subscription(Image, "/camera/depth/image_rect_raw", self.depth_callback, qos_sensor)
        self.goal_sub = self.create_subscription(PoseStamped, "/goal_pose", self.goal_callback, 1)

        # 2. ROS2 Pub ---
        self.ctrl_pub = self.create_publisher(Twist, "/cmd_vel", 1)
        
        self.get_logger().info("Wheelbot Inference Node Initialized.")

        # self.last_omega = 0.0
        # 平滑系数 (0.0~1.0)
        # self.smooth_alpha = 0.8

    def load_model(self):
        try:
            self.model = Model(dim_obs=6, dim_action=2, input_w=32, input_h=24).to(self.device)
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            self.model.eval()
            self.get_logger().info(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")

    def odom_callback(self, msg: Odometry):

        # get position now from odom
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        
        # (Quaternion -> Yaw)
        q = msg.pose.pose.orientation
        rot = R.from_quat([q.x, q.y, q.z, q.w])
        yaw = rot.as_euler('zxy')[0] 
        
        self.current_pose = np.array([x, y, yaw])
        
        # get linear velocity
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.current_vel = np.sqrt(vx**2 + vy**2)

    def goal_callback(self, msg: PoseStamped):
        x = msg.pose.position.x
        y = msg.pose.position.y
        new_goal = np.array([x, y])

        #  过滤重复目标逻辑
        if self.goal_pose is not None:
            old_goal = np.array(self.goal_pose)
            dist = np.linalg.norm(new_goal - old_goal)
            if dist < 1.0:
                return
            
        self.goal_pose = new_goal
        self.get_logger().info(f"New goal received: [{x:.2f}, {y:.2f}]")
    
        # 发现bug了，每次新目标到来时要重置GRU的隐藏状态，而模拟器会一1hz发布目标，我草了，破案了 
        self.gru_h = None

    def depth_callback(self, msg: Image):
        """
        核心推理循环：图像处理 -> 状态构建 -> 模型推理 -> 发布控制
        """
        # 1. check pose and goal,if not exist then return
        if self.current_pose is None or self.goal_pose is None:
            return # 等待 odom 和 goal

        # 2. 图像预处理
        try:
            # 兼容 16UC1 (mm) 和 32FC1 (m)
            if "16UC" in msg.encoding:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                cv_image = cv_image.copy() 
                cv_image[cv_image == 0] = 10000 
                
                depth_in_meters = cv_image.astype(np.float32) / 1000.0
                
            elif "32FC" in msg.encoding:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                depth_in_meters = cv_image.copy()
                depth_in_meters = np.nan_to_num(depth_in_meters, nan=10.0, posinf=10.0, neginf=0.1)
                
            else:
                self.get_logger().error(f"Unsupported encoding: {msg.encoding}")
                return

        except Exception as e:
            self.get_logger().error(f"CV Bridge Error: {e}")
            return

        input_h, input_w = 24, 32
        depth_resized = cv2.resize(depth_in_meters, (input_w, input_h), interpolation=cv2.INTER_AREA)
        depth_tensor = torch.from_numpy(depth_resized).float().to(self.device)
        # 维度匹配 (B, C, H, W) （1, 1, 24, 32）
        depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0)
        depth_input = 3.0 / depth_tensor.clamp(0.2, 10.0) -0.6

        # 1. Version 1 作者版本，先取倒数去掉
        # inv_depth_tensor = 1.0 / (depth_tensor + 1e-5)
        # depth_input = 3.0 / depth_tensor.clamp(0.1, 10.0) - 0.5
        # # depth_input = inv_depth_tensor.clamp(0.1, 10.0) * 0.5 - 0.2

        # # 线性版本，用于高速
        # max_sensor_dist = 10.0
        # depth_input = inv_depth_tensor.clamp(min=0.1,max=10.0)
        # true_dist = 1.0 / depth_input

        # depth_linear = (max_sensor_dist - true_dist) / max_sensor_dist
        # depth_linear = depth_input.clamp(0.0, 1.0)
        # noise = torch.randn_like(depth_linear) * 0.05
        # depth_input = depth_linear + noise

        # 3. 状态向量构建
        dx = self.goal_pose[0] - self.current_pose[0]
        dy = self.goal_pose[1] - self.current_pose[1]
        
        yaw = self.current_pose[2]
        cos_th = np.cos(yaw)
        sin_th = np.sin(yaw)
        
        # global frame to local frame
        local_x = dx * cos_th + dy * sin_th
        local_y = dx * (-sin_th) + dy * cos_th

        dist_target = np.sqrt(local_x**2 + local_y**2)

        # 约束一下输入范围,面得因为地图太大跟训练环境不一样导致神经网络发癫
        # local_x = np.clip(local_x, -10.0, 10.0)
        # local_y = np.clip(local_y, -10.0, 10.0)
        # dist_target = np.clip(dist_target, 0.0, 10.0)

        dist_target = np.sqrt(local_x**2 + local_y**2)

        # --- 解决小车摇头的关键代码，这个要记一辈子哈 ---
        # 逻辑：如果距离超过 10米，缩放到 10米，但方向不变
        max_dist = 10.0
        if dist_target > max_dist:
            scale = max_dist / dist_target
            local_x = local_x * scale
            local_y = local_y * scale
            dist_target = max_dist
        
        if dist_target < 1.0:
            self.publish_stop()
            # self.goal_pose = None # 保持目标，或者等待新目标
            return
        
        # 这里的local_x, local_y因为训练的地图尺寸是10m x 10m，所以除以10归一化,看看效果，后面有时间再改
        state_list = [
            local_x / 10.0,
            local_y / 10.0,
            cos_th,
            sin_th,
            dist_target / 10.0,
            self.current_vel
        ]
        
        state_tensor = torch.tensor(state_list, dtype=torch.float32, device=self.device).unsqueeze(0) # (1, 6)

        # 4. 模型推理
        with torch.no_grad():
            # action, _, self.gru_h = self.model(depth_input, state_tensor, self.gru_h)

            # 去掉强化学习的value了,后面用到再说，目前看没必要
            action, self.gru_h = self.model(depth_input, state_tensor, self.gru_h)
            
            # 提取动作
            v_cmd = action[0, 0].item()
            w_cmd = action[0, 1].item()

            # # 角速度低通滤波处理，目前看没有必要
            # filtered_w = self.smooth_alpha * self.last_omega + (1.0 - self.smooth_alpha) * w_cmd
            # self.last_omega = filtered_w 

        # 5. 发布控制
        twist = Twist()
        twist.linear.x = float(v_cmd)
        # twist.angular.z = float(filtered_w)
        twist.angular.z = float(w_cmd)
        self.ctrl_pub.publish(twist)

    def publish_stop(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.ctrl_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = WheelbotInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()