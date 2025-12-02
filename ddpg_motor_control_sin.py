#ddpg_compare_pid.py (Modified for Sinusoidal Tracking)
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------- Motor ----------
class DifferentiableMotor:
    def __init__(self, K=10, tau=0.5, zeta=0.7, Ts=0.01):
        self.Ts = Ts; self.K = K; self.tau = tau; self.zeta = zeta
        self.reset()
    def reset(self):
        self.omega = 0.0
        self.omega_prev = 0.0
    def step(self, u):
        a1 = 1 + 2 * (-self.zeta * self.Ts / self.tau)
        a2 = - (self.Ts / self.tau)**2
        b1 = self.K * self.Ts**2 / (self.tau**2)
        omega_next = a1 * self.omega + a2 * self.omega_prev + b1 * u
        self.omega_prev = self.omega
        self.omega = float(omega_next)
        return self.omega

# ---------- Actor ----------
class Actor(nn.Module):
    # 确保网络结构和 action_max 与训练脚本一致
    def __init__(self, state_dim=4, action_max=24.0):
        super().__init__()
        hidden = 256
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden//2), nn.ReLU(),
            nn.Linear(hidden//2, 1), nn.Tanh()
        )
        self.action_max = action_max
    def forward(self, s):
        return self.net(s) * self.action_max

# ---------- PID ----------
class PIDController:
    # 确保 u_max 与 Actor 的 action_max 一致
    def __init__(self, Kp, Ki, Kd, Ts, u_max=24.0): 
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.Ts = Ts
        self.u_max = u_max
        self.e_prev = 0.0
        self.e_sum = 0.0
    def compute(self, error):
        self.e_sum += error * self.Ts
        dedt = (error - self.e_prev)/self.Ts
        u = self.Kp*error + self.Ki*self.e_sum + self.Kd*dedt
        self.e_prev = error
        return float(np.clip(u, -self.u_max, self.u_max))

# ---------- 比較主函數 ----------
def compare_ddpg_pid(actor_path="ddpg_actor_stable_sin.pth",
                     motor_params={"K":10,"tau":0.5,"zeta":0.7},
                     Ts=0.01,
                     steps=10000):
    
    # --- 正弦波目标追踪参数 (需与训练脚本一致) ---
    T_cycle = 10.0
    omega_freq = 2 * math.pi / T_cycle
    amplitude = 1.0  
    offset = 0.0   
    
    # 初始化
    motor_pid = DifferentiableMotor(**motor_params, Ts=Ts)
    motor_ddpg = DifferentiableMotor(**motor_params, Ts=Ts)
    
    # PID 参数可以保持不变，或针对追踪任务进行调优
    #pid = PIDController(Kp=8.9, Ki=2.0, Kd=4.3, Ts=Ts)
    pid = PIDController(Kp=9.0, Ki=2.6, Kd=2.2, Ts=Ts)
    
    # actor
    state_dim = 4
    actor = Actor(state_dim).to(device)
    try:
        actor.load_state_dict(torch.load(actor_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Actor model file '{actor_path}' not found. Please check the path and file name.")
        return
    
    actor.eval()
    
    # 记录
    speed_pid = [0.0] 
    speed_ddpg = [0.0]
    force_pid = []
    force_ddpg = []
    target_speeds_log = [] 
    time_points = [0.0]
    
    # DDPG 状态追踪初始化 (t=0)
    initial_target_speed = float(amplitude * math.sin(0) + offset) 
    initial_target_speed = max(0.1, initial_target_speed) 
    prev_error_ddpg = initial_target_speed - motor_ddpg.omega 

    
    error_sum = 0.0  # 累积误差 I 项 (积分项)
    # 推论
    with torch.no_grad():
        for k in range(1, steps):
            current_time = k * Ts
            time_points.append(current_time)

            # *** 目标速度动态更新 ***
            target_speed = float(amplitude * math.sin(omega_freq * current_time) + offset)
            #target_speed = 1.0 #pulse
            target_speed = max(0.1, target_speed) 
            target_speed=abs(target_speed)
            target_speeds_log.append(target_speed)
            
            # --- PID ---
            error_pid = target_speed - speed_pid[-1]
            u_pid = pid.compute(error_pid)
            force_pid.append(u_pid)
            omega_pid = motor_pid.step(u_pid)
            speed_pid.append(omega_pid)
            
            # --- DDPG (修正状态输入) ---
            current_speed = speed_ddpg[-1]
            
            # 1. 计算当前误差和误差变化量
            current_error_ddpg = target_speed - current_speed
            error_sum += current_error_ddpg * Ts # 累积积分误差
            delta_error_ddpg = current_error_ddpg - prev_error_ddpg
            
            # 2. 构建状态张量: [omega, target_speed, delta_error,error_sum]
            state_tensor = torch.tensor([[current_speed, target_speed, delta_error_ddpg, error_sum]], 
                                        dtype=torch.float32, device=device)
            
            # 3. 得到控制量 u_ddpg
            u_ddpg = actor(state_tensor).item()
            force_ddpg.append(u_ddpg)
            omega_ddpg = motor_ddpg.step(u_ddpg)
            speed_ddpg.append(omega_ddpg)
            
            # 4. 更新 prev_error_ddpg
            prev_error_ddpg = current_error_ddpg
    
    # 移除初始零点
    speed_pid = speed_pid[1:]
    speed_ddpg = speed_ddpg[1:]
    time_points = time_points[1:]
    
    # 绘图
    plt.figure(figsize=(12,10))
    
    # 速度曲线
    plt.subplot(2,1,1)
    plt.plot(time_points, target_speeds_log, label="Target Speed (Sin Wave)", color="green", linestyle="--")
    plt.plot(time_points, speed_pid, label="PID Controller Speed", color="blue", alpha=0.7)
    plt.plot(time_points, speed_ddpg, label="DDPG Actor Speed", color="red")
    plt.ylabel("Angular Speed [rad/s]")
    plt.title(f"DDPG vs PID Sinusoidal Tracking (T={T_cycle}s, Params: {motor_params})")
    plt.grid(True)
    plt.legend()
    
    # 控制量 u 曲线
    plt.subplot(2,1,2)
    plt.plot(time_points, force_pid, label="PID Control Input (u)", color="blue")
    plt.plot(time_points, force_ddpg, label="DDPG Control Input (u)", color="red")
    plt.xlabel("Time [s]")
    plt.ylabel("Control Input (Voltage) [V]")
    plt.grid(True)
    plt.legend()
    
    plt.show()
    
    # 计算跟踪性能指标 (Mean Squared Error, MSE)
    def metrics_tracking(speed_arr, target_arr):
        # 排除初始暂态，例如只计算后 90% 的数据
        start_idx = len(speed_arr) // 10 
        mse = np.mean((speed_arr[start_idx:] - target_arr[start_idx:])**2)
        return mse

    target_arr = np.array(target_speeds_log)
    mse_pid = metrics_tracking(np.array(speed_pid), target_arr)
    mse_ddpg = metrics_tracking(np.array(speed_ddpg), target_arr)
    
    print("-" * 30)
    print(f"Tracking Metrics (T={T_cycle}s, Ts: {Ts})")
    print(f"PID  → Mean Squared Error (MSE, last 90%)={mse_pid:.6f}")
    print(f"DDPG → Mean Squared Error (MSE, last 90%)={mse_ddpg:.6f}")
    print("-" * 30)

# ---------- 執行 ----------
if __name__ == "__main__":
    compare_ddpg_pid(actor_path="ddpg_actor_stable_sin.pth", # 注意文件名已更改
                     motor_params={"K":10,"tau":0.5,"zeta":0.7},
                     Ts=0.01,
                     steps=10000)