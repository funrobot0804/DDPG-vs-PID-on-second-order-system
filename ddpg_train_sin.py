# ddpg_train_stable.py (Modified for Sinusoidal Target Tracking)
import random, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import matplotlib.pyplot as plt

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------- Motor ----------
#這是一個離散化的二階系統（差分形式）用來模擬馬達動態，參數：
#K：增益（輸入到輸出）
#tau：時間常數，影響系統反應速度
#zeta：阻尼比
#Ts：時間步長（秒）
class DifferentiableMotor:
    def __init__(self, K=10, tau=0.5, zeta=0.7, Ts=0.01):
        self.Ts = Ts; self.K = K; self.tau = tau; self.zeta = zeta
        self.reset()
    def reset(self):
        self.omega = 0.0
        self.omega_prev = 0.0
    def step(self, u):
        # u: control input (voltage)
        a1 = 1 + 2 * (-self.zeta * self.Ts / self.tau)
        a2 = - (self.Ts / self.tau)**2
        b1 = self.K * self.Ts**2 / (self.tau**2)
        omega_next = a1 * self.omega + a2 * self.omega_prev + b1 * u
        self.omega_prev = self.omega
        self.omega = float(omega_next)
        return self.omega

# ---------- Replay Buffer ----------
#儲存經驗 (state, action, reward, next_state, done)，用 uniform random sample。
#回放大小、batch 大小會影響資料多樣性與收斂穩定性。
Transition = namedtuple("Transition", ["s","a","r","s2","done"])
class ReplayBuffer:
    def __init__(self, maxlen=200000):
        self.buf = deque(maxlen=maxlen)
    def push(self, s,a,r,s2,done):
        self.buf.append(Transition(s,a,r,s2,done))
    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s = torch.tensor(np.vstack([b.s for b in batch]), dtype=torch.float32, device=device)
        a = torch.tensor(np.vstack([b.a for b in batch]), dtype=torch.float32, device=device)
        r = torch.tensor(np.vstack([b.r for b in batch]), dtype=torch.float32, device=device)
        s2 = torch.tensor(np.vstack([b.s2 for b in batch]), dtype=torch.float32, device=device)
        d = torch.tensor(np.vstack([b.done for b in batch]).astype(np.float32), dtype=torch.float32, device=device)
        return s,a,r,s2,d
    def __len__(self): return len(self.buf)

# ---------- Networks ----------
def mlp(in_dim, out_dim, hidden=256, final_act=None):
    layers = [nn.Linear(in_dim, hidden), nn.ReLU(),
              nn.Linear(hidden, hidden//2), nn.ReLU(),
              nn.Linear(hidden//2, out_dim)]
    if final_act: layers.append(final_act)
    return nn.Sequential(*layers)

class Actor(nn.Module):
    def __init__(self, state_dim, action_max=24.0):
        super().__init__()
        self.net = mlp(state_dim, 1, hidden=256, final_act=nn.Tanh())
        self.action_max = action_max
    #state
    def forward(self, s):
        #actor 使用 tanh * action_max 可以自然將動作壓到合適範圍。
        return self.net(s) * self.action_max

class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = mlp(state_dim + 1, 1, hidden=256)
    #state, action
    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        return self.net(x)

# ---------- OU noise ----------
#OUActionNoise 實作的是 Ornstein–Uhlenbeck (OU) 隨機過程 的離散近似，用來在連續動作強化學習（例如 DDPG）中產生有時間相關性的「平滑」（temporally correlated）噪聲，避免每步都是獨立白噪聲，讓探索在實際控制上更合理。
#mu: OU 過程的長期平均（mean）或平衡值（目標值）。在沒有外力時，x 會往 mu 回歸。
#sigma: 隨機擾動強度（噪聲幅度），越大波動越大。
#theta: 回歸係數（mean reversion rate），值越大代表偏離 mu 時會更快拉回來。
#dt: 時間步長（離散化的時間間隔）。若你的 agent 每次更新不是 1 秒，可以用實際時間間隔；常常為 1。
#x_prev: 保存上一時間步的噪聲值（OU 過程是有記憶的），初始設為 0.
#
#mu(μ) = 0.0：對 actions 常見，噪聲平均為 0。 
#theta(θ)：0.1 ~ 0.3（如果希望快速回歸可設更高）。
#sigma(σ)：0.1 ~ 0.6 常見；開始時大一點（例如 0.3~0.6），後期可 anneal 到 0.01。
#dt：如果每步更新為 1，設 1；若每 step 表示 0.02s，設 0.02（並讓噪聲幅度與時間步長一致）。
#
# μ — 平衡點 / 長期目標       物理上：類似「阻尼彈簧的平衡位置」
# θ — 阻尼/回歸速度          越大：系統迅速被拉回 μ 噪聲變得回歸更快、變動較小
# σ — 熱擾動強度             越大：噪聲幅度越大 探索更激進
# dt — 時間尺度              如果模擬每一步代表更小的時間： 噪聲放大方式會不同 方差對應 𝑑𝑡 dt
#
#OU 噪聲是一種模擬「彈簧 + 隨機撞擊」的過程
#
class OUActionNoise:
    def __init__(self, mu=0.0, sigma=0.5, theta=0.15, dt=1.0):
        self.mu = mu; self.sigma = sigma; self.theta = theta; self.dt = dt
        self.x_prev = 0.0
    def __call__(self):
        x = self.x_prev + self.theta*(self.mu - self.x_prev)*self.dt + self.sigma*math.sqrt(self.dt)*np.random.randn()
        self.x_prev = x
        return x
    #reset()：在每個 episode 開始時把過程重置（通常要這麼做，否則噪聲會跨越 episode）
    def reset(self): self.x_prev = 0.0 
    def set_sigma(self, sigma): self.sigma = sigma

# ---------- soft update ----------
def soft_update(target, source, tau):
    for t,p in zip(target.parameters(), source.parameters()):
        t.data.copy_(t.data * (1.0 - tau) + p.data * tau)

# ---------- training function ----------
def train_ddpg_stable(num_episodes=1000, episode_len=400, batch_size=128):
    state_dim = 4  # [omega, target_speed, delta_error,error_sum]
    actor = Actor(state_dim, action_max=24.0).to(device)
    critic = Critic(state_dim).to(device)
    actor_t = Actor(state_dim, action_max=24.0).to(device)
    critic_t = Critic(state_dim).to(device)
    actor_t.load_state_dict(actor.state_dict())
    critic_t.load_state_dict(critic.state_dict())

    actor_opt = optim.Adam(actor.parameters(), lr=1e-4)
    critic_opt = optim.Adam(critic.parameters(), lr=1e-3)

    buffer = ReplayBuffer()
    noise = OUActionNoise(sigma=0.6)  # initial exploration sigma 越大：噪聲幅度越大 探索更激進
    gamma = 0.99; tau_soft = 0.005
    Ts = 0.01  # Assuming DifferentiableMotor default Ts=0.01

    rewards_log = []
    u_max_log = []
    u_mean_log = []
    error_log = []
    
    # 幅值
    A_min=0.1
    A_max=0.5
    
    #偏置，中心点在 1.0
    O_min=-0.3
    O_max=0.3
    
    # 频率/周期随机化范围
    T_min = 3.0
    T_max = 100.0

    #课程学习 (Curriculum Learning)
    for ep in range(1, num_episodes+1):
        # curriculum: first few episodes fixed params, then randomize
        if ep <= 100:
            K, tau_m, zeta = 10.0, 0.5, 0.7
            
            T_cycle = float(T_min+T_max)/2.0
            omega_freq = 2 * math.pi / T_cycle
            amplitude = float(A_min+A_max)/2.0
            offset = float(O_min+O_max)/2.0 

        elif ep <= 200:
            K, tau_m, zeta = 10.0, 0.5, 0.7

            T_cycle = float(np.random.uniform(T_min, T_max))
            omega_freq = 2 * math.pi / T_cycle
            amplitude = float(A_min+A_max)/2.0
            offset = float(O_min+O_max)/2.0                
        elif ep <= 300:
            K, tau_m, zeta = 10.0, 0.5, 0.7
            
            T_cycle = float(np.random.uniform(T_min, T_max))
            omega_freq = 2 * math.pi / T_cycle
            amplitude = float(np.random.uniform(A_min, A_max))
            offset = float(O_min+O_max)/2.0  
        else:
            K, tau_m, zeta = 10.0, 0.5, 0.7
	
            T_cycle = float(np.random.uniform(T_min, T_max))
            omega_freq = 2 * math.pi / T_cycle
            amplitude = float(np.random.uniform(A_min, A_max))
            offset = float(np.random.uniform(O_min, O_max))     
        
        # 初始目标速度 (t=0 时 sin(0)=0)
        initial_target_speed = float(amplitude * math.sin(0) + offset)
        target_speed = initial_target_speed
        

        env = DifferentiableMotor(K=K, tau=tau_m, zeta=zeta, Ts=Ts)
        env.reset()
        

        error_sum = 0.0  # 累积误差 I 项 (积分项)
        ep_reward = 0.0
        noise.reset()

        # decay noise sigma across episodes
        sigma_decay = max(0.05, 0.6 * (1.0 - (ep/num_episodes)))
        noise.set_sigma(sigma_decay)

        # stats
        episode_us = []
        
        #[omega, target_speed, delta_error, error_sum]
        state = np.array([env.omega, target_speed, 0.0,error_sum], dtype=np.float32)
        prev_error = target_speed - env.omega # 初始误差

        for t in range(episode_len):
            
            # *****************************************
            # --- 目标速度动态更新 ---
            current_time = t * Ts
            target_speed = float(amplitude * math.sin(omega_freq * current_time) + offset)
            target_speed = target_speed
            # ------------------------
            
            s_t = torch.tensor(state.reshape(1,-1), dtype=torch.float32, device=device)
            with torch.no_grad():
                a_det = actor(s_t).cpu().numpy().flatten()[0]

            # exploration: add noise during training
            a_t = a_det + noise()
            a_t = float(np.clip(a_t, -actor.action_max, actor.action_max))
            episode_us.append(a_t)

            # step env
            omega_next = env.step(a_t)
            
            # 误差计算基于最新的 target_speed
            error = target_speed - omega_next
            error_sum += error * Ts # 累积积分误差
            delta_error = error - prev_error

            # reward: 强调 error reduction (prev_error - error), small penalty on magnitude
            # 为了提高跟踪精度，减少了控制量惩罚 (0.0005 -> 0.0001)
            #r = 2.0 * (prev_error - error) - 0.8 * abs(error) - 0.0001 * (a_t**2)
            #r = 2.0 * (prev_error - error) - 0.8 * abs(error) - 0.001 * abs(error_sum) - 0.0001 * (a_t**2)
            r = -2.0 * abs(delta_error) - 0.8 * abs(error) - 0.001 * abs(error_sum) - 0.0001 * abs(a_t)
            r = float(np.clip(r, -10.0, 10.0))

            # next_state 必须包含当前的目标速度
            next_state = np.array([omega_next, target_speed, delta_error,error_sum], dtype=np.float32)
            
            #(state, action, reward, next_state, done)
            buffer.push(state, 
                        np.array([a_t], dtype=np.float32), 
                        np.array([r], dtype=np.float32), 
                        next_state, 
                        False)

            state = next_state
            prev_error = error # 更新 prev_error 以供下一轮使用
            ep_reward += r

            # learning step
            if len(buffer) >= batch_size:
                #(state, action, reward, next_state, done)
                s_b, a_b, r_b, s2_b, d_b = buffer.sample(batch_size)
                with torch.no_grad():
                    a2 = actor_t(s2_b)
                    q_next = critic_t(s2_b, a2)
                    q_target = r_b + (1.0 - d_b) * gamma * q_next

                q_val = critic(s_b, a_b)
                critic_loss = nn.MSELoss()(q_val, q_target)
                critic_opt.zero_grad(); critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                critic_opt.step()

                actor_loss = -critic(s_b, actor(s_b)).mean()
                actor_opt.zero_grad(); actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                actor_opt.step()

                soft_update(actor_t, actor, tau_soft)
                soft_update(critic_t, critic, tau_soft)

        error_log.append(error)
        rewards_log.append(ep_reward)
        u_max_log.append(max(episode_us) if episode_us else 0.0)
        u_mean_log.append(np.mean(episode_us) if episode_us else 0.0)

        if ep % 10 == 0:
            print(f"Ep {ep}/{num_episodes} reward {ep_reward:.3f}  u_max={u_max_log[-1]:.3f} u_mean={u_mean_log[-1]:.3f} noise_sigma={noise.sigma:.3f} error={error_log[-1]:.3f}")

        if ep % 200 == 0:
            torch.save(actor.state_dict(), "ddpg_actor_stable_sin.pth") # 更改保存文件名

    torch.save(actor.state_dict(), "ddpg_actor_stable_sin.pth") # 更改保存文件名
    print("Training finished. Actor saved as ddpg_actor_stable_sin.pth")

    # plot reward and u stats
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(rewards_log); plt.xlabel("episode"); plt.ylabel("episode reward"); plt.grid(True)
    plt.subplot(1,2,2)
    plt.plot(u_max_log, label="u_max"); plt.plot(u_mean_log, label="u_mean"); plt.xlabel("episode"); plt.legend(); plt.grid(True)
    plt.show()
    return actor

if __name__ == "__main__":
    # 推荐增加训练集数以适应动态目标追踪
    train_ddpg_stable(num_episodes=400, episode_len=400, batch_size=64)