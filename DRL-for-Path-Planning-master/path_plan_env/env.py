# -*- coding: utf-8 -*-
"""
路径规划强化学习环境
 Created on Tue May 16 2023 17:54:17
 Modified on 2023-8-02 17:38:00
 
 @auther: HJ https://github.com/zhaohaojie1998
"""
#
import gym
import math
import numpy as np
import matplotlib.pyplot as plt
from gym import spaces
from copy import deepcopy
from pathlib import Path
from collections import deque
from scipy.integrate import odeint
from shapely import geometry as geo
from shapely.plotting import plot_polygon
from matplotlib.patches import Rectangle, Polygon
from shapely.geometry import Point as shp_Point
from shapely.geometry import Polygon as shp_Polygon

# __all__ = ["DynamicPathPlanning", "StaticPathPlanning", "NormalizedActionsWrapper"]



#----------------------------- ↓↓↓↓↓ 地图设置 ↓↓↓↓↓ ------------------------------#
class MAP:
    size = [[-2.0, -0.6], [2.0, 0.4]]  # x, z最小值; x, z最大值
    start_size = [[-2.0, -0.3], [2.0, 0.0]]  # 约束训练时起始坐标
    start_pos = [-0.75, -0.2]              # 起点坐标
    end_pos = [0, 0.10]                   # 终点坐标
    obstacles = [                          # 障碍物, 要求为 geo.Polygon 或 带buffer的 geo.Point/geo.LineString
        # geo.Polygon([(-1.0, 0), (-0.99, 0), (-0.99, -0.5), (-1.0, -0.5)]),
        # geo.Polygon([(1.0, 0), (0.99, 0), (0.99, -0.5), (1.0, -0.5)]),
        # geo.Polygon([(-0.99, -0.49), (-0.99, -0.5), (0.99, -0.5), (0.99, -0.49)]),
        geo.Polygon([(-5.0, 0.4), (-5.0, 0.3), (5.0, 0.3), (5.0, 0.4)]),
        geo.Polygon([(-5.0, -0.4), (-5.0, -0.5), (5.0, -0.5), (5.0, -0.4)]),
        geo.Polygon([(-1.0, 0.3), (-1.0, 0), (-0.15, 0), (-0.15, 0.3)]),
        geo.Polygon([(1.0, 0.3), (1.0, 0), (0.15, 0), (0.15, 0.3)]),
    ]


    #这两个函数没用
    @classmethod
    def show(cls):
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.close('all')
        fig, ax = plt.subplots()
        ax.clear()
        cls.plot(ax)
        ax.scatter(cls.start_pos[0], cls.start_pos[1], s=30, c='k', marker='x', label='起点')
        ax.scatter(cls.end_pos[0], cls.end_pos[1], s=30, c='k', marker='o', label='终点')
        ax.legend(loc='best').set_draggable(True)
        plt.show(block=True)

    @classmethod
    def plot(cls, ax, title='Map'):
        ax.clear()
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.grid(alpha=0.3, ls=':')
        ax.set_xlim(cls.size[0][0], cls.size[1][0])
        ax.set_ylim(cls.size[1][1], cls.size[0][1])
        ax.invert_yaxis()
        for o in cls.obstacles:
            plot_polygon(o, ax=ax, facecolor='w', edgecolor='k', add_points=False)
        



#----------------------------- ↓↓↓↓↓ 动力学避障环境 ↓↓↓↓↓ ------------------------------#
if __name__ == '__main__':
    from lidar_sim import LidarModel
else:
    from .lidar_sim import LidarModel
class Logger:
    pass


# 运动速度设置
V_LOW = 0.01  # 最小速度
V_HIGH = 0.30  # 最大速度
V_MIN = V_LOW + 0.05  # 惩罚区间下限(大于V_LOW)
V_MAX = V_HIGH - 0.05  # 惩罚区间上限(小于V_HIGH)
# 质心动力学状态设置
STATE_LOW = [MAP.start_size[0][0], MAP.start_size[0][1], V_LOW, -math.pi] # x, z, V, ψ
STATE_HIGH = [MAP.start_size[1][0], MAP.start_size[1][1], V_HIGH, math.pi] # x, z, V, ψ
# 观测设置
OBS_STATE_LOW = [0, V_LOW, -math.pi]                                                                  # 相对终点距离 + 速度 + 终点与速度的夹角(单位rad)
OBS_STATE_HIGH = [1.414*max(STATE_HIGH[0]-STATE_LOW[0], STATE_HIGH[1]-STATE_LOW[1]), V_HIGH, math.pi] # 相对终点距离 + 速度 + 终点与速度的夹角(单位rad)
# 控制设置
CTRL_LOW = [-0.02, -0.005]  # 切向过载 + 速度滚转角(单位rad/s)
CTRL_HIGH = [0.02, 0.005]  # 切向过载 + 速度滚转角(单位rad/s)
# 雷达设置
SCAN_RANGE = 12  # 扫描距离
SCAN_ANGLE = 360  # 扫描范围(单位deg)
SCAN_NUM = 180   # 扫描点个数
SCAN_CEN = 20    # 中心区域范围(小于SCAN_NUM/2)
# 距离设置
D_SAFE = 0.10  # 碰撞半径,这个已经没有用了
D_BUFF = 0.15  # 缓冲距离(大于D_SAFE)
D_ERR = 0.02  # 目标误差距离
D_HEAD = 0.04  # 雷达偏移距离
# 序列观测长度
TIME_STEP = 4
# 小车参数
CAR_LENGTH = 0.24  # 小车长
CAR_WIDTH = 0.2    # 小车宽



class DynamicPathPlanning(gym.Env):
    """从力学与控制的角度进行规划 (东天南坐标系)
    # >>> dx/dt = V * cos(ψ)
    # >>> dz/dt = -V * sin(ψ)
    # >>> dV/dt = g * nx
    # >>> dψ/dt = -g / V * tan(μ)
    # >>> u = [nx, μ]
    """

    def __init__(self, max_episode_steps=500, dt=0.2, normalize_observation=True, old_gym_style=True):
        """
        Args:
            max_episode_steps (int): 最大仿真步数. 默认500.
            dt (float): 决策周期. 默认0.2.
            normalize_observation (bool): 是否输出归一化的观测. 默认True.
            old_gym_style (bool): 是否采用老版gym接口. 默认True.
        """
        # 仿真
        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.log = Logger()
        # 障碍 + 雷达
        self.obstacles = MAP.obstacles
        self.lidar = LidarModel(SCAN_RANGE, SCAN_ANGLE, SCAN_NUM)
        self.lidar.add_obstacles(MAP.obstacles)
        # 状态空间 + 控制空间
        self.state_space = spaces.Box(np.array(STATE_LOW), np.array(STATE_HIGH))
        self.control_space = spaces.Box(np.array(CTRL_LOW), np.array(CTRL_HIGH))
        # 观测空间 + 动作空间
        points_space = spaces.Box(-1, SCAN_RANGE, (TIME_STEP, SCAN_NUM, )) # seq_len, dim
        vector_space = spaces.Box(np.array([OBS_STATE_LOW]*TIME_STEP), np.array([OBS_STATE_HIGH]*TIME_STEP)) # seq_len, dim
        self.observation_space = spaces.Dict({'seq_points': points_space, 'seq_vector': vector_space})
        self.action_space = spaces.Box(-1, 1, (len(CTRL_LOW), ))
        # 序列观测
        self.deque_points = deque(maxlen=TIME_STEP)
        self.deque_vector = deque(maxlen=TIME_STEP)
        # 环境控制
        self.__render_not_called = True
        self.__need_reset = True
        self.__norm_observation = normalize_observation
        self.__old_gym = old_gym_style
        self.car_length, self.car_width = CAR_LENGTH, CAR_WIDTH
        # plt设置
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK', 'DejaVu Sans', 'Arial Unicode MS', 'WenQuanYi Zen Hei','Microsoft YaHei']        # 设置通用中文字体（跨平台使用）
        plt.rcParams['axes.unicode_minus'] = False
        plt.close("all")

    def reset(self, mode=0, fixed_end_pos=None):
        """重置环境
           mode=0, 随机初始化起点终点, 速度、方向随机
           mode=1, 初始化起点终点到地图设置, 速度、方向随机
           fixed_end_pos: 设定目标位置的可选参数，格式为 (x, y)，如果为 None，则随机选择目标位置
        """
        self.__need_reset = False
        self.time_step = 0
        # 初始化航程/状态/控制
        while 1:
            self.state = self.state_space.sample()
            if mode == 0:
                self.start_pos = deepcopy(self.state[:2])  # 分不清引用传递, 暴力deepcopy就完事了
                if fixed_end_pos is None:
                    self.end_pos = deepcopy(self.state_space.sample()[:2])  # 随机目标位置
                else:
                    self.end_pos = np.array(fixed_end_pos, dtype=np.float32)  # 固定目标位置
                self.state = np.array([*self.start_pos[:2], 0.01, 0.0], dtype=np.float32)
            else:
                self.start_pos = np.array(MAP.start_pos, dtype=np.float32)
                if fixed_end_pos is None:
                    self.end_pos = np.array(MAP.end_pos, dtype=np.float32)  # 从地图中获取目标位置
                else:
                    self.end_pos = np.array(fixed_end_pos, dtype=np.float32)  # 使用传入的目标位置
                self.state = np.array([*self.start_pos[:2], 0.01, 0.0], dtype=np.float32)
                # self.state = np.array([*self.start_pos[:2], *self.state[2:]], dtype=np.float32)

            # 检查起点和终点是否在障碍物中
            for o in self.obstacles:
                if o.contains(geo.Point(*self.start_pos)) or o.contains(geo.Point(*self.end_pos)):
                    if mode != 0:
                        raise ValueError("地图的初始/目标位置不能设置在障碍里面!!!")
                    break
            else:
                break
        self.L = 0.0                                                         # 航程
        self.ctrl = np.zeros(self.action_space.shape, dtype=np.float32) # 初始控制量
        # 初始化观测
        self.deque_points.extend([np.array([-1]*SCAN_NUM, dtype=np.float32)]*(TIME_STEP-1)) # 初始化到 -1
        self.deque_vector.extend([np.array(OBS_STATE_LOW, dtype=np.float32)]*(TIME_STEP-1)) # 初始化到 low
        obs = self._get_obs(self.state)
        # 初始化记忆
        self.exist_last = None # 上一时刻中心区域是否存在障碍
        self.D_init = deepcopy(obs['seq_vector'][-1][0]) # 初始到目标距离
        self.D_last = deepcopy(obs['seq_vector'][-1][0]) # 上一时刻到目标距离
        # 重置log
        self.log.start_pos = self.start_pos       # 起点
        self.log.end_pos = self.end_pos           # 目标
        self.log.path = [self.start_pos]          # 路径
        self.log.ctrl = [self.ctrl]               # 控制
        self.log.speed = [self.state[2]]          # 速度
        self.log.yaw = [self.state[3]]            # 偏角
        self.log.length = [[self.L, self.D_last]] # 航程+距离
        self.log.curr_scan_pos = []               # 当前时刻扫描的障碍坐标
        # 输出
        if self.__old_gym:
            return self._norm_obs(obs)
        return self._norm_obs(obs), {}
    
    def _get_ctrl(self, act, tau=0.9):
        """获取控制"""
        lb = self.control_space.low
        ub = self.control_space.high
        u = lb + (act + 1.0) * 0.5 * (ub - lb) # [-1,1] -> [lb,ub]
        u = np.clip(u, lb, ub) # NOTE 浮点数误差有时会出现类似 1.0000001 情况
        # smooth control signal
        if tau is not None:
            return (1.0 - tau) * self.ctrl + tau * u
        return u

    def _get_obs(self, state):
        """获取原始观测"""
        x, z, V, ψ = state
        # 相对状态
        V_vec = np.array([V*math.cos(ψ), -V*math.sin(ψ)], np.float32)
        R_vec = self.end_pos - state[:2]
        D = np.linalg.norm(R_vec)
        #ε = self._compute_azimuth(state[:2], self.end_pos) # 方位角
        q = self._vector_angle(V_vec, R_vec) # 视线角
        vector = np.array([D, V, q], np.float32)
        self.deque_vector.append(vector)
        # 雷达测距
        points, self.log.curr_scan_pos = self.lidar.scan(x, z, -ψ, mode=1) 
        # NOTE 东北天坐标系(LidarModel) 和 东天南坐标系(ControlModel) 航迹偏角ψ 的数值是相反的
        self.deque_points.append(points[1]) # 只需要测距维度
        # 观测空间
        return {'seq_points': np.array(self.deque_points, np.float32),
                'seq_vector': np.array(self.deque_vector, np.float32)}

    def _get_rew(self):
        """获取奖励"""
        rew = -0.01
        # 1.主动避障奖励 [-2, 2]
        point0 = self.deque_points[-2] # 0 上一时刻
        center0 = point0[(SCAN_NUM-SCAN_CEN)//2:-(SCAN_NUM-SCAN_CEN)//2]
        point1 = self.deque_points[-1] # 1 当前时刻
        center1 = point1[(SCAN_NUM-SCAN_CEN)//2:-(SCAN_NUM-SCAN_CEN)//2]
        # # 中心区域center障碍变化程度
        # if self.exist_last is None:
        #     self.exist_last = np.any(center0>-0.5)
        # exist = np.any(center1>-0.5)
        # if exist:
        #     # 一直有障碍：看距离变化
        #     if self.exist_last:
        #         effective_center0, effective_center1 = center0[center0>-0.5], center1[center1>-0.5] # 不可能是空数组
        #         d0_mean = np.mean(effective_center0) + 1e-8  # 平均距离 (障碍整体远离程度)
        #         d1_mean = np.mean(effective_center1) + 1e-8
        #         d0_min = min(effective_center0) # 最小距离 (是否远离障碍物)
        #         d1_min = min(effective_center1)
        #         rew += np.clip(d1_mean/d0_mean, 0.2, 2) if d1_min > d0_min else -np.clip(d0_mean/d1_mean, 0.2, 2)
        #     # 无障碍 -> 有障碍
        #     else:
        #         rew -= 0.5
        # else:
        #     # 有障碍 -> 无障碍
        #     if self.exist_last:
        #         rew += 1.0
        #     # 一直无障碍, r += 0
        #     pass
        # 2.被动避障奖励 [-1, 0]
        d_min = min([*point1[point1>-0.5], np.inf])
        if d_min <= D_BUFF:
            rew += d_min/D_BUFF - 1 # -1~0
        # 3.接近目标奖励 {-1.5, 2.0}
        D = self.deque_vector[-1][0]
        rew += 2.0 if D < self.D_last else -1.5
        # 4.速度保持奖励 [-1, 0]
        V = self.deque_vector[-1][1]
        if V < V_MIN:
            rew += (V - V_MIN) / (V_MIN - V_LOW) # -1~0
        elif V > V_MAX:
            rew += (V - V_MAX) / (V_MAX - V_HIGH) # 0~-1
        # 5.视线角奖励 [-1, 1]
        q = self.deque_vector[-1][2]
        rew += math.sin(q) # 1~-1
        # 6.任务奖励
        done = False
        info = {'state': 'none'}
        # 获取小车矩形碰撞区域
        rectangle_points,_,_,_,_ = self._calculate_car_position_and_rotate()
        # 碰撞判定（是否与障碍物相交）
        if self._check_collision(rectangle_points, self.log.curr_scan_pos) or d_min < 0.01:
            rew -= 150  # 碰撞的惩罚
            done = True
            info['state'] = 'fail'
        # elif d_min < D_SAFE:  # 碰撞
        #     rew -= 150
        #     done = True
        #     info['state'] = 'fail'
        elif D < D_ERR:  # 成功
            η = np.nanmax([3.5 - 2.5 * self.L / (self.D_init + 1e-8), 0.5])  # 航程折扣 (实现路径最短)
            rew += 200 * η  # 100~700+
            done = True
            info['state'] = 'success'
        if V < V_MIN or V > V_MAX or d_min < D_BUFF:
            rew -= 5
        # 更新记忆
        # self.exist_last = deepcopy(exist)
        self.D_last = deepcopy(D)

        # 输出
        return rew, done, info

    def _check_collision(self, rectangle_points, obstacles):
        """检测小车的矩形与障碍物是否发生碰撞"""
        for obstacle in obstacles:
            if self._is_point_inside_rectangle(rectangle_points,obstacle):
                return True
        return False

    def _is_point_inside_rectangle(self, rotated_points, obstacle):
        """
        判断点是否在旋转矩形内。

        参数:
            rotated_points (np.ndarray): 旋转后的矩形四个顶点的全局坐标，形状为 (4, 2)。
            obstacle (tuple): 障碍物的坐标 (x, y)，其中 y 对应于 z 轴。

        返回:
            bool: 如果点在矩形内，返回 True，否则返回 False。
        """
        # 将障碍物的 (x, y) 映射到 (x, z)
        x, y = obstacle
        point = shp_Point(x, y)  # y 对应于 z 轴

        # 创建多边形对象
        polygon = shp_Polygon(rotated_points)

        # 使用 Shapely 的 contains 方法判断点是否在多边形内
        return polygon.contains(point)

    def step(self, act: np.ndarray, tau: float = None):
        """状态转移
        Args:
            act (np.ndarray): 动作a(取值-1~1).
            tau (float): 控制量u(取值u_min~u_max)的平滑系数: u = tau*u + (1-tau)*u_last. 默认None不平滑.
        """
        assert not self.__need_reset, "调用step前必须先reset"
        # 数值鸡分
        self.time_step += 1
        u = self._get_ctrl(act, tau)
        new_state = self._ode45(self.state, u, self.dt)
        truncated = False
        if self.time_step >= self.max_episode_steps:
            truncated = True
        # elif new_state[0] > self.state_space.high[0] \
        #     or new_state[1] > self.state_space.high[1] \
        #     or new_state[0] < self.state_space.low[0] \
        #     or new_state[1] < self.state_space.low[1]:
        #         truncated = True
        elif self.D_last > self.observation_space["seq_vector"].high[-1][0]:
            truncated = True
        # 更新航程/状态/唱跳rap篮球
        self.L += np.linalg.norm(new_state[:2] - self.state[:2])
        self.state = deepcopy(new_state)
        self.ctrl = deepcopy(u)
        # 获取转移元组
        obs = self._get_obs(new_state)
        rew, done, info = self._get_rew()
        info["done"] = done
        info["truncated"] = truncated
        if truncated or done:
            info["terminal"] = True
            self.__need_reset = True
        else:
            info["terminal"] = False
        info["reward"] = rew
        info["time_step"] = self.time_step
        info["voyage"] = self.L
        info["distance"] = self.D_last
        # 记录
        self.log.path.append(self.state[:2])
        # self.log.car_leftup.append(np.array(
        #     [self.log.car_leftup[-1][0]+self.log.path[-1][0]-self.log.path[-2][0],
        #      self.log.car_leftup[-1][1]+self.log.path[-1][1]-self.log.path[-2][1]]
        #      , dtype=np.float32))
        self.log.ctrl.append(u)
        self.log.speed.append(self.state[2])
        self.log.yaw.append(self.state[3])
        self.log.length.append([self.L, self.D_last])
        # 输出
        if self.__old_gym:
            return self._norm_obs(obs), rew, done, info
        return self._norm_obs(obs), rew, done, truncated, info
    
    def _norm_obs(self, obs):
        """归一化观测"""
        if not self.__norm_observation:
            return obs
        obs['seq_vector'] = self._linear_mapping(
            obs['seq_vector'], 
            self.observation_space['seq_vector'].low, 
            self.observation_space['seq_vector'].high
        )
        obs['seq_points'] = self._normalize_points(obs['seq_points'])
        return obs

    def _calculate_car_position_and_rotate(self):
        """
        计算小车在东天南坐标系下的旋转矩形位置。

        返回:
            rotated_points (np.ndarray): 旋转后的矩形四个顶点的全局坐标。
            radar_x (float): 雷达的全局 x 坐标。
            radar_z (float): 雷达的全局 z 坐标。
            car_center_x (float): 小车几何中心的全局 x 坐标。
            car_center_z (float): 小车几何中心的全局 z 坐标。
        """
        # 获取小车的姿态位置（雷达的位置）
        car_x, car_z = self.log.path[-1][0], self.log.path[-1][1]  # 雷达的位置
        yaw = self.log.yaw[-1]  # 确保 yaw 是弧度

        v_vector = np.array([np.cos(yaw),np.sin(-yaw)])  #速度的单位向量

        # 定义矩形的尺寸
        half_length = self.car_length / 2  # 小车长度的一半（前后）
        half_width = self.car_width / 2  # 小车宽度的一半（左右）

        # 雷达的位置（全局坐标系）
        radar_x = car_x
        radar_z = car_z

        # 定义小车的几何中心位置（相对于雷达位置）
        # 假设 D_HEAD 是雷达到小车几何中心的距离，方向由 yaw 决定
        car_center_x = radar_x - D_HEAD * np.cos(yaw)
        car_center_z = radar_z - D_HEAD * np.sin(-yaw)

        # 定义矩形四个顶点的坐标，相对于几何中心
        rectangle_points = np.array([
            [-half_length, -half_width],  # 左下
            [half_length, -half_width],  # 右下
            [half_length, half_width],  # 右上
            [-half_length, half_width]  # 左上
        ])  # 形状为 (4, 2)

        # 将矩形顶点移动到相对于雷达的位置
        relative_to_radar = rectangle_points - D_HEAD*v_vector

        # 计算旋转矩阵（绕雷达旋转，yaw 为顺时针旋转角度）
        rotation_matrix = np.array([
            [np.cos(yaw), np.sin(yaw)],
            [-np.sin(yaw), np.cos(yaw)]
        ])

        # 对相对于雷达的点进行旋转
        rotated_relative = rectangle_points @ rotation_matrix.T

        # 将旋转后的点移动回全局坐标系
        rotated_points = rotated_relative + np.array([car_center_x, car_center_z])

        # 更新当前时刻的旋转后的矩形坐标
        self.rotated_points = rotated_points

        return rotated_points, radar_x, radar_z,car_center_x,car_center_z

    def render(self, mode="human", figsize=[8,8]):
        """测试时可视化环境, 和step交替调用 (不要和plot一起调用, 容易卡)"""
        assert not self.__need_reset, "调用render前必须先reset"
        
        # 创建绘图窗口
        if self.__render_not_called:
            self.__render_not_called = False
            with plt.ion():
                fig = plt.figure("render", figsize=figsize)
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            MAP.plot(ax, "Path Plan Environment")
            self.__plt_car_path, = ax.plot([], [], 'k-.')
            self.__plt_car_point = ax.scatter([], [], s=5, c='b', marker='o', label='Agent')
            self.__plt_car_point2 = ax.scatter([], [], s=5, c='r', marker='o', label='Agent')
            self.__plt_targ_range, = ax.plot([], [], 'g:', linewidth=1.0)
            self.__plt_targ_point = ax.scatter([], [], s=15, c='g', marker='o', label='Target')
            self.__plt_lidar_scan, = ax.plot([], [], 'ro', markersize=1.5, label='Points')
            self.__plt_lidar_left, = ax.plot([], [], 'c--', linewidth=0.5)
            self.__plt_lidar_right, = ax.plot([], [], 'c--', linewidth=0.5)
        
        # 绘图
        self.__plt_car_path.set_data(np.array(self.log.path).T) # [xxxxyyyy]
        self.__plt_car_point.set_offsets(self.log.path[-1])     # [xyxyxyxy]
        θ = np.linspace(0, 2*np.pi, 18)
        self.__plt_targ_range.set_data(self.log.end_pos[0]+D_ERR*np.cos(θ), self.log.end_pos[1]+D_ERR*np.sin(θ))
        self.__plt_targ_point.set_offsets(self.log.end_pos)

        rotated_points,radar_x, radar_y,car_center_x,car_center_y = self._calculate_car_position_and_rotate()

        # 创建 Polygon 对象，绘制旋转后的矩形
        if hasattr(self, 'car_polygon'):
            # 如果已经有一个 Polygon 对象，直接更新数据
            self.car_polygon.set_xy(rotated_points)
        else:
            # 创建一个新的 Polygon 对象
            self.car_polygon = Polygon(rotated_points, closed=True, edgecolor='r', facecolor='none', label='Car')
            ax.add_patch(self.car_polygon)

        self.__plt_car_point2.set_offsets([car_center_x, car_center_y])  # 更新小车的位置

        if self.log.curr_scan_pos:
            points = np.array(self.log.curr_scan_pos)
            self.__plt_lidar_scan.set_data(points[:, 0], points[:, 1])
        else:
            self.__plt_lidar_scan.set_data([], [])
        
        # 更新激光扫描线
        x, y, yaw = *self.log.path[-1], self.log.yaw[-1]
        x1 = x + self.lidar.max_range * np.cos(-yaw + np.deg2rad(self.lidar.scan_angle/2))
        x2 = x + self.lidar.max_range * np.cos(-yaw - np.deg2rad(self.lidar.scan_angle/2))
        y1 = y + self.lidar.max_range * np.sin(-yaw + np.deg2rad(self.lidar.scan_angle/2))
        y2 = y + self.lidar.max_range * np.sin(-yaw - np.deg2rad(self.lidar.scan_angle/2))
        self.__plt_lidar_left.set_data([x, x1], [y, y1])
        self.__plt_lidar_right.set_data([x, x2], [y, y2])
        
        # 窗口暂停
        plt.pause(0.001)

    def close(self): 
        """关闭环境"""
        self.__render_not_called = True
        self.__need_reset = True
        plt.close("render")

    def plot(self, file, figsize=[10,10], dpi=100):
        """训练时观察输出状态 (不要和render一起调用, 容易卡)"""
        file = Path(file).with_suffix(".png")
        file.parents[0].mkdir(parents=True, exist_ok=True)
        fig = plt.figure("Output", figsize=figsize)
        gs = fig.add_gridspec(2, 2) 
        ax1 = fig.add_subplot(gs[0, 0]) 
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        MAP.plot(ax1, "Trajectory")
        ax1.scatter(*self.log.path[0], s=30, c='k', marker='x', label='start')
        ax1.scatter(*self.log.end_pos, s=30, c='k', marker='o', label='target')
        ax1.plot(*np.array(self.log.path).T, color='b', label='path') # [xxxxyyyy]
        ax1.legend(loc="best").set_draggable(True)
        ax2.set_title("Control Signal")
        ax2.set_xlabel("time step")
        ax2.set_ylabel("control")
        ctrl = np.array(self.log.ctrl).T
        for i, u in enumerate(ctrl):
            ax2.plot(u, label=f'u{i}')
        ax2.legend(loc="best").set_draggable(True)
        ax3.set_title("Speed Signal")
        ax3.set_xlabel("time step")
        ax3.set_ylabel("speed")
        ax3.plot(self.log.speed, label='V')
        ax3.legend(loc="best").set_draggable(True)
        ax4.set_title("Length Signal")
        ax4.set_xlabel("time step")
        ax4.set_ylabel("length")
        length = np.array(self.log.length).T
        ax4.plot(length[0], label='voyage')
        ax4.plot(length[1], label='distance')
        ax4.legend(loc="best").set_draggable(True)
        plt.tight_layout()
        fig.savefig(fname=file, dpi=dpi)
        plt.close("Output")
    
    @staticmethod
    def _limit_angle(x, domain=1):
        """限制角度 x 的区间: 1限制在(-π, π], 2限制在[0, 2π)"""
        x = x - x//(2*math.pi) * 2*math.pi # any -> [0, 2π)
        if domain == 1 and x > math.pi:
            return x - 2*math.pi           # [0, 2π) -> (-π, π]
        return x
    
    @staticmethod
    def _limit_angle_half_pi(x):
        """限制角度 x 的区间为 [-π/2, π/2]"""
        x = x - x // (2 * math.pi) * 2 * math.pi  # 将角度限制到 [0, 2π)
        if x > math.pi:  # 转换到 (-π, π]
            x -= 2 * math.pi
        if x > math.pi / 2:  # 转换到 [-π/2, π/2]
            return x - math.pi
        if x < -math.pi / 2:
            return x + math.pi
        return x

    @staticmethod
    def _limit_angle_pi_third(x):
        """限制角度 x 的区间为 [-π/3, π/3]"""
        x = x - x // (2 * math.pi) * 2 * math.pi  # 将角度限制到 [0, 2π)
        if x > math.pi:  # 转换到 (-π, π]
            x -= 2 * math.pi
        if x > math.pi / 3:  # 转换到 [-π/3, π/3]
            return x - 2 * math.pi / 3
        if x < -math.pi / 3:
            return x + 2 * math.pi / 3
        return x


    @staticmethod
    def _linear_mapping(x, x_min, x_max, left=0.0, right=1.0):  
        """x 线性变换: [x_min, x_max] -> [left, right]"""
        y = left + (right - left) / (x_max - x_min) * (x - x_min)
        return y
    
    @staticmethod
    def _normalize_points(points, max_range=SCAN_RANGE):
        """归一化雷达测距 (无效数据变成0, 有效数据0.1~1)"""
        points = np.array(points)
        points[points>-0.5] = 0.9*points[points>-0.5]/max_range + 0.1
        points[points<-0.5] = 0.0
        return points
    
    @staticmethod
    def _vector_angle(x_vec, y_vec, EPS=1e-8):
        """计算向量 x_vec 与 y_vec 之间的夹角 [0, π]"""
        x = np.linalg.norm(x_vec) * np.linalg.norm(y_vec)
        y = np.dot(x_vec, y_vec)
        if x < EPS: # 0向量情况
            return 0.0
        if y < EPS: # 90°情况
            return math.pi/2
        return math.acos(np.clip(y/x, -1, 1)) # note: x很小的时候, 可能会超过+-1
    
    @staticmethod
    def _compute_azimuth(pos1, pos2, use_3d_pos=False):
        """东天南坐标系计算pos2相对pos1的方位角 [-π, π] 和高度角(3D情况) [-π/2, π/2] """
        if use_3d_pos:
            x, y, z = np.array(pos2) - pos1
            q = math.atan(y / (math.sqrt(x**2 + z**2) + 1e-8)) # 高度角 [-π/2, π/2]
            ε = math.atan2(-z, x)                              # 方位角 [-π, π]
            return ε, q
        else:
            x, z = np.array(pos2) - pos1
            return math.atan2(-z, x)
    
    @staticmethod
    def _fixed_wing_2d(s, t, u):
        """东天南坐标系平面运动ode模型 (_fixed_wing_3d简化版) 
        s = [x, z, V, ψ]
        u = [nx, μ]
        """
        _, _, V, ψ = s
        nx, μ = u

        if V != 0:
            μ_new = - 9.8/V * math.tan(μ)
        else:
            μ_new = 0  # 或者根据实际情况处理零值的情况

        dsdt = [
            V * math.cos(ψ),
            -V * math.sin(ψ),
            9.8 * nx,
            μ_new # μ<90, 不存在inf情况
        ]
        return dsdt

    @classmethod
    def _ode45(cls, s_old, u, dt):
        """微分方程积分"""
        s_new = odeint(cls._fixed_wing_2d, s_old, (0.0, dt), args=(u, )) # shape=(len(t), len(s))
        x, z, V, ψ = s_new[-1]
        V = np.clip(V, V_LOW, V_HIGH)
        ψ = cls._limit_angle(ψ)
        # ψ = cls._limit_angle_half_pi(ψ)
        # ψ = cls._limit_angle_pi_third(ψ)
        return np.array([x, z, V, ψ], dtype=np.float32) # deepcopy

    







'''------------------------↑↑↑↑↑ 动态避障环境 ↑↑↑↑↑---------------------------
#                                          ...
#                                       .=BBBB#-
#                                      .B%&&&&&&#
#                                .=##-.#&&&&&#%&%
#                              -B&&&&&#=&&&&&B=-.
#                             =&@&&&&&&==&&&&@B
#                           -%@@@&&&&&&&.%&&&&@%.
#                          =&@@@%%@&&&&@#=B&&@@@%
#                         =@@@$#.%@@@@@@=B@@&&@@@-
#                         .&@@@%&@@@@@@&-&@@@%&@@=
#                            #&@@@&@@@@@B=@@@@&B@@=
#                             -%@@@@@@@#B@@@@@B&@-
#                              .B%&&&&@B&@@@@@&@#
#                             #B###BBBBBBB%%&&%#
#                            .######BBBBBBBBBB.
#                            =####BBBBBBBBBBBB#-
#                          .=####BB%%B%%%%%%BB##=
#                         .=##BBB%%#-  -#%%%BBB##.
#                        .=##BBB%#.      .#%%BBBB#.
#                        =##BB%%-          =%%BBBB=
#                       =#BB%%B-            .B%%%B#-
#                      =##BBB-                -BB###.
#                     -=##BB-                  -##=#-
#                     ==##B=-                  -####=
#                     =##B#-                   -####=
#                     ###B=                     =###=
#                    =##B#-                      ###=
#                    =BB#=                       =BB=
#                   -%&%                         =&&#
#                   %&%%                         B%&&=
---------------------------↓↓↓↓↓ 静态避障环境 ↓↓↓↓↓---------------------------'''










#----------------------------- ↓↓↓↓↓ 路径搜索环境 ↓↓↓↓↓ ------------------------------#
class StaticPathPlanning(gym.Env):
    """从航点搜索的角度进行规划"""

    def __init__(self, num_pos=6, max_search_steps=200, old_gym_style=True):
        """
        Args:
            num_pos (int): 起点终点之间的航点个数. 默认6.
            max_search_steps (int): 最大搜索步数. 默认200.
            old_gym_style (bool): 是否采用老版gym接口. 默认True.
        """
        self.num_pos = num_pos
        self.map = MAP
        self.max_episode_steps = max_search_steps

        lb = np.array(self.map.size[0] * num_pos)
        ub = np.array(self.map.size[1] * num_pos)
        self.observation_space = spaces.Box(lb, ub, dtype=np.float32)
        self.action_space = spaces.Box(lb/10, ub/10, dtype=np.float32)

        self.__render_not_called = True
        self.__need_reset = True
        self.__old_gym = old_gym_style

        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.close("all")

    def reset(self):
        self.__need_reset = False
        self.time_steps = 0 # NOTE: 容易简写成step, 会和step method重名, 还不容易发现 BUG
        self.obs = self.observation_space.sample()
        # New Gym: obs, info
        # Old Gym: obs
        if self.__old_gym:
            return self.obs
        return self.obs, {}
    
    def step(self, act):
        """
        转移模型 1
        Pos_new = act = Actor(Pos_old)
        Q = Critic(Pos_old, act) = Critic(Pos_old, Pos_new)
        转移模型 2
        Pos_new = Pos_old + act, act = Actor(Pos_old)
        Q = Critic(Pos_old, act) = Critic(Pos_old, Pos_new-Pos_old)
        """
        assert not self.__need_reset, "调用step前必须先reset"
        # 状态转移
        obs = np.clip(self.obs + act, self.observation_space.low, self.observation_space.high)
        self.time_steps += 1
        # 计算奖励
        rew, done, info = self._get_reward(obs)
        # 回合终止
        truncated = self.time_steps >= self.max_episode_steps
        if truncated or done:
            info["terminal"] = True
            self.__need_reset = True
        else:
            info["terminal"] = False
        # 更新状态
        self.obs = deepcopy(obs)
        # New Gym: obs, rew, done, truncated, info
        # Old Gym: obs, rew, done, info
        if self.__old_gym:
            return obs, rew, done, info
        return obs, rew, done, truncated, info
    
    def _get_reward(self, obs):
        traj = np.array(self.map.start_pos + obs.tolist() + self.map.end_pos) # [x,y,x,y,x,y,...]
        traj = traj.reshape(self.num_pos+2, 2) # [[x,y],[x,y],...]
        # 惩罚
        num_over = 0 # 超出约束的状态个数
        for o, l, u in zip(obs, self.observation_space.low, self.observation_space.high):
            if o <= l or o >= u:
                num_over += 2
        # 轨迹检测
        d = 0.0      # 总长度 -> 衡量能量
        dθall = 0.0  # 总的角度变化 -> 衡量能量
        θ_last = 0.0 # 转角变化 < 45度
        num_theta = 0 # 不合理转角次数 -> 惩罚
        num_crash = 0 # 碰撞次数 -> 惩罚
        for i in range(len(traj) - 1):
            vec = traj[i+1] - traj[i]   # 轨迹向量 (2, )
            # 累积长度
            d += np.linalg.norm(vec) 
            # 转角检测
            θ = math.atan2(vec[1], vec[0])
            dθ = abs(θ - θ_last) if i !=0 else 0.0 # 第一段没转角变化
            if dθ >= math.pi/4: 
                num_theta += 1
            dθall += dθ
            θ_last = deepcopy(θ)
            # 碰撞检测
            for o in self.map.obstacles:
                line = geo.LineString(traj[i:i+2])
                if o.intersects(line): # 判断是否有交集
                    num_crash += 1
            #end 
        #end
        # 总奖励
        rew = -d -dθall/self.num_pos -num_theta -num_crash -num_over
        # 是否终止
        if num_theta == 0 and num_crash == 0:
            rew += 100  # 给个终端奖励
            done = True # 轨迹合理
        else:
            done = False # 轨迹不合理
        info = {}
        info['碰撞次数'] = num_crash
        info['不平滑次数'] = num_theta
        info['done'] = done
        return rew, done, info
    
    def render(self, mode="human"):
        """环境可视化, 和step交替调用"""
        assert not self.__need_reset, "调用render前必须先reset"
        if self.__render_not_called:
            self.__render_not_called = False
            plt.ion() # 打开交互绘图, 只能开一次
        # 创建窗口
        plt.clf()
        plt.axis('equal')
        plt.xlim(self.map.size[0][0], self.map.size[1][0])
        plt.ylim(self.map.size[1][1], self.map.size[0][1]) # NOTE min/max调换顺序可反转坐标轴
        # 绘制
        for o in self.map.obstacles:
            plot_polygon(o, facecolor='w', edgecolor='k', add_points=False)
        plt.scatter(self.map.start_pos[0], self.map.start_pos[1], s=30, c='k', marker='x', label='起点')
        plt.scatter(self.map.end_pos[0], self.map.end_pos[1], s=30, c='k', marker='o', label='终点')
        traj = self.map.start_pos + self.obs.tolist() + self.map.end_pos # [x,y,x,y,x,y,]
        plt.plot(traj[::2], traj[1::2], label='path', color='b')
        # 设置信息
        plt.title('Path Search Environment')
        plt.legend(loc='best')
        plt.xlabel("x")
        plt.ylabel("z")
        plt.grid(alpha=0.3, ls=':')
        # 关闭窗口
        plt.pause(0.001)
        plt.ioff()

    def close(self):
        """关闭环境"""
        self.__render_not_called = True
        self.__need_reset = True
        plt.close()

    def plot(self, *args, **kwargs):
        """输出"""
        print("你干嘛")












#----------------------------- ↓↓↓↓↓ 环境-算法适配 ↓↓↓↓↓ ------------------------------#
class NormalizedActionsWrapper(gym.ActionWrapper):
    """非-1~1的连续动作空间环境装饰器"""
    def __init__(self, env):
        super(NormalizedActionsWrapper, self).__init__(env)
        assert isinstance(env.action_space, spaces.Box), '只用于Box动作空间'
  
    # 将神经网络输出转换成gym输入形式
    def action(self, action): 
        # 连续情况 scale action [-1, 1] -> [lb, ub]
        lb, ub = self.action_space.low, self.action_space.high
        action = lb + (action + 1.0) * 0.5 * (ub - lb)
        action = np.clip(action, lb, ub)
        return action

    # 将gym输入形式转换成神经网络输出形式
    def reverse_action(self, action):
        # 连续情况 normalized action [lb, ub] -> [-1, 1]
        lb, ub = self.action_space.low, self.action_space.high
        action = 2 * (action - lb) / (ub - lb) - 1
        return np.clip(action, -1.0, 1.0)
       








if __name__ == '__main__':
    env = DynamicPathPlanning()
    for ep in range(10):
        print(f"episode{ep}: begin")
        obs = env.reset(mode=0,fixed_end_pos=MAP.end_pos)
        while 1:
            try:
                env.render()
                obs, rew, done, info = env.step(np.array([0.5, 0.2]))
                print(info)
            except AssertionError:
                break
        #env.plot(f"output{ep}")
        print(f"episode{ep}: end")