import numpy as np
from scipy.optimize import linear_sum_assignment

# 定义导弹位置和飞行方向（指向原点）
# missiles = {
#     'M1': np.array([20000, 0, 2000]),
#     'M2': np.array([19000, 600, 2100]),
#     'M3': np.array([18000, -600, 1900])
# }

# drones = {
#     'FY1': np.array([17800, 0, 1800]),
#     'FY2': np.array([12000, 1400, 1400]),
#     'FY3': np.array([6000, -3000, 700]),
#     'FY4': np.array([11000, 2000, 1800]),
#     'FY5': np.array([13000, -2000, 1300])
# }

# # 原点（假目标）
# origin = np.array([0, 0, 0])

# def distance_to_line(point, line_point, direction):
#     """
#     计算点到直线的距离
#     point: 无人机位置
#     line_point: 导弹位置（直线上一点）
#     direction: 直线方向向量（指向原点）
#     """
#     # 计算向量
#     vec_to_point = point - line_point
#     # 点到直线的距离公式：|(AP) × u| / |u|
#     cross_product = np.linalg.norm(np.cross(vec_to_point, direction))
#     distance = cross_product / np.linalg.norm(direction)
#     return distance

# # 创建成本矩阵（行：导弹，列：无人机）
# cost_matrix = np.zeros((3, 5))

# missile_names = list(missiles.keys())
# drone_names = list(drones.keys())

# for i, m in enumerate(missile_names):
#     missile_pos = missiles[m]
#     # 导弹飞行方向向量（指向原点）
#     direction_vector = origin - missile_pos
    
#     for j, d in enumerate(drone_names):
#         drone_pos = drones[d]
#         # 计算无人机到导弹飞行直线的距离
#         dist = distance_to_line(drone_pos, missile_pos, direction_vector)
#         cost_matrix[i, j] = dist

# # 使用匈牙利算法找到最小总成本的分配
# row_ind, col_ind = linear_sum_assignment(cost_matrix)

# # 输出分配结果
# print("分配结果（每个导弹分配无人机，基于到导弹飞行直线的距离）：")
# for i in range(len(row_ind)):
#     m = missile_names[row_ind[i]]
#     d = drone_names[col_ind[i]]
#     dist = cost_matrix[row_ind[i], col_ind[i]]
#     print(f"{m} 分配给 {d}，垂直距离为 {dist:.2f} 米")

# # 输出所有距离矩阵
# print("\n所有无人机到各导弹飞行直线的距离矩阵（行：导弹，列：无人机）：")
# header = "导弹\\无人机" + "".join([f"{d:>12}" for d in drone_names])
# print(header)
# for i, m in enumerate(missile_names):
#     row = f"{m:>10}" + "".join([f"{cost_matrix[i, j]:>12.2f}" for j in range(5)])
#     print(row)

# # 输出每个导弹的飞行方向向量
# print("\n各导弹的飞行方向向量（指向原点）：")
# for m in missile_names:
#     direction = origin - missiles[m]
#     print(f"{m}: {direction} (长度: {np.linalg.norm(direction):.2f} m)")
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

def calculate_smoke_obscuration(drone_direction, drone_speed, drop_time, blast_delay, 
                               visualize=False, verbose=False):
    """
    计算烟幕干扰弹对导弹M1的有效遮蔽时间
    
    参数:
    drone_direction: 无人机飞行方向向量 (3D向量)
    drone_speed: 无人机飞行速度 (m/s)
    drop_time: 受领任务后投放时间 (s)
    blast_delay: 投放后到起爆的时间间隔 (s)
    visualize: 是否可视化结果
    verbose: 是否输出详细信息
    
    返回:
    effective_time: 有效遮蔽时间 (s)
    """
    
    # 定义常量

    g = 9.8
    v_missile = 300
    v_smoke_sink = 3
    effective_radius = 10
    effective_duration = 20

    # 目标位置
    fake_target = np.array([0, 0, 0])
    real_target = np.array([0, 200, 0])
# missiles = {
#     'M1': np.array([20000, 0, 2000]),
#     'M2': np.array([19000, 600, 2100]),
#     'M3': np.array([18000, -600, 1900])
# }

# drones = {
#     'FY1': np.array([17800, 0, 1800]),
#     'FY2': np.array([12000, 1400, 1400]),
#     'FY3': np.array([6000, -3000, 700]),
#     'FY4': np.array([11000, 2000, 1800]),
#     'FY5': np.array([13000, -2000, 1300])
# }
    # 导弹M1初始位置
    M_start = np.array([[20000, 0, 2000],
                        [19000, 600, 2100],
                        [18000, -600, 1900]])

    # 无人机FY1初始位置
    FY_start = np.array([[17800, 0, 1800],
                        [12000, 1400, 1400],
                        [6000, -3000, 700],
                        [11000, 2000, 1800],
                        [13000, -2000, 1300]])
    
    # 标准化无人机方向向量
    # # drone_direction = drone_direction
    # drone_direction = np.arange(5 * 3).reshape((5, 3))
    for i in range(5):
            drone_direction[i] = drone_direction[i] / np.linalg.norm(drone_direction[i])
        # if np.linalg.norm(Drone_Direction[i]) > 0:
        #     drone_direction = Drone_Direction / np.linalg.norm(Drone_Direction)
    
    # 无人机速度向量
    v_drone_vector = drone_speed * drone_direction
    
    if verbose:
        print(f"无人机1速度向量: {v_drone_vector[0]}")
        print(f"无人机2速度向量: {v_drone_vector[1]}")
        print(f"无人机3速度向量: {v_drone_vector[2]}")
        print(f"无人机4速度向量: {v_drone_vector[3]}")
        print(f"无人机5速度向量: {v_drone_vector[4]}")
    

    # 时间参数
    t_drop = drop_time
    t_blast_delay = blast_delay
    t_blast = t_drop + t_blast_delay

    # 计算位置
    drop_position = np.zeros((5, 3, 3))
    blast_position = np.zeros((5, 3, 3))
    for i in range(5):
        for j in range(3):
            drop_position[i, j, ] = FY_start[i] + v_drone_vector[i] * t_drop[i, j]
            vertical_drop = 0.5 * g * t_blast_delay[i, j]**2
            blast_position[i, j, ] = [drop_position[i, j, 0] + v_drone_vector[i, 0] * t_blast_delay[i, j], 
                                    drop_position[i, j, 1] + v_drone_vector[i, 1] * t_blast_delay[i, j], 
                                    drop_position[i, j, 2] - vertical_drop]

    if verbose:
        print(f"无人机1烟雾弹1投放点位置: {drop_position[0, 0, ]}")
        print(f"无人机1烟雾弹1起爆点位置: {blast_position[0, 0, ]}")
        print(f"无人机1烟雾弹2投放点位置: {drop_position[0, 1, ]}")
        print(f"无人机1烟雾弹2起爆点位置: {blast_position[0, 1, ]}")
        print(f"无人机1烟雾弹3投放点位置: {drop_position[0, 2, ]}")
        print(f"无人机1烟雾弹3起爆点位置: {blast_position[0, 2, ]}")
        print(f"无人机2烟雾弹1投放点位置: {drop_position[1, 0, ]}")
        print(f"无人机2烟雾弹1起爆点位置: {blast_position[1, 0, ]}")
        print(f"无人机2烟雾弹2投放点位置: {drop_position[1, 1, ]}")
        print(f"无人机2烟雾弹2起爆点位置: {blast_position[1, 1, ]}")
        print(f"无人机2烟雾弹3投放点位置: {drop_position[1, 2, ]}")
        print(f"无人机2烟雾弹3起爆点位置: {blast_position[1, 2, ]}")
        print(f"无人机3烟雾弹1投放点位置: {drop_position[2, 0, ]}")
        print(f"无人机3烟雾弹1起爆点位置: {blast_position[2, 0, ]}")
        print(f"无人机3烟雾弹2投放点位置: {drop_position[2, 1, ]}")
        print(f"无人机3烟雾弹2起爆点位置: {blast_position[2, 1, ]}")
        print(f"无人机3烟雾弹3投放点位置: {drop_position[2, 2, ]}")
        print(f"无人机3烟雾弹3起爆点位置: {blast_position[2, 2, ]}")
        print(f"无人机4烟雾弹1投放点位置: {drop_position[3, 0, ]}")
        print(f"无人机4烟雾弹1起爆点位置: {blast_position[3, 0, ]}")
        print(f"无人机4烟雾弹2投放点位置: {drop_position[3, 1, ]}")
        print(f"无人机4烟雾弹2起爆点位置: {blast_position[3, 1, ]}")
        print(f"无人机4烟雾弹3投放点位置: {drop_position[3, 2, ]}")
        print(f"无人机4烟雾弹3起爆点位置: {blast_position[3, 2, ]}")
        print(f"无人机5烟雾弹1投放点位置: {drop_position[4, 0, ]}")
        print(f"无人机5烟雾弹1起爆点位置: {blast_position[4, 0, ]}")
        print(f"无人机5烟雾弹2投放点位置: {drop_position[4, 1, ]}")
        print(f"无人机5烟雾弹2起爆点位置: {blast_position[4, 1, ]}")
        print(f"无人机5烟雾弹3投放点位置: {drop_position[4, 2, ]}")
        print(f"无人机5烟雾弹3起爆点位置: {blast_position[4, 2, ]}")
    

    # 导弹飞行方向（指向假目标）
    missile_direction = np.zeros((3, 3))
    v_missile_vector = np.zeros((3, 3))
    for i in range(3):
        missile_direction[i] = fake_target - M_start[i]
        missile_direction[i] = missile_direction[i] / np.linalg.norm(missile_direction[i, ])
    v_missile_vector = v_missile * missile_direction
    
    if verbose:
        print(f"导弹1速度向量: {v_missile_vector[0]}")
        print(f"导弹1初始位置: {M_start[0]}")
        print(f"导弹2速度向量: {v_missile_vector[1]}")
        print(f"导弹2初始位置: {M_start[1]}")
        print(f"导弹3速度向量: {v_missile_vector[2]}")
        print(f"导弹3初始位置: {M_start[2]}")

    # 计算导弹到达假目标时间
    def time_to_target(num, position, velocity, target):
        t_x = (target[0] - position[num, 0]) / velocity[num, 0]
        return t_x

    t_missile_to_target = np.array([time_to_target(0, M_start, v_missile_vector, fake_target), 
                                    time_to_target(1, M_start, v_missile_vector, fake_target), 
                                    time_to_target(2, M_start, v_missile_vector, fake_target)])
    
    if verbose:
        print(f"导弹1到达假目标时间: {t_missile_to_target[0]:.2f} s")
        print(f"导弹2到达假目标时间: {t_missile_to_target[1]:.2f} s")
        print(f"导弹3到达假目标时间: {t_missile_to_target[2]:.2f} s")

    # 计算导弹在时间t的位置
    def missile_position(num, t):
        return M_start[num] + v_missile_vector[num] * t

    # 计算烟幕云团在时间t的位置（从起爆开始）
    def smoke_position(Dnum, Snum, t_smoke):
        return np.array([blast_position[Dnum, Snum, 0], blast_position[Dnum, Snum, 1], blast_position[Dnum, Snum, 2] - v_smoke_sink * t_smoke])#[Dnum, Snum]

    # 计算角度条件：判断烟幕是否在导弹和真目标之间
    def is_smoke_between(Mnum, Dnum, Snum, missile_pos, smoke_pos, target_pos):
        missile_to_smoke = smoke_pos[Dnum, Snum] - missile_pos[Mnum, Dnum, Snum]
        missile_to_target = target_pos - missile_pos[Mnum, Dnum, Snum]
        
        dot_product = np.dot(missile_to_smoke, missile_to_target)
        cos_angle = dot_product / (np.linalg.norm(missile_to_smoke) * np.linalg.norm(missile_to_target))
        
        return cos_angle > 0

    # 计算点到直线距离
    def distance_point_to_line(point, line_point, line_direction):
        ap = point - line_point
        projection = np.dot(ap, line_direction) / np.linalg.norm(line_direction)
        foot_point = line_point + projection * line_direction
        return np.linalg.norm(point - foot_point)

    # 判断导弹是否经过云团
    def is_missile_through_smoke(Mnum, Dnum, Snum, missile_pos, smoke_pos, missile_prev_pos, smoke_prev_pos):
        radius = effective_radius
        
        missile_move = missile_pos[Mnum, Dnum, Snum] - missile_prev_pos[Mnum, Dnum, Snum]
        missile_move_length = np.linalg.norm(missile_move)
        
        if missile_move_length == 0:
            return False
        
        smoke_move = smoke_pos[Dnum, Snum] - smoke_prev_pos[Dnum, Snum]
        relative_move = missile_move - smoke_move
        relative_move_length = np.linalg.norm(relative_move)
        
        if relative_move_length == 0:
            return np.linalg.norm(missile_pos[Mnum, Dnum, Snum] - smoke_pos[Dnum, Snum]) <= radius
        
        relative_direction = relative_move / relative_move_length
        smoke_to_missile = missile_prev_pos[Mnum, Dnum, Snum] - smoke_prev_pos[Dnum, Snum]
        projection = np.dot(smoke_to_missile, relative_direction)
        perpendicular_dist = np.linalg.norm(smoke_to_missile - projection * relative_direction)
        
        if perpendicular_dist <= radius and 0 <= projection <= relative_move_length:
            return True
        
        if np.linalg.norm(missile_prev_pos[Mnum, Dnum, Snum] - smoke_prev_pos[Dnum, Snum]) <= radius:
            return True
        if np.linalg.norm(missile_pos[Mnum, Dnum, Snum] - smoke_pos[Dnum, Snum]) <= radius:
            return True
        
        return False

    # 计算有效遮蔽时间
    start_time = 0.0
    End_time = np.arange(3 * 5 * 3).reshape(3, 5, 3)
    for i in range(3):
        for j in range(5):
            for k in range(3):
                tx = t_missile_to_target[i] - t_blast[j, k]
                if tx <= 0:
                    tx = 0
                End_time[i, j, k] = min(effective_duration, tx)
    for i in range(3):
        for j in range(5):
            for k in range(3):
                if End_time[i, j, k] <= 0:
                    End_time[i, j, k] = 0
    end_time = np.max(End_time)
    
    time_step = 0.01
    total_effective_time = 0.0
    current_time = start_time
    
    # 存储上一时刻的位置用于判断穿越
    prev_missile_pos = np.zeros((3, 5, 3, 3))
    prev_smoke_pos = np.zeros((5, 3, 3))
    pos_missile = np.zeros((3, 5, 3, 3))
    pos_smoke = np.zeros((5, 3, 3))

    for i in range(3):
        for j in range(5): 
            for k in range(3):
                prev_missile_pos[i, j, k] = missile_position(i, t_blast[j, k])
                prev_smoke_pos[j, k, ] = smoke_position(j, k, 0)
                
                while current_time <= end_time:
                    t_missile = t_blast[j, k] + current_time
                    if t_missile > t_missile_to_target[i]:
                        break
                        
                    pos_missile[i, j, k, ] = missile_position(i, t_missile)
                    pos_smoke[j, k] = smoke_position(j, k, current_time)
                    
                    # 计算距离
                    missile_to_target_direction = real_target - pos_missile[i, j, k, ]
                    missile_to_target_direction = missile_to_target_direction / np.linalg.norm(missile_to_target_direction)
                    distance = distance_point_to_line(pos_smoke[j, k], pos_missile[i, j, k], missile_to_target_direction)
                    
                    # 检查角度条件
                    is_between = is_smoke_between(i, j, k, pos_missile, pos_smoke, real_target)
                    
                    # 检查导弹是否穿过烟幕云团
                    is_through = is_missile_through_smoke(i, j, k, pos_missile, pos_smoke, prev_missile_pos, prev_smoke_pos)
                    
                    # 检查导弹是否在云团内部
                    in_smoke = np.linalg.norm(pos_missile[i, j, k] - pos_smoke[j, k]) <= effective_radius
                    
                    if (distance <= effective_radius and is_between) or is_through or in_smoke:
                        total_effective_time += time_step
                    
                    # 更新上一时刻位置
                    prev_missile_pos[i, j, k, ] = pos_missile[i, j, k, ]
                    prev_smoke_pos[j, k, ] = pos_smoke[j, k, ]
                    
                    current_time += time_step
                
    # if verbose:
    #     print(f"\n=== 计算结果 ===")
    #     print(f"烟幕弹投放时间: {t_drop:.1f} s")
    #     print(f"烟幕弹起爆时间: {t_blast:.1f} s")
    #     print(f"起爆点位置: ({blast_position[0]:.1f}, {blast_position[1]:.1f}, {blast_position[2]:.1f})")
    #     print(f"导弹到达假目标时间: {t_missile_to_target:.2f} s")
    #     print(f"有效遮蔽时长: {total_effective_time:.3f} 秒")

    #     # 详细分析关键时间点
    #     print(f"\n=== 关键时间点详细分析 ===")
    #     for t_check in [0, 1, 2, 3, 4, 5]:
    #         if t_check <= min(effective_duration, t_missile_to_target - t_blast):
    #             t_missile_check = t_blast + t_check
    #             pos_missile = missile_position(t_missile_check)
    #             pos_smoke = smoke_position(t_check)
                
    #             missile_to_target_direction = real_target - pos_missile
    #             missile_to_target_direction = missile_to_target_direction / np.linalg.norm(missile_to_target_direction)
    #             distance = distance_point_to_line(pos_smoke, pos_missile, missile_to_target_direction)
                
    #             is_between = is_smoke_between(pos_missile, pos_smoke, real_target)
    #             direct_distance = np.linalg.norm(pos_missile - pos_smoke)
                
    #             missile_to_smoke = pos_smoke - pos_missile
    #             missile_to_target = real_target - pos_missile
    #             dot_product = np.dot(missile_to_smoke, missile_to_target)
    #             angle_deg = np.degrees(np.arccos(dot_product / (np.linalg.norm(missile_to_smoke) * np.linalg.norm(missile_to_target))))
                
    #             print(f"起爆后 {t_check}s:")
    #             print(f"  导弹位置: {pos_missile}")
    #             print(f"  云团位置: {pos_smoke}")
    #             print(f"  到视线距离: {distance:.2f}m")
    #             print(f"  直接距离: {direct_distance:.2f}m")
    #             print(f"  烟幕在导弹目标之间: {'是' if is_between else '否'}")
    #             print(f"  导弹-烟幕-目标角度: {angle_deg:.1f}°")
    #             print(f"  是否在云团内: {'是' if direct_distance <= effective_radius else '否'}")
    #             print(f"  是否有效遮蔽: {'是' if (distance <= effective_radius and is_between) or direct_distance <= effective_radius else '否'}")

    # if visualize:
    #     # 可视化
    #     fig = plt.figure(figsize=(15, 10))
    #     ax = fig.add_subplot(111, projection='3d')

    #     # 绘制轨迹
    #     missile_times = np.linspace(0, t_missile_to_target, 100)
    #     missile_traj = np.array([missile_position(t) for t in missile_times])
    #     ax.plot(missile_traj[:, 0], missile_traj[:, 1], missile_traj[:, 2], 'r-', label='导弹轨迹', linewidth=2)

    #     # 绘制烟幕下沉轨迹
    #     smoke_times = np.linspace(0, min(effective_duration, t_missile_to_target - t_blast), 50)
    #     smoke_traj = np.array([smoke_position(t) for t in smoke_times])
    #     ax.plot(smoke_traj[:, 0], smoke_traj[:, 1], smoke_traj[:, 2], 'b-', label='烟幕下沉轨迹', linewidth=2)

    #     # 标记关键点
    #     ax.scatter(*fake_target, color='red', s=200, label='假目标', marker='x')
    #     ax.scatter(*real_target, color='blue', s=200, label='真目标', marker='o')
    #     ax.scatter(*M1_start, color='orange', s=100, label='导弹起点')
    #     ax.scatter(*blast_position, color='green', s=100, label='起爆点')

    #     # 设置视角和标签
    #     ax.view_init(elev=20, azim=45)
    #     ax.set_xlabel('X (m)')
    #     ax.set_ylabel('Y (m)')
    #     ax.set_zlabel('Z (m)')
    #     ax.set_title(f'烟幕干扰弹对导弹M1的遮蔽效果分析\n有效遮蔽时间: {total_effective_time:.3f}秒')
    #     ax.legend()
    #     ax.grid(True)

    #     plt.tight_layout()
    #     plt.show()

    return total_effective_time

effective_time = calculate_smoke_obscuration(
        drone_direction = np.array([[-0.5, -0.5, 0], 
                                    [-0.5, -0.5, 0], 
                                    [-0.5, -0.5, 0], 
                                    [-0.5, -0.5, 0], 
                                    [-0.5, -0.5, 0]]),
        drone_speed = np.array([[100, 100, 100],
                                [100, 100, 100],
                                [100, 100, 100],
                                [100, 100, 100],
                                [100, 100, 100]]),
        drop_time = np.array([[10, 15, 20],
                              [10, 15, 20],
                              [10, 15, 20],
                              [10, 15, 20],
                              [10, 15, 20]]),
        blast_delay = np.array([[6, 6, 6],
                                [6, 6, 6],
                                [6, 6, 6],
                                [6, 6, 6],
                                [6, 6, 6]]),
        visualize = True,
        verbose = False
    )
import numpy as np
import tqdm
import random


# 自定义遗传算法实现
class GeneticAlgorithm:
    def __init__(self, pop_size=100, crossover_rate=0.7, mutation_rate=0.2, generations=50):
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.generations = generations
        
        # 参数范围
        self.param_ranges = [
            (0, 63),   # i1的范围
            (0, 63),   # i2的范围
            (0, 63),   # i3的范围
            (0, 63),   # i4的范围
            (0, 63),   # i5的范围
            (80, 120),  # j1的范围
            (80, 120),  # j2的范围
            (80, 120),  # j3的范围
            (80, 120),  # j4的范围
            (80, 120),  # j5的范围
            (0, 30),   # q11的范围
            (0, 15),     # s11的范围
            (0, 30),   # q12的范围
            (0, 15),     # s12的范围
            (0, 30),   # q13的范围
            (0, 15),     # s13的范围
            (0, 30),   # q21的范围
            (0, 15),     # s21的范围
            (0, 30),   # q22的范围
            (0, 15),     # s22的范围
            (0, 30),   # q23的范围
            (0, 15),     # s23的范围
            (0, 30),   # q31的范围
            (0, 15),     # s31的范围
            (0, 30),   # q32的范围
            (0, 15),     # s32的范围
            (0, 30),   # q33的范围
            (0, 15),     # s33的范围
            (0, 30),   # q41的范围
            (0, 15),     # s41的范围
            (0, 30),   # q42的范围
            (0, 15),     # s42的范围
            (0, 30),   # q43的范围
            (0, 15),     # s43的范围
            (0, 30),   # q51的范围
            (0, 15),     # s51的范围
            (0, 30),   # q52的范围
            (0, 15),     # s52的范围
            (0, 30),   # q53的范围
            (0, 15)     # s53的范围
        ]
    
    def create_individual(self):
        # 创建一个个体（一组参数）
        individual = []
        for param_range in self.param_ranges:
            individual.append(random.randint(param_range[0], param_range[1] - 1))
        return individual
    
    def create_population(self):
        # 创建初始种群
        return [self.create_individual() for _ in range(self.pop_size)]
    
    def evaluate(self, individual):
        # 评估个体的适应度
        i1, i2, i3, i4, i5, j1, j2, j3, j4, j5, q11, s11, q12, s12, q13, s13, q21, s21, q22, s22, q23, s23, q31, s31, q32, s32, q33, s33, q41, s41, q42, s42, q43, s43, q51, s51, q52, s52, q53, s53 = individual
        result = calculate_smoke_obscuration(
            drone_direction=np.array([[np.cos(i1/10), np.sin(i1/10), 0], 
                                      [np.cos(i2/10), np.sin(i2/10), 0], 
                                      [np.cos(i3/10), np.sin(i3/10), 0], 
                                      [np.cos(i4/10), np.sin(i4/10), 0], 
                                      [np.cos(i5/10), np.sin(i5/10), 0]]),
            drone_speed=np.array([[j1, j1, j1],
                                  [j2, j2, j2],
                                  [j3, j3, j3],
                                  [j4, j4, j4],
                                  [j5, j5, j5]]),
            drop_time=np.array([[q11, q12, q13], 
                                [q21, q22, q23], 
                                [q31, q32, q33], 
                                [q41, q42, q43], 
                                [q51, q52, q53]]),
            blast_delay=np.array([[s11, s12, s13], 
                                  [s21, s22, s23], 
                                  [s31, s32, s33], 
                                  [s41, s42, s43], 
                                  [s51, s52, s53]]),
            visualize=False,
            verbose=False
        )
        return result  # 返回适应度值
    
    def select(self, population, fitnesses):
        # 锦标赛选择
        selected = []
        for _ in range(self.pop_size):
            # 随机选择3个个体进行竞争
            candidates = random.sample(list(zip(population, fitnesses)), 3)
            # 选择适应度最高的
            winner = max(candidates, key=lambda x: x[1])[0]
            selected.append(winner)
        return selected
    
    def crossover(self, parent1, parent2):
        # 单点交叉
        if random.random() < self.crossover_rate:
            point = random.randint(1, len(parent1) - 1)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
            return child1, child2
        return parent1, parent2
    
    def mutate(self, individual):
        # 均匀变异
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                low, high = self.param_ranges[i]
                mutated[i] = random.randint(low, high - 1)
        return mutated
    
    def run(self):
        # 创建初始种群
        population = self.create_population()
        
        # 存储所有满足条件的解
        solutions = []
        
        print("开始遗传算法优化...")
        for gen in tqdm.tqdm(range(self.generations)):
            # 评估种群中所有个体
            fitnesses = [self.evaluate(ind) for ind in population]
            
            # 记录满足条件的解
            for i, fit in enumerate(fitnesses):
                if fit != 0:
                    solutions.append((population[i], fit))
            
            # 选择
            selected = self.select(population, fitnesses)
            
            # 交叉和变异
            next_population = []
            for i in range(0, self.pop_size, 2):
                parent1 = selected[i]
                parent2 = selected[i+1] if i+1 < self.pop_size else selected[0]
                
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                next_population.extend([child1, child2])
            
            # 确保种群大小不变
            population = next_population[:self.pop_size]
        
        # 去除重复的解
        unique_solutions = []
        seen = set()
        for sol, fit in solutions:
            key = tuple(sol)
            if key not in seen:
                seen.add(key)
                unique_solutions.append((sol, fit))
        
        return unique_solutions

def main():
    # 创建遗传算法实例并运行
    ga = GeneticAlgorithm(pop_size=100, generations=50)
    solutions = ga.run()
    # 输出并选择最优解
    if solutions:
        print(f"\n找到了 {len(solutions)} 个满足条件的候选解（去重后）.")

        # 按适应度（假设值越大越好）排序，选择前 N 个最优解
        solutions_sorted = sorted(solutions, key=lambda x: x[1], reverse=True)
        # solutions_sorted = heapq.nlargest(top_n, solutions, key=lambda x: x[1])
        top_n = min(5, len(solutions_sorted))
        print(f"\n前 {top_n} 个最优解:")
        for rank, (sol, fit) in enumerate(solutions_sorted[:top_n], 1):
            print(f"  排名 {rank}: 适应度={fit:.3f}")
        
        # 打印最优解的详细参数
        best_sol, best_fit = solutions_sorted[0]
        print("\n最优解详细参数:")
        print(f"i1={best_sol[0]},    j1={best_sol[5]},    q11={best_sol[10]},  s11={best_sol[11]},  q12={best_sol[12]},  s12={best_sol[13]},  q13={best_sol[14]},  s13={best_sol[15]}")
        print(f"i2={best_sol[1]},    j2={best_sol[6]},    q21={best_sol[16]},  s21={best_sol[17]},  q22={best_sol[18]},  s22={best_sol[19]},  q23={best_sol[20]},  s23={best_sol[21]}")
        print(f"i3={best_sol[2]},    j3={best_sol[7]},    q31={best_sol[22]},  s31={best_sol[23]},  q32={best_sol[24]},  s32={best_sol[25]},  q33={best_sol[26]},  s33={best_sol[27]}")
        print(f"i4={best_sol[3]},    j4={best_sol[8]},    q41={best_sol[28]},  s41={best_sol[29]},  q42={best_sol[30]},  s42={best_sol[31]},  q43={best_sol[32]},  s43={best_sol[33]}")
        print(f"i5={best_sol[4]},    j5={best_sol[9]},    q51={best_sol[34]},  s51={best_sol[35]},  q52={best_sol[36]},  s52={best_sol[37]},  q53={best_sol[38]},  s53={best_sol[39]}")
        print(f"适应度={best_fit}\n")
    else:
        print("\n未找到满足条件的解")

if __name__ == "__main__":
    main()

