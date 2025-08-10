import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import tkinter as tk
from tkinter import ttk
import threading
import time

# 全局变量用于存储参数和控制可视化
global_params = {
    'oral': 0.1, # 0.02
    'highlight': 0.3,  # 0.1
    'reject': 0.2,  # 0.5
    'select_index': [1000],
    'update_needed': False,
    'visualization_running': True
}


# 加载数据的函数，便于重新加载
def load_data():
    coord_path = r"D:\04-Datasets\vis\coord.npy"
    feat_path = r"D:\04-Datasets\vis\feat.npy"
    color_path = r"D:\04-Datasets\vis\color.npy"

    coord = np.load(coord_path)  # 点云数据的三维坐标
    feat = np.load(feat_path)  # 每个点提取的特征
    color = np.load(color_path)  # 原始颜色

    # 如果颜色值在0-255范围，需要归一化到0-1
    if color.max() > 1.0:
        color = color / 255.0

    return coord, feat, color


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.clone().detach().cpu().numpy()
    assert isinstance(x, np.ndarray)
    return x


# 函数：让用户选择中心点
def select_center_point(coord, color):
    # 创建点云用于选择
    selection_pcd = o3d.geometry.PointCloud()
    selection_pcd.points = o3d.utility.Vector3dVector(coord)
    selection_pcd.colors = o3d.utility.Vector3dVector(color)

    # 创建可视化器并添加点云
    vis_select = o3d.visualization.VisualizerWithVertexSelection()
    vis_select.create_window(window_name="选择查询点 - 按住Shift并点击, 按Q继续")
    vis_select.add_geometry(selection_pcd)

    print("\n可视化窗口已打开，请选择一个点...")
    vis_select.run()
    picked_points = vis_select.get_picked_points()
    vis_select.destroy_window()

    # 检查是否选择了点
    if picked_points:
        select_index = [picked_points[0].index]
        print(f"\n成功选择查询点，索引: {select_index[0]}")
        print(f"选择的点坐标: {coord[select_index[0]]}")
        return select_index
    else:
        print("\n未选择点，使用默认索引1000")
        return [1000]


# 计算特征相似性
def compute_similarity(feat_tensor, select_index):
    # 特征归一化
    target = F.normalize(feat_tensor, p=2, dim=-1)
    refer = F.normalize(feat_tensor, p=2, dim=-1)

    inner_self = target[select_index] @ target.t()  # 查询点与所有点的相似性
    inner_cross = target[select_index] @ refer.t()  # 同样的计算（因为是同一个点云）

    return inner_self, inner_cross


# 动态阈值设定
def compute_thresholds(inner_cross, oral, highlight, reject):
    sorted_inner = torch.sort(inner_cross, descending=True)[0]
    total_points = inner_cross.shape[1]
    oral_threshold = sorted_inner[0, int(total_points * oral)]
    highlight_threshold = sorted_inner[0, int(total_points * highlight)]
    reject_threshold = sorted_inner[0, -int(total_points * reject)]
    return oral_threshold, highlight_threshold, reject_threshold


# 相似性分数的Sigmoid归一化处理
def normalize_similarity(inner_self, oral_threshold, highlight_threshold, reject_threshold):
    inner_self_normalized = inner_self - highlight_threshold
    inner_self_normalized_pos = inner_self_normalized.clone()
    inner_self_normalized_neg = inner_self_normalized.clone()

    # 处理正值部分
    pos_mask = inner_self_normalized > 0
    if pos_mask.any():
        inner_self_normalized_pos[pos_mask] = F.sigmoid(
            inner_self_normalized[pos_mask] / (oral_threshold - highlight_threshold)
        )

    # 处理负值部分
    neg_mask = inner_self_normalized < 0
    if neg_mask.any():
        inner_self_normalized_neg[neg_mask] = (
                F.sigmoid(inner_self_normalized[neg_mask] / (highlight_threshold - reject_threshold)) * 0.9
        )

    # 合并处理结果
    inner_self_final = torch.where(pos_mask, inner_self_normalized_pos, inner_self_normalized_neg)
    return inner_self_final


# 找到最佳匹配点（除了自身）
def find_best_match(inner_cross, select_index):
    # 创建一个掩码排除查询点本身
    mask = torch.ones(inner_cross.shape[1], dtype=torch.bool)
    mask[select_index[0]] = False
    inner_cross_masked = inner_cross[0, mask]
    matched_index_local = torch.argmax(inner_cross_masked)
    # 转换回原始索引
    matched_index_global = torch.arange(inner_cross.shape[1])[mask][matched_index_local]
    return matched_index_global


# 颜色映射和可视化函数
def update_visualization(coord, color, inner_self, inner_cross, oral_threshold, highlight_threshold, reject_threshold,
                         select_index):
    # 相似性分数的Sigmoid归一化处理
    inner_self_final = normalize_similarity(inner_self, oral_threshold, highlight_threshold, reject_threshold)
    inner_cross_final = inner_self_final.clone()

    # 找到最佳匹配点
    matched_index_global = find_best_match(inner_cross, select_index)

    # 颜色映射
    cmap = plt.get_cmap("Spectral_r")
    local_heat_color = cmap(inner_self_final.squeeze(0).numpy())[:, :3]
    global_heat_color = cmap(inner_cross_final.squeeze(0).numpy())[:, :3]

    # 为了可视化清晰，将点云分为两部分显示
    center_point = coord[select_index[0]]
    # 选择距离查询点较近的点作为局部区域
    distances = np.sum(np.square(coord - center_point), axis=-1)
    point_num = 10000  # 显示局部区域的数量
    local_indices = np.argsort(distances)[:point_num]  # 选择最近的10000个点

    # 创建局部和全局点云
    local_coord = coord[local_indices]
    local_color = local_heat_color[local_indices]
    global_coord = coord
    global_color = global_heat_color

    # 添加偏移以便区分
    bias = np.array([[-3.5, 1, 0]])  # 空间偏移
    local_coord_biased = local_coord + bias

    # 创建可视化对象
    pcds = []

    # 全局点云
    pcd_global = o3d.geometry.PointCloud()
    pcd_global.points = o3d.utility.Vector3dVector(global_coord)
    pcd_global.colors = o3d.utility.Vector3dVector(global_color)
    pcds.append(pcd_global)

    # 局部点云（带偏移）
    pcd_local = o3d.geometry.PointCloud()
    pcd_local.points = o3d.utility.Vector3dVector(local_coord_biased)
    pcd_local.colors = o3d.utility.Vector3dVector(local_color)
    pcds.append(pcd_local)

    # 添加连接线（查询点到最匹配点）
    query_point_biased = coord[select_index[0]] + bias[0]
    matched_point = coord[matched_index_global.item()]
    line_coord = np.array([query_point_biased, matched_point])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_coord)
    line_set.lines = o3d.utility.Vector2iVector(np.array([[0, 1]]))  # 修正线段索引
    line_set.colors = o3d.utility.Vector3dVector(np.array([[0, 0, 0]]))  # 黑色线
    pcds.append(line_set)

    # 突出显示查询点（红色）
    query_point_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    query_point_sphere.paint_uniform_color([1, 0, 0])  # 红色
    query_point_sphere.translate(query_point_biased)
    pcds.append(query_point_sphere)

    # 突出显示最匹配点（蓝色）
    matched_point_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    matched_point_sphere.paint_uniform_color([0, 0, 1])  # 蓝色
    matched_point_sphere.translate(matched_point)
    pcds.append(matched_point_sphere)

    return pcds, inner_cross_final, matched_index_global


# 创建GUI控制面板
class ThresholdController:
    def __init__(self, coord, feat, color):
        self.coord = coord
        self.feat = feat
        self.color = color

        # 初始选择中心点
        self.feat_tensor = torch.from_numpy(feat).float()
        self.select_index = select_center_point(coord, color)

        # 计算初始相似性
        self.inner_self, self.inner_cross = compute_similarity(
            self.feat_tensor, self.select_index
        )

        self.root = tk.Tk()
        self.root.title("点云相似性匹配控制面板")
        self.root.geometry("400x450")

        # 创建参数控制面板
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Oral阈值控制
        ttk.Label(control_frame, text="Oral阈值:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.oral_scale = ttk.Scale(control_frame, from_=0.001, to=0.1, orient=tk.HORIZONTAL,
                                    command=self.on_oral_change, length=200)
        self.oral_scale.grid(row=0, column=1, pady=5)
        self.oral_label = ttk.Label(control_frame, text=f"{global_params['oral']:.3f}")
        self.oral_label.grid(row=0, column=2, padx=(10, 0), pady=5)
        self.oral_scale.set(global_params['oral'])

        # Highlight阈值控制
        ttk.Label(control_frame, text="Highlight阈值:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.highlight_scale = ttk.Scale(control_frame, from_=0.01, to=0.3, orient=tk.HORIZONTAL,
                                         command=self.on_highlight_change, length=200)
        self.highlight_scale.grid(row=1, column=1, pady=5)
        self.highlight_label = ttk.Label(control_frame, text=f"{global_params['highlight']:.3f}")
        self.highlight_label.grid(row=1, column=2, padx=(10, 0), pady=5)
        self.highlight_scale.set(global_params['highlight'])

        # Reject阈值控制
        ttk.Label(control_frame, text="Reject阈值:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.reject_scale = ttk.Scale(control_frame, from_=0.1, to=1.0, orient=tk.HORIZONTAL,
                                      command=self.on_reject_change, length=200)
        self.reject_scale.grid(row=2, column=1, pady=5)
        self.reject_label = ttk.Label(control_frame, text=f"{global_params['reject']:.3f}")
        self.reject_label.grid(row=2, column=2, padx=(10, 0), pady=5)
        self.reject_scale.set(global_params['reject'])

        # 重置按钮
        reset_btn = ttk.Button(control_frame, text="重置默认值", command=self.reset_defaults)
        reset_btn.grid(row=3, column=0, columnspan=3, pady=10)

        # 重新选择中心点按钮
        reselect_btn = ttk.Button(control_frame, text="重新选择中心点", command=self.reselect_center)
        reselect_btn.grid(row=4, column=0, columnspan=3, pady=10)

        # 信息显示区域
        info_frame = ttk.Frame(self.root, padding="10")
        info_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.info_text = tk.Text(info_frame, height=12, width=50)
        self.info_text.grid(row=0, column=0, pady=5)

        # 退出按钮
        exit_btn = ttk.Button(info_frame, text="退出程序", command=self.on_exit)
        exit_btn.grid(row=1, column=0, pady=10)

        # 计算初始阈值
        self.oral_threshold, self.highlight_threshold, self.reject_threshold = compute_thresholds(
            self.inner_cross, global_params['oral'], global_params['highlight'], global_params['reject']
        )

        # 启动可视化线程
        self.vis_thread = threading.Thread(target=self.run_visualization, daemon=True)
        self.vis_thread.start()

    def on_oral_change(self, value):
        global_params['oral'] = float(value)
        self.oral_label.config(text=f"{float(value):.3f}")
        global_params['update_needed'] = True

    def on_highlight_change(self, value):
        global_params['highlight'] = float(value)
        self.highlight_label.config(text=f"{float(value):.3f}")
        global_params['update_needed'] = True

    def on_reject_change(self, value):
        global_params['reject'] = float(value)
        self.reject_label.config(text=f"{float(value):.3f}")
        global_params['update_needed'] = True

    def reset_defaults(self):
        global_params['oral'] = 0.02
        global_params['highlight'] = 0.1
        global_params['reject'] = 0.5
        self.oral_scale.set(0.02)
        self.highlight_scale.set(0.1)
        self.reject_scale.set(0.5)
        self.oral_label.config(text="0.020")
        self.highlight_label.config(text="0.100")
        self.reject_label.config(text="0.500")
        global_params['update_needed'] = True

    def reselect_center(self):
        """重新选择中心点"""
        # 打开选择窗口
        new_select_index = select_center_point(self.coord, self.color)

        if new_select_index:
            # 更新选择点索引
            self.select_index = new_select_index
            print(f"\n重新选择查询点，新索引: {self.select_index[0]}")
            print(f"新选择的点坐标: {self.coord[self.select_index[0]]}")

            # 更新相似性计算
            self.inner_self, self.inner_cross = compute_similarity(
                self.feat_tensor, self.select_index
            )

            # 计算阈值
            self.oral_threshold, self.highlight_threshold, self.reject_threshold = compute_thresholds(
                self.inner_cross, global_params['oral'], global_params['highlight'], global_params['reject']
            )

            # 设置更新标志
            global_params['update_needed'] = True

    def on_exit(self):
        global_params['visualization_running'] = False
        self.root.quit()
        self.root.destroy()

    def update_info(self, oral_threshold, highlight_threshold, reject_threshold,
                    inner_cross_final, matched_index_global):
        self.info_text.delete(1.0, tk.END)
        info_str = f"""阈值设定:
  Oral阈值: {oral_threshold:.4f}
  Highlight阈值: {highlight_threshold:.4f}
  Reject阈值: {reject_threshold:.4f}

查询点信息:
  查询点索引: {self.select_index[0]}
  坐标位置: {self.coord[self.select_index[0]]}
  最匹配点索引: {matched_index_global.item()}

统计信息:
  总点数: {len(self.coord)}
  相似性分数范围: [{inner_cross_final.min():.4f}, {inner_cross_final.max():.4f}]
"""
        self.info_text.insert(1.0, info_str)

    def run_visualization(self):
        try:
            # 创建可视化窗口
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="点云相似性匹配结果 - 实时调节阈值", width=1200, height=800)

            # 初始化几何体
            pcds, inner_cross_final, matched_index_global = update_visualization(
                self.coord, self.color,
                self.inner_self, self.inner_cross,
                self.oral_threshold, self.highlight_threshold, self.reject_threshold,
                self.select_index
            )

            for pcd in pcds:
                vis.add_geometry(pcd)

            # 第一次更新信息
            self.update_info(
                self.oral_threshold, self.highlight_threshold, self.reject_threshold,
                inner_cross_final, matched_index_global
            )

            while global_params['visualization_running']:
                # 检查参数是否更新
                if global_params['update_needed']:
                    # 重新计算阈值
                    self.oral_threshold, self.highlight_threshold, self.reject_threshold = compute_thresholds(
                        self.inner_cross, global_params['oral'], global_params['highlight'], global_params['reject']
                    )

                    # 更新可视化
                    pcds, inner_cross_final, matched_index_global = update_visualization(
                        self.coord, self.color,
                        self.inner_self, self.inner_cross,
                        self.oral_threshold, self.highlight_threshold, self.reject_threshold,
                        self.select_index
                    )

                    # 清除旧几何体
                    for pcd in pcds:
                        vis.remove_geometry(pcd, reset_bounding_box=False)

                    # 添加新几何体
                    for pcd in pcds:
                        vis.add_geometry(pcd, reset_bounding_box=False)

                    # 更新信息显示
                    self.update_info(
                        self.oral_threshold, self.highlight_threshold, self.reject_threshold,
                        inner_cross_final, matched_index_global
                    )

                    global_params['update_needed'] = False

                # 更新可视化
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.01)  # 短暂延迟以减少CPU使用

            vis.destroy_window()
        except Exception as e:
            print(f"可视化线程出错: {e}")

    def run(self):
        self.root.mainloop()


# 主程序
if __name__ == "__main__":
    # 加载数据
    coord, feat, color = load_data()
    print(f"点云坐标形状: {coord.shape}")
    print(f"特征形状: {feat.shape}")
    print(f"颜色形状: {color.shape}")

    # 启动GUI控制面板
    controller = ThresholdController(coord, feat, color)
    controller.run()