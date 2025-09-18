import numpy as np
import open3d as o3d
import copy
import os

import time
# 创建一个状态类来存储选择结果
class SelectionState:
    def __init__(self):
        self.selected_point = None
        self.selected_index = None


class DensityPerturbationViewGenerator(object):
    def __init__(
            self,
            global_view_num=2,
            global_view_size=(10.0, 30.0),
            local_view_num=4,
            local_view_size=(2.0, 10.0),
            max_size=65536,
            center_height_scale=(0.2, 0.8),
            shared_global_view=False,
            view_keys=("coord", "origin_coord", "intensity"),
            shape_type="cube",
            # 密度扰动参数 - 简化为固定范围
            density_perturbation=True,
            density_perturbation_prob=0.8,
            density_factor_range=(0.5, 2.0),  # 表示将点数调整到原数量的 50% 到 200%
            grid_size=0.1,  # 添加 grid_size 参数
    ):
        # 基本参数初始化
        self.global_view_num = global_view_num
        self.global_view_size = global_view_size
        self.local_view_num = local_view_num
        self.local_view_size = local_view_size

        self.max_size = max_size
        self.center_height_scale = center_height_scale
        self.shared_global_view = shared_global_view
        self.view_keys = view_keys
        self.shape_type = shape_type
        assert "coord" in view_keys

        # 密度扰动参数
        self.density_perturbation = density_perturbation
        self.density_perturbation_prob = density_perturbation_prob
        self.density_factor_range = density_factor_range
        self.grid_size = grid_size

    # 通过重复索引实现上采样
    def simple_upsample(self, indices, num_to_add):
        """非常简单的上采样：仅通过重复采样现有点索引来增加点数"""
        if num_to_add <= 0 or len(indices) == 0:
            return indices

        # 通过有放回抽样增加索引数量，模拟上采样
        additional_indices = np.random.choice(indices, num_to_add, replace=True)
        return np.concatenate([indices, additional_indices])

    def perturb_density(self, indices):
        """实现密度扰动 - 使用简化的基于点数的策略"""
        if not self.density_perturbation or np.random.rand() > self.density_perturbation_prob or len(indices) == 0:
            # 如果超过最大点数，仍需处理
            if len(indices) > self.max_size:
                return np.random.choice(indices, self.max_size, replace=False)
            return indices

        # --- 简化密度信息获取 ---
        # 1. 获取视图内的点数
        num_points = len(indices)

        # --- 决定扰动策略 ---
        # 随机选择密度变化因子 (相对于当前点数的比例)
        density_factor = np.random.uniform(*self.density_factor_range)

        # 计算目标点数
        target_num_points = int(num_points * density_factor)
        target_num_points = max(1, target_num_points)  # 至少保留一个点

        # 应用采样策略
        if target_num_points < num_points:
            # 下采样 (快速)
            final_num_points = min(target_num_points, self.max_size)
            return np.random.choice(indices, final_num_points, replace=False)

        elif target_num_points > num_points:
            # 上采样 (使用简单复制以提高速度)
            num_to_add = target_num_points - num_points
            max_addable = self.max_size - num_points
            num_to_add = min(num_to_add, max_addable)

            if num_to_add > 0:
                # 使用简单的复制上采样
                return self.simple_upsample(indices, num_to_add)
            else:
                return indices
        else:
            # 密度因子约为1，不改变点数，但仍需检查 max_size
            if len(indices) > self.max_size:
                return np.random.choice(indices, self.max_size, replace=False)
            return indices

    def get_view_by_size(self, point, center, size):
        """获取基于物理尺寸的视图，包含密度扰动"""
        coord = point["coord"]

        # 计算距离中心点在指定物理尺寸范围内的点
        if self.shape_type == "cube":
            x_mask = np.abs(coord[:, 0] - center[0]) < size / 2
            y_mask = np.abs(coord[:, 1] - center[1]) < size / 2
            mask = np.logical_and(x_mask, y_mask)
        else:  # sphere
            distances = np.sqrt(np.sum((coord[:, :2] - center[:2]) ** 2, axis=1))
            mask = distances < size / 2

        indices = np.where(mask)[0]

        # 如果点太少，返回空视图
        if len(indices) == 0:
            return None

        # 应用密度扰动 - 作为固定数据预处理
        indices = self.perturb_density(indices)

        # 确保不超过max_size (perturb_density 内部已处理，这里是双重保险)
        if len(indices) > self.max_size:
            indices = np.random.choice(indices, self.max_size, replace=False)

        # 创建视图
        view = dict(index=indices)
        for key in point.keys():
            if key in self.view_keys:
                view[key] = point[key][indices]

        if "index_valid_keys" in point.keys():
            view["index_valid_keys"] = point["index_valid_keys"]

        return view

    def __call__(self, data_dict):
        # 原始视图生成逻辑保持不变
        coord = data_dict["coord"]
        point = copy.deepcopy(data_dict)

        # 创建高度掩码
        z_min = coord[:, 2].min()
        z_max = coord[:, 2].max()
        z_min_ = z_min + (z_max - z_min) * self.center_height_scale[0]
        z_max_ = z_min + (z_max - z_min) * self.center_height_scale[1]
        center_mask = np.logical_and(coord[:, 2] >= z_min_, coord[:, 2] <= z_max_)

        # 选择一个合理的中心点（避免边缘)
        valid_indices = np.where(center_mask)[0]
        if len(valid_indices) == 0:
            return data_dict

        # 确保中心点不在边缘
        max_global_size = max(self.global_view_size)
        edge_buffer = max_global_size / 2
        x_min, y_min = np.min(coord[:, :2], axis=0)
        x_max, y_max = np.max(coord[:, :2], axis=0)

        # 过滤掉靠近边缘的点
        edge_mask = np.logical_and(
            np.logical_and(coord[:, 0] >= x_min + edge_buffer, coord[:, 0] <= x_max - edge_buffer),
            np.logical_and(coord[:, 1] >= y_min + edge_buffer, coord[:, 1] <= y_max - edge_buffer)
        )
        center_mask = np.logical_and(center_mask, edge_mask)

        valid_indices = np.where(center_mask)[0]
        if len(valid_indices) == 0:
            # 如果没有足够远离边缘的点，放宽条件
            valid_indices = np.where(np.logical_and(coord[:, 2] >= z_min_, coord[:, 2] <= z_max_))[0]

        if len(valid_indices) == 0:
            return data_dict

        # 获取主要全局视图
        # 修改这里：使用交互式选择的中心点，而不是随机选择
        # major_center = coord[np.random.choice(valid_indices)]
        major_center = self.interactive_select_center(coord, valid_indices)
        if major_center is None:  # 用户取消选择
            return data_dict

        global_size = np.random.uniform(*self.global_view_size)
        major_view = self.get_view_by_size(point, major_center, global_size)

        if major_view is None:
            return data_dict

        major_coord = major_view["coord"]

        # 获取其他全局视图
        global_views = []
        for _ in range(self.global_view_num - 1):
            if major_coord.shape[0] == 0:
                break
            center_idx = np.random.randint(major_coord.shape[0])
            center = major_coord[center_idx]
            size = np.random.uniform(*self.global_view_size)
            view = self.get_view_by_size(point, center, size)
            if view is not None:
                global_views.append(view)

        if len(global_views) < self.global_view_num - 1:
            # 如果无法生成足够的全局视图，用major_view填充
            while len(global_views) < self.global_view_num - 1:
                global_views.append({key: value.copy() for key, value in major_view.items()})

        global_views = [major_view] + global_views

        # 获取局部视图
        cover_mask = np.zeros_like(major_view["index"], dtype=bool)
        local_views = []
        for i in range(self.local_view_num):
            if sum(~cover_mask) == 0:
                # 重置覆盖掩码
                cover_mask[:] = False

            if major_coord.shape[0] == 0:
                break

            # 优先选择未覆盖区域的点
            available_indices = np.where(~cover_mask)[0]
            if len(available_indices) == 0:
                available_indices = np.arange(major_coord.shape[0])

            center_idx = np.random.choice(available_indices)
            center = major_coord[center_idx]
            size = np.random.uniform(*self.local_view_size)
            local_view = self.get_view_by_size(point, center, size)

            if local_view is not None:
                local_views.append(local_view)
                # 更新覆盖掩码
                local_indices = local_view["index"]
                major_indices = major_view["index"]
                mask = np.isin(major_indices, local_indices)
                cover_mask[mask] = True

        # 应用变换和拼接
        view_dict = {}
        for global_view in global_views:
            if global_view is None:
                continue
            global_view.pop("index")

            for key in self.view_keys:
                if f"global_{key}" in view_dict.keys():
                    view_dict[f"global_{key}"].append(global_view[key])
                else:
                    view_dict[f"global_{key}"] = [global_view[key]]

        if "global_coord" in view_dict and len(view_dict["global_coord"]) > 0:
            view_dict["global_offset"] = np.cumsum(
                [data.shape[0] for data in view_dict["global_coord"]]
            )

        for local_view in local_views:
            if local_view is None:
                continue
            local_view.pop("index")
            local_view = local_view
            for key in self.view_keys:
                if f"local_{key}" in view_dict.keys():
                    view_dict[f"local_{key}"].append(local_view[key])
                else:
                    view_dict[f"local_{key}"] = [local_view[key]]

        if "local_coord" in view_dict and len(view_dict["local_coord"]) > 0:
            view_dict["local_offset"] = np.cumsum(
                [data.shape[0] for data in view_dict["local_coord"]]
            )

        for key in view_dict.keys():
            if "offset" not in key and len(view_dict[key]) > 0:
                view_dict[key] = np.concatenate(view_dict[key], axis=0)

        data_dict.update(view_dict)
        return data_dict

    def interactive_select_center(self, coord, valid_indices):
        """交互式选择中心点"""
        print("\n交互式选择中心点模式:")
        print("1. 在点云上点击选择一个中心点")
        print("2. 按住Ctrl键并点击可选择多个点")
        print("3. 选择后，程序将使用该点作为主要全局视图的中心")
        print("4. 按'q'退出选择模式（将使用随机点）")

        # 创建点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coord)

        # 创建可视化器（专门用于顶点选择）
        vis = o3d.visualization.VisualizerWithVertexSelection()
        vis.create_window(window_name="选择中心点", width=1000, height=800)

        # 添加点云
        vis.add_geometry(pcd)

        # 设置渲染选项
        opt = vis.get_render_option()
        opt.point_size = 3.0
        opt.background_color = np.asarray([0.1, 0.1, 0.1])

        # 添加坐标轴
        axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
        vis.add_geometry(axis_pcd)

        print("\n请在点云上点击选择一个中心点...")
        print("提示：按住Ctrl键并点击可以选择点，选择后会在控制台显示点号")

        # 创建状态对象来存储选择结果
        state = SelectionState()

        # 直接在方法内部定义回调函数（关键修改）
        def selection_callback():
            # 获取选中的顶点
            selected = vis.get_picked_points()

            if selected and len(selected) > 0:
                # 获取第一个选中的点
                picked_point = selected[0]
                point_idx = picked_point.index

                # 检查点是否在有效区域内
                if point_idx in valid_indices:
                    print(f"\n✓ 已选择有效点: 索引={point_idx}, 坐标={coord[point_idx]}")
                    print("  按'q'确认选择并继续，或选择其他点")
                else:
                    print(f"\n✗ 选择的点(索引={point_idx})不在有效高度范围内")
                    print("  请在绿色区域（高度适中区域）内选择点")

                # 高亮显示选中的点
                highlight_pcd = o3d.geometry.PointCloud()
                highlight_pcd.points = o3d.utility.Vector3dVector([coord[point_idx]])
                highlight_pcd.paint_uniform_color([1, 0, 0])  # 红色

                # 移除之前的高亮点（如果有）
                try:
                    vis.remove_geometry("highlight_point")
                except:
                    pass

                # 添加新的高亮点
                vis.add_geometry(highlight_pcd, reset_bounding_box=False)

                # 保存选择结果
                state.selected_point = coord[point_idx]
                state.selected_index = point_idx

        # 注册回调函数
        vis.register_selection_changed_callback(selection_callback)

        # 运行可视化并等待选择
        vis.run()
        vis.destroy_window()

        # 检查是否有选择
        if state.selected_point is not None:
            print(f"\n已确认选择: 索引={state.selected_index}, 坐标={state.selected_point}")
            return state.selected_point
        else:
            print("\n未选择点，将使用随机点作为中心")
            return coord[np.random.choice(valid_indices)]


# ======================
# 修复后的可视化工具函数
# ======================
def create_open3d_point_cloud(coords: np.ndarray, colors: np.ndarray = None) -> o3d.geometry.PointCloud:
    """创建Open3D点云对象 (已修复颜色覆盖问题)"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)

    if colors is not None:
        # 确保颜色在[0,1]范围内
        if colors.max() > 1.0:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def visualize_view(pcd, window_name, background_color, point_size=3.0):
    """可视化单个视图（非阻塞模式）"""
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=800, height=600)

    # 添加点云
    vis.add_geometry(pcd)

    # 添加坐标轴
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
    vis.add_geometry(axis_pcd)

    # 设置渲染选项
    opt = vis.get_render_option()
    opt.point_size = point_size
    opt.background_color = background_color

    # 更新视图
    vis.poll_events()
    vis.update_renderer()

    return vis


def visualize_views(original_coords: np.ndarray,
                    original_colors: np.ndarray,
                    generator: DensityPerturbationViewGenerator,
                    output_dir: str = "view_visualizations"):
    """可视化原始点云和生成的视图 - 所有视图一次性显示"""
    # 创建原始点云
    original_pcd = create_open3d_point_cloud(original_coords, original_colors)

    # 准备数据字典
    data_dict = {
        "coord": original_coords.copy(),
        "color": original_colors.copy(),
        "origin_coord": original_coords.copy(),
    }

    # 应用视图生成器
    processed_data = generator(data_dict)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 设置背景颜色 (RGB: 169,233,228)
    background_color = np.array([169 / 255, 233 / 255, 228 / 255])

    # 存储所有可视化器对象
    visualizers = []

    # 可视化原始点云
    print("显示原始点云...")
    vis_original = visualize_view(original_pcd, "原始点云", background_color)
    visualizers.append(vis_original)

    # 提取生成的视图
    view_count = 1

    # 全局视图
    if "global_coord" in processed_data:
        global_offset = processed_data.get("global_offset", [])
        start = 0
        for i, end in enumerate(global_offset):
            coords = processed_data["global_coord"][start:end]
            colors = processed_data["global_color"][start:end] if "global_color" in processed_data else None

            # 创建点云对象
            pcd = create_open3d_point_cloud(coords, colors)

            # 可视化全局视图
            print(f"显示全局视图 {i + 1}...")
            vis_global = visualize_view(pcd, f"全局视图 {i + 1}", background_color)
            visualizers.append(vis_global)

            start = end
            view_count += 1

    # 局部视图
    if "local_coord" in processed_data:
        local_offset = processed_data.get("local_offset", [])
        start = 0
        for i, end in enumerate(local_offset):
            coords = processed_data["local_coord"][start:end]
            colors = processed_data["local_color"][start:end] if "local_color" in processed_data else None

            # 创建点云对象
            pcd = create_open3d_point_cloud(coords, colors)

            # 可视化局部视图
            print(f"显示局部视图 {i + 1}...")
            vis_local = visualize_view(pcd, f"局部视图 {i + 1}", background_color)
            visualizers.append(vis_local)

            start = end
            view_count += 1

    print("\n所有视图已显示！")
    print("请查看打开的窗口查看点云视图")
    print("按'q'键关闭当前窗口")

    # 保持程序运行，直到所有窗口关闭
    try:
        while any(vis.poll_events() for vis in visualizers):
            for vis in visualizers:
                if vis.poll_events():
                    vis.update_renderer()
            time.sleep(0.05)
    finally:
        # 关闭所有窗口
        for vis in visualizers:
            vis.destroy_window()
        print("所有窗口已关闭")


# ======================
# 主执行函数
# ======================
def main():
    # dir_path = "/datasets/paper/processed/test/test01/"
    dir_path = r"D:\04-Datasets\paper\processed\test\test01"
    # 1. 加载点云数据
    try:
        coords = np.load(os.path.join(dir_path, "coord.npy"))
        colors = np.load(os.path.join(dir_path, "color.npy"))
        print(f"成功加载点云: {coords.shape[0]}个点")
    except FileNotFoundError:
        print("错误: 未找到 coord.npy 或 color.npy 文件")
        print("请确保文件位于当前目录")
        return

    # 2. 验证数据格式
    if coords.ndim != 2 or coords.shape[1] != 3:
        print(f"错误: coord.npy 格式不正确 (期望Nx3, 实际{coords.shape})")
        return
    if colors.ndim != 2 or colors.shape[0] != coords.shape[0]:
        print(f"错误: color.npy 格式不正确 (期望Nx3, 实际{colors.shape})")
        return
    if colors.shape[1] == 3:
        print("检测到RGB颜色数据")
    elif colors.shape[1] == 1:
        print("检测到单通道强度数据，将转换为灰度")
        colors = np.repeat(colors, 3, axis=1)
    else:
        print(f"错误: 不支持的颜色格式 (通道数={colors.shape[1]})")
        return

    # 3. 创建视图生成器 (使用更合理的参数)
    generator = DensityPerturbationViewGenerator(
        global_view_num=2,
        global_view_size=(25.0, 45.0),
        local_view_num=4,
        local_view_size=(10.0, 15.0),  # 修复：增大局部视图尺寸
        density_perturbation=True,
        density_perturbation_prob=0.8,
        density_factor_range=(0.5, 1.5),
        grid_size=0.1,
        view_keys=("coord", "color")
    )

    # 4. 可视化结果
    print("\n正在生成可视化... (请稍候)")
    visualize_views(
        original_coords=coords,
        original_colors=colors,
        generator=generator,
        output_dir="view_visualizations"
    )

    print("\n可视化完成! 所有视图截图保存在 view_visualizations 目录")


if __name__ == "__main__":
    main()