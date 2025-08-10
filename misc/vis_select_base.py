import numpy as np
import open3d as o3d

# 加载点云坐标和特征数据
coord_path = r"D:\04-Datasets\vis\coord.npy"
feat_path = r"D:\04-Datasets\vis\feat.npy"
coord = np.load(coord_path)  # 点云数据的三维坐标
feat = np.load(feat_path)  # 每个点提取的特征（1232维度）

# 将点云坐标转换为Open3D的PointCloud格式
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(coord)


# 定义一个函数，用于计算两个特征向量之间的相似度
def compute_similarity(feat1, feat2):
    # 这里使用余弦相似度作为示例，你可以根据需要选择其他相似度度量方式
    return np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))


# 定义一个回调函数，用于处理鼠标点击事件
def pick_points(pcd, feat):
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # 用户通过鼠标点击选择一个点
    vis.destroy_window()

    # 获取用户点击的点的索引
    picked_points = vis.get_picked_points()
    if len(picked_points) == 0:
        print("没有点被选中")
        return

    picked_point_idx = picked_points[0]
    picked_point_feat = feat[picked_point_idx]

    # 搜索周围相似的点
    similar_points_idx = []
    similarity_threshold = 0.9  # 相似度阈值，可根据需要调整
    for i in range(len(feat)):
        if i != picked_point_idx:
            similarity = compute_similarity(picked_point_feat, feat[i])
            if similarity > similarity_threshold:
                similar_points_idx.append(i)

    # 可视化选中的点和相似的点
    picked_point_color = [1, 0, 0]  # 红色
    similar_points_color = [0, 1, 0]  # 绿色
    pcd.colors = o3d.utility.Vector3dVector(np.zeros_like(coord))
    pcd.colors[picked_point_idx] = picked_point_color
    for idx in similar_points_idx:
        pcd.colors[idx] = similar_points_color

    o3d.visualization.draw_geometries([pcd])


# 调用回调函数
pick_points(pcd, feat)