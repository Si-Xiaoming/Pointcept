import open3d as o3d
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from sklearn.preprocessing import StandardScaler

from collections import Counter


def load_and_process_data(coord_path, segment_path, feat_path, color_path=None):
    """
    加载点云数据
    """
    coord = np.load(coord_path)
    segment = np.load(segment_path)
    feat = np.load(feat_path)
    print(f"Coord shape: {coord.shape}")
    print(f"Segment shape: {segment.shape}")
    print(f"Feat shape: {feat.shape}")
    print(f"Unique labels: {np.unique(segment)}")

    if color_path is not None:
        color = np.load(color_path)
        # 如果有颜色数据，将其添加到特征中
        return coord, segment, feat, color
    else:
        # 如果没有颜色数据，返回特征和坐标
        return coord, segment, feat
    






def sample_points_by_class(feat, segment, max_points_per_class=500):
    """
    根据类别标签对特征进行采样，每个类别最多保留max_points_per_class个点
    """
    unique_labels = np.unique(segment)
    sampled_indices = []
    sampled_labels = []

    for label in unique_labels:
        # 找到该类别的所有点的索引
        label_indices = np.where(segment == label)[0]

        # 如果该类别的点数超过限制，则随机采样
        if len(label_indices) > max_points_per_class:
            selected_indices = np.random.choice(label_indices,
                                                size=max_points_per_class,
                                                replace=False)
        else:
            selected_indices = label_indices

        sampled_indices.extend(selected_indices)
        sampled_labels.extend([label] * len(selected_indices))

    # 提取采样后的特征和标签
    sampled_feat = feat[sampled_indices]
    sampled_segment = np.array(sampled_labels)

    print(f"采样后总点数: {len(sampled_indices)}")
    print("各类别采样后的点数分布:")
    label_counts = Counter(sampled_segment)
    for label, count in sorted(label_counts.items()):
        print(f"  类别 {label}: {count} 个点")

    return sampled_feat, sampled_segment, sampled_indices


def compute_pca_then_tsne(feat, n_pca_components=50, n_tsne_components=2,
                          tsne_perplexity=30, random_state=42):
    """
    先进行PCA降维，再进行T-SNE降维
    """
    print(f"原始特征维度: {feat.shape[1]}")

    # 1. 标准化特征
    print("正在进行特征标准化...")
    scaler = StandardScaler()
    feat_normalized = scaler.fit_transform(feat)

    # 2. PCA降维
    print(f"正在进行PCA降维 (目标维度: {n_pca_components})...")
    pca = PCA(n_components=n_pca_components, random_state=random_state)
    feat_pca = pca.fit_transform(feat_normalized)

    print(f"PCA降维后维度: {feat_pca.shape[1]}")
    print(f"PCA解释的方差比例: {pca.explained_variance_ratio_.sum():.4f}")

    # 3. T-SNE降维
    print("正在进行T-SNE降维...")
    tsne = TSNE(n_components=n_tsne_components,
                perplexity=tsne_perplexity,
                random_state=random_state,
                verbose=1)
    feat_tsne = tsne.fit_transform(feat_pca)

    print("PCA + T-SNE计算完成!")
    return feat_tsne, feat_pca, pca


def plot_tsne_results(feat_tsne, segment, title="PCA + T-SNE Visualization", save_path=None):
    """
    绘制T-SNE结果
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建颜色映射
    unique_labels = np.unique(segment)
    n_colors = len(unique_labels)

    # 根据类别数量选择合适的颜色映射
    if n_colors <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, n_colors))
    elif n_colors <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, n_colors))
    else:
        colors = plt.cm.nipy_spectral(np.linspace(0, 1, n_colors))

    color_map = dict(zip(unique_labels, colors))

    # 绘图
    plt.figure(figsize=(12, 10))

    # 为每个类别绘制散点图
    for i, label in enumerate(sorted(unique_labels)):
        mask = segment == label
        plt.scatter(feat_tsne[mask, 0], feat_tsne[mask, 1],
                    c=[color_map[label]], label=f'类别 {label}',
                    alpha=0.7, s=20, edgecolors='none')

    plt.title(title, fontsize=16)
    plt.xlabel('T-SNE Component 1', fontsize=12)
    plt.ylabel('T-SNE Component 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")

    plt.show()


def plot_pca_explained_variance(feat, max_components=100):
    """
    绘制PCA解释方差比例图，帮助选择合适的PCA维度
    """
    # 标准化
    scaler = StandardScaler()
    feat_normalized = scaler.fit_transform(feat)

    # 计算PCA
    n_components = min(max_components, feat.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(feat_normalized)

    # 绘制累计解释方差比例
    plt.figure(figsize=(10, 6))
    cumsum_variance = np.cumsum(pca.explained_variance_ratio_)

    plt.plot(range(1, len(cumsum_variance) + 1), cumsum_variance, 'b-', linewidth=2)
    plt.xlabel('PCA Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('PCA Explained Variance Ratio')
    plt.grid(True, alpha=0.3)

    # 添加参考线
    plt.axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='80% Variance')
    plt.axhline(y=0.9, color='orange', linestyle='--', alpha=0.7, label='90% Variance')
    plt.axhline(y=0.95, color='green', linestyle='--', alpha=0.7, label='95% Variance')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 打印建议的PCA维度
    for threshold in [0.8, 0.9, 0.95]:
        n_comp = np.argmax(cumsum_variance >= threshold) + 1
        print(f"达到 {threshold * 100}% 方差需要 {n_comp} 个主成分")


def analyze_class_distribution(segment):
    """
    分析各类别的分布情况
    """
    label_counts = Counter(segment)
    print("\n原始数据各类别分布:")
    for label, count in sorted(label_counts.items()):
        print(f"  类别 {label}: {count} 个点")

    return label_counts


def vis_color(coord, feat, segment, color=None):
    """
    可视化点云数据
    """
    if color is None:
        color = np.random.rand(len(segment), 3)  # 随机颜色

    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    pcd.colors = o3d.utility.Vector3dVector(color)

    # 可视化
    o3d.visualization.draw_geometries([pcd], window_name="Point Cloud Visualization")

def main_vis():
    print("--------start----------")
    # 1. load data
    coord_path = r"D:\04-Datasets\vis\coord.npy"
    feat_path = r"D:\04-Datasets\vis\feat.npy"
    color = r"D:\04-Datasets\vis\color.npy"
    segment_path = r"D:\04-Datasets\vis\segment.npy"
    # 加载数据
    coord, segment, feat, color = load_and_process_data(coord_path, segment_path, feat_path, color)
    # 可视化点云数据
    vis_color(coord, feat, segment, color)

def main_process():

    print("--------start----------")
    # 1. load data
    coord_path = r"D:\04-Datasets\vis\coord.npy"
    feat_path = r"D:\04-Datasets\vis\feat.npy"
    color = r"D:\04-Datasets\vis\color.npy"
    segment_path = r"D:\04-Datasets\vis\segment.npy"


    # 加载数据
    coord, segment, feat = load_and_process_data(coord_path, segment_path, feat_path)

    # 分析原始分布
    analyze_class_distribution(segment)

    # 绘制PCA方差解释比例图（可选）
    print("\n正在分析PCA维度选择...")
    plot_pca_explained_variance(feat, max_components=100)

    # 按类别采样（每个类别最多500个点）
    print("\n正在进行类别采样...")
    sampled_feat, sampled_segment, sampled_indices = sample_points_by_class(
        feat, segment, max_points_per_class=2000
    )

    # 先PCA再T-SNE
    print("\n正在进行PCA + T-SNE计算...")
    feat_tsne, feat_pca, pca_model = compute_pca_then_tsne(
        sampled_feat,
        n_pca_components=50,  # 可以根据上面的方差图调整
        n_tsne_components=2,
        tsne_perplexity=30
    )

    # 可视化结果
    name = "std"
    epoch = 5
    plot_tsne_results(feat_tsne, sampled_segment,
                      title="PCA + T-SNE Visualization (每类别最多500个点)",
                      save_path=f"D:\\04-Datasets\\vis\\pca_tsne_{name}_{epoch}.png")

    # 保存结果
    # np.save('sampled_feat.npy', sampled_feat)
    # np.save('sampled_segment.npy', sampled_segment)
    # np.save('sampled_indices.npy', sampled_indices)
    # np.save('feat_pca.npy', feat_pca)
    # np.save('feat_tsne.npy', feat_tsne)

    print("处理完成！")
    print(f"最终T-SNE特征形状: {feat_tsne.shape}")
    print(f"PCA特征形状: {feat_pca.shape}")



if __name__ == "__main__":
    # main_vis()
    main_process()








