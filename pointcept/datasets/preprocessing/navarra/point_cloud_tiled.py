import os
import pdal
import json
import numpy as np
from pathlib import Path


def tile_las_files(input_folder, output_folder, tile_size=100):
    """
    处理文件夹下所有LAS文件，按指定大小进行切割

    Parameters:
    input_folder: 输入LAS文件夹路径
    output_folder: 输出文件夹路径
    tile_size: 切割大小（米），默认100米
    """

    # 创建输出文件夹
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # 获取所有LAS文件
    input_path = Path(input_folder)
    las_files = list(input_path.glob("*.las")) + list(input_path.glob("*.laz"))

    if not las_files:
        print(f"在 {input_folder} 中未找到LAS文件")
        return

    print(f"找到 {len(las_files)} 个LAS文件")

    # 处理每个LAS文件
    for las_file in las_files:
        print(f"处理文件: {las_file.name}")
        process_single_las(las_file, output_folder, tile_size)


def process_single_las(las_file_path, output_folder, tile_size=100):
    """
    处理单个LAS文件，按tile_size进行切割
    """

    try:
        # 读取LAS文件信息
        pipeline_json = {
            "pipeline": [
                str(las_file_path),
                {
                    "type": "filters.info"
                }
            ]
        }

        # 创建并执行管道以获取文件边界信息
        pipeline = pdal.Pipeline(json.dumps(pipeline_json))
        pipeline.execute()

        # 获取点云边界
        metadata = pipeline.metadata
        bounds = metadata['metadata']['readers.las']['bounds']

        # 解析边界坐标
        min_x = bounds['X']['minimum']
        max_x = bounds['X']['maximum']
        min_y = bounds['Y']['minimum']
        max_y = bounds['Y']['maximum']

        print(f"  文件边界: X({min_x:.2f}, {max_x:.2f}), Y({min_y:.2f}, {max_y:.2f})")

        # 计算切割网格
        x_tiles = np.arange(np.floor(min_x / tile_size) * tile_size,
                            np.ceil(max_x / tile_size) * tile_size + tile_size,
                            tile_size)
        y_tiles = np.arange(np.floor(min_y / tile_size) * tile_size,
                            np.ceil(max_y / tile_size) * tile_size + tile_size,
                            tile_size)

        print(f"  将切割为 {len(x_tiles) - 1} x {len(y_tiles) - 1} = {(len(x_tiles) - 1) * (len(y_tiles) - 1)} 个瓦片")

        # 对每个瓦片进行处理
        tile_count = 0
        for i in range(len(x_tiles) - 1):
            for j in range(len(y_tiles) - 1):
                x_min = x_tiles[i]
                x_max = x_tiles[i + 1]
                y_min = y_tiles[j]
                y_max = y_tiles[j + 1]

                # 构建输出文件名
                base_name = las_file_path.stem
                output_filename = f"{base_name}_tile_{int(x_min)}_{int(y_min)}.las"
                output_path = os.path.join(output_folder, output_filename)

                # 创建切割管道
                pipeline_json = {
                    "pipeline": [
                        str(las_file_path),
                        {
                            "type": "filters.crop",
                            "bounds": f"([{x_min:.2f}, {x_max:.2f}], [{y_min:.2f}, {y_max:.2f}])"
                        },
                        {
                            "type": "writers.las",
                            "filename": output_path,
                            "compression": "laszip"  # 可选：压缩输出
                        }
                    ]
                }

                # 执行管道
                try:
                    pipeline = pdal.Pipeline(json.dumps(pipeline_json))
                    count = pipeline.execute()

                    if count > 0:
                        tile_count += 1
                        print(f"    生成瓦片: {output_filename} ({count} 个点)")
                    else:
                        # 如果没有点，删除空文件
                        if os.path.exists(output_path):
                            os.remove(output_path)

                except Exception as e:
                    print(f"    警告: 处理瓦片 {output_filename} 时出错: {str(e)}")

        print(f"  完成处理 {las_file_path.name}，生成 {tile_count} 个瓦片")

    except Exception as e:
        print(f"处理文件 {las_file_path.name} 时出错: {str(e)}")


def batch_tile_las(input_folder, output_folder, tile_size=100):
    """
    批量处理LAS文件的主函数
    """
    print(f"开始批量处理LAS文件")
    print(f"输入文件夹: {input_folder}")
    print(f"输出文件夹: {output_folder}")
    print(f"切割大小: {tile_size}米")
    print("-" * 50)

    tile_las_files(input_folder, output_folder, tile_size)

    print("-" * 50)
    print("批量处理完成!")


# 使用示例
if __name__ == "__main__":
    # 设置输入和输出文件夹路径
    input_folder = "./input_las"  # 替换为你的输入文件夹路径
    output_folder = "./tiled_las"  # 替换为你的输出文件夹路径

    # 执行批量处理
    batch_tile_las(input_folder, output_folder, tile_size=100)