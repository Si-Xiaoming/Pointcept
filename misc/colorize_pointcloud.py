'''
文件夹：8-6_laz下有8-6_trans_trans.las和EOSDOMImage.tif文件，使用tif文件为las文件赋色
要求：若点无对应RGB值（无效点），则直接删除该点
      同时删除所有RGB值为(0,0,0)的点
'''

import laspy
import numpy as np
from osgeo import gdal

# 配置文件路径
LAS_PATH = "D:/04-Datasets/8-6_laz/8-6_trans_trans.las"
TIFF_PATH = "D:/04-Datasets/8-6_laz/EOSDOMImage.tif"
OUTPUT_LAS_PATH = "D:/04-Datasets/8-6_laz/8-6_trans_trans_colored.las"


def main():
    # 1. 读取LAS文件
    print("Reading LAS file...")
    las = laspy.read(LAS_PATH)
    points = np.vstack([las.x, las.y, las.z]).transpose()

    # 2. 读取GeoTIFF并获取地理参考
    print("Reading GeoTIFF and getting georeference...")
    ds = gdal.Open(TIFF_PATH)
    if ds is None:
        raise RuntimeError(f"无法打开GeoTIFF文件: {TIFF_PATH}")

    # 获取地理变换参数 [top_left_x, x_size, 0, top_left_y, 0, y_size]
    geotransform = ds.GetGeoTransform()
    if not geotransform:
        raise RuntimeError("GeoTIFF缺少地理参考信息")

    # 获取图像数据（假设为RGB三波段）
    band1 = ds.GetRasterBand(1)
    band2 = ds.GetRasterBand(2)
    band3 = ds.GetRasterBand(3)

    # 读取整个图像到内存（适用于中等大小图像）
    img_r = band1.ReadAsArray()
    img_g = band2.ReadAsArray()
    img_b = band3.ReadAsArray()

    # 3. 坐标转换：点云坐标 -> 图像行列号
    print("Mapping point cloud coordinates to image pixels...")
    cols = ((points[:, 0] - geotransform[0]) / geotransform[1]).astype(int)
    rows = ((points[:, 1] - geotransform[3]) / geotransform[5]).astype(int)

    # 4. 有效性检查
    width = ds.RasterXSize
    height = ds.RasterYSize
    valid_mask = (
            (cols >= 0) & (cols < width) &
            (rows >= 0) & (rows < height)
    )

    valid_points_count = np.sum(valid_mask)
    print(f"Valid points (georeferenced): {valid_points_count}/{len(points)}")

    # 5. 直接删除无效点，只处理有效点
    if valid_points_count == 0:
        raise RuntimeError("没有有效的点可以赋色！")

    # 6. 提取有效点的RGB值
    print("Extracting RGB values for valid points...")
    # 8位TIFF -> 16位LAS颜色 (255*257=65535)
    scale_factor = 257 if band1.DataType == gdal.GDT_Byte else 1

    # 为有效点赋值 (直接使用valid_mask筛选)
    rgb_r = img_r[rows[valid_mask], cols[valid_mask]] * scale_factor
    rgb_g = img_g[rows[valid_mask], cols[valid_mask]] * scale_factor
    rgb_b = img_b[rows[valid_mask], cols[valid_mask]] * scale_factor

    # 7. 新增：过滤RGB全为0的点
    color_valid_mask = (rgb_r != 0) | (rgb_g != 0) | (rgb_b != 0)
    non_zero_color_count = np.sum(color_valid_mask)
    print(f"Non-zero color points: {non_zero_color_count}/{valid_points_count}")

    if non_zero_color_count == 0:
        raise RuntimeError("所有有效点的RGB均为0！")

    # 8. 创建最终有效点掩码
    final_valid_mask = np.zeros(len(points), dtype=bool)
    final_valid_mask[valid_mask] = color_valid_mask

    # 9. 创建新的LAS对象（只包含最终有效点）
    print(f"Creating new LAS with {non_zero_color_count} colored points...")
    new_las = laspy.LasData(las.header)

    # 复制所有最终有效点的原始属性
    new_las.points = las.points[final_valid_mask]

    # 设置新点云的颜色
    new_las.red = (rgb_r[color_valid_mask]).astype(np.uint16)
    new_las.green = (rgb_g[color_valid_mask]).astype(np.uint16)
    new_las.blue = (rgb_b[color_valid_mask]).astype(np.uint16)

    # 10. 保存新LAS文件
    print(f"Saving to {OUTPUT_LAS_PATH}...")
    new_las.write(OUTPUT_LAS_PATH)
    print("Done!")


if __name__ == "__main__":
    main()