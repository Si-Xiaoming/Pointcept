import argparse
import os
import numpy as np
import torch
from safetensors.torch import save_file

# --- 1. 导入必要的模块 ---
# 假设这些模块存在于您的项目结构中
# 如果不存在，请调整导入路径或确保环境已正确设置
try:
    from pointcept.models import build_model
    from pointcept.models.utils.structure import Point
    from pointcept.datasets import build_dataset # 可能用于参考transform
    # 假设 default transform 在 datasets 或类似模块中定义
    # 如果没有，可以参考 transform.txt 手动构建
    from pointcept.datasets import transform # 确保能找到 default 或类似函数
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure your Python environment includes the 'pointcept' package or adjust the import paths accordingly.")
    exit(1)

# --- 2. 模型配置 ---
# 根据 point_transformer_v3m2_sonata.txt 和 config.txt 的信息配置模型
# 注意：这里使用的是 PT-v3m2，参数可能与 config.txt 中的 PT-v3m1 示例不同
MODEL_CFG = dict(
    type="DefaultSegmentorV2",
    num_classes=4,
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m2",
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(48, 96, 192, 384, 512),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 96, 192, 384),
        dec_num_head=(4, 6, 12, 24),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
        traceable=False,
        mask_token=False,
        enc_mode=False,
        freeze_encoder=True,
    ),
)

# --- 3. 权重路径 ---
WEIGHT_PATH = "/datasets/exp-0801/model/model_best-centerloss.pth" # 请替换为实际的 .pth 权重文件路径

# --- 4. ONNX 保存路径 ---
ONNX_SAVE_PATH = "./pt_v3m2_exported.onnx"
SAFETENSORS_SAVE_PATH = "./pt_v3m2_weights.safetensors"

# --- 5. 加载和预处理示例数据 ---
def load_and_preprocess_sample_data():
    """加载并预处理用于 ONNX 导出的示例数据"""
    # --- 模拟或加载点云数据 ---
    # 这里使用随机数据作为示例。在实际应用中，应加载真实数据。
    num_points = 10000 # 示例点数
    coord = np.random.rand(num_points, 3).astype(np.float32) * 10 # 示例坐标 (0-10)
    color = np.random.rand(num_points, 3).astype(np.float32) * 255 # 示例颜色 (0-255)
    # segment = np.random.randint(0, MODEL_CFG['num_classes'], size=(num_points,)) # 示例标签 (如果需要)

    # 创建数据字典
    points_data_dict = {
        "coord": coord,
        "color": color,
        # "segment": segment # 标签在推理时不需要
    }

    # --- 应用 Transform ---
    # 使用 pointcept 风格的 transform (参考 transform.txt)
    # 这里构建一个基础的 transform pipeline，模拟测试时的处理
    # 注意：必须包含模型 forward 所需的 key
    transform_pipeline = [
        # dict(type="CenterShift", apply_z=True), # 可选：中心化
        dict(type="GridSample", grid_size=1.0, hash_type="fnv", mode="train", return_grid_coord=True, return_inverse=False),
        # 注意：模型可能需要 'inverse'，但 ONNX 导出时如果处理得当可以不需要。
        # 如果遇到问题，可以尝试在 transform 中添加 return_inverse=True 并处理。
        dict(type="NormalizeColor"), # 归一化颜色
        dict(type="ToTensor"), # 转换为 Tensor
        # Collect 确保包含模型需要的 key
        dict(type="Collect", keys=("coord", "grid_coord", "color"), feat_keys=("coord", "color")),
    ]

    # 构建 transform (假设 pointcept.datasets.transform 可用)
    composed_transform = transform.Compose(transform_pipeline)
    points_data_dict = composed_transform(points_data_dict)

    # --- 确保包含所有必需的键 ---
    # 检查并添加 offset (对于单个点云 batch size=1)
    if 'offset' not in points_data_dict:
         # offset 指向每个 batch (这里是1个) 的结束索引
        points_data_dict['offset'] = torch.tensor([points_data_dict['coord'].shape[0]], dtype=torch.long)

    print("Sample data keys and shapes after preprocessing:")
    for k, v in points_data_dict.items():
        print(f"  {k}: {v.shape if hasattr(v, 'shape') else type(v)}")

    return points_data_dict

# --- 6. 加载模型 ---
def load_model(cfg, weight_path, device):
    """加载模型并载入权重"""
    print("Building model...")
    model = build_model(cfg)
    model = model.to(device)

    if weight_path and os.path.isfile(weight_path):
        print(f"Loading weights from {weight_path}...")
        checkpoint = torch.load(weight_path, map_location=device)
        # 处理可能的 'state_dict' 键
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint # 直接是 state_dict

        # 加载权重
        try:
            model.load_state_dict(state_dict, strict=False) # strict=False 允许部分匹配
            print("Weights loaded successfully.")
        except Exception as e:
             # 如果 strict=True 失败，尝试移除 'module.' 前缀 (如果存在，常见于 DDP)
             print(f"Strict load failed: {e}. Trying to remove 'module.' prefix...")
             new_state_dict = {}
             for k, v in state_dict.items():
                 name = k[7:] if k.startswith('module.') else k # remove `module.`
                 new_state_dict[name] = v
             try:
                 model.load_state_dict(new_state_dict, strict=False)
                 print("Weights loaded successfully after removing 'module.' prefix.")
             except Exception as e2:
                 print(f"Failed to load weights even after prefix removal: {e2}")
                 raise e2 from e

    elif weight_path:
        raise FileNotFoundError(f"=> No checkpoint found at '{weight_path}'")
    else:
        print("No weight path provided, using randomly initialized model.")

        # ------- 新增代码：禁用 FlashAttention 以支持 ONNX 导出 -------
    def disable_flash_attention(module):
        if hasattr(module, 'enable_flash'):
            print(f"  Disabling flash attention for module: {type(module).__name__}")
            module.enable_flash = False
        # 如果 FlashAttention 是在更深层的子模块中，递归处理
        for child in module.children():
            disable_flash_attention(child)

    print("  Disabling FlashAttention in the model for ONNX export compatibility...")
    disable_flash_attention(model)
    # -------------------------------------------------------------

    model.eval() # 设置为评估模式
    return model



# --- 6. ONNX 导出 ---
def export_onnx(model, input_data_dict, save_path, device):  # <-- 参数名改为 input_data_dict
    """执行 ONNX 导出"""
    print(f"Starting ONNX export to {save_path}...")

    print(f"  Model type before export call: {type(model)}")
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"Expected model to be a torch.nn.Module, but got {type(model)}")

    print(f"  Input data type: {type(input_data_dict)}")  # <-- 应该是 <class 'dict'>
    if not isinstance(input_data_dict, dict):
        raise TypeError(f"Expected input_data_dict to be a dict, but got {type(input_data_dict)}")

    # --- 确保输入字典中的张量在正确的设备上 ---
    # 注意：torch.onnx.export 通常会处理设备移动，但显式移动更安全
    input_data_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in input_data_dict.items()}

    try:
        # --- 关键修改：直接传递 input_data_dict 字典 ---
        torch.onnx.export(
            model,  # 模型实例 (nn.Module)
            # (input_data_dict,),       # <-- 错误：这是传递包含字典的元组
            input_data_dict,  # <-- 正确：直接传递字典本身
            save_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input_data'],  # <-- 输入名称可以改为更通用的 'input_data' 或列出字典的所有键
            output_names=['output_logits'],  # DefaultSegmentorV2 输出 logits
            # dynamic_axes 可以根据字典中的键来设置
            # dynamic_axes={
            #     'coord': {0: 'num_points'},
            #     'grid_coord': {0: 'num_points'},
            #     'color': {0: 'num_points'},
            #     'offset': {0: 'batch_plus_one'}, # 如果适用
            #     'output_logits': {0: 'num_points'}
            # }
        )
        print(f"✅ ONNX model exported to {save_path}")

        # --- (可选) 保存 safetensors 权重 ---
        try:
            state_dict = model.state_dict()
            cpu_state_dict = {k: v.cpu() for k, v in state_dict.items()}
            save_file(cpu_state_dict, SAFETENSORS_SAVE_PATH)
            print(f"✅ Safetensors weights saved to {SAFETENSORS_SAVE_PATH}")
        except Exception as e:
            print(f"❌ Error saving Safetensors: {e}")

    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        import traceback
        traceback.print_exc()

# --- 8. 主函数 ---
def main():
    """主处理函数"""
    # - 确定设备 -
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 注意：ONNX Runtime 支持 CPU 和 CUDA 执行。导出时使用哪个设备通常不影响通用性，
    # 但最好在目标设备上进行推理测试。
    print(f"Using device: {device}")

    # - 加载和预处理数据 -
    points_data_dict = load_and_preprocess_sample_data()

    # - 将数据字典转换为 Point 对象 -
    # 模型 DefaultSegmentorV2 的 forward 方法期望一个 Point 对象
    print("Wrapping input data into a Point object...")
    dummy_input_point = points_data_dict
    print(f"  dummy_input_point type: {type(dummy_input_point)}")
    print(f"  dummy_input_point keys: {list(dummy_input_point.keys())}")

    # - 加载模型 -
    model = load_model(MODEL_CFG, WEIGHT_PATH, device)
    print(f"Type of model after loading: {type(model)}")  # 关键调试信息
    print("Model loaded, moved to device, and set to eval mode.")
    if not isinstance(model, torch.nn.Module):
        raise TypeError("load_model did not return a valid torch.nn.Module instance!")

    # - ONNX 转换 -
    export_onnx(model, dummy_input_point, ONNX_SAVE_PATH, device)

if __name__ == "__main__":
    main()