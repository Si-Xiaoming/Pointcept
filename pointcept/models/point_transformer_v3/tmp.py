"""
Point Transformer - V3 Mode2

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from addict import Dict
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
import spconv.pytorch as spconv
import torch_scatter
from timm.layers import DropPath

try:
    import flash_attn
except ImportError:
    flash_attn = None

from pointcept.models.builder import MODELS
from pointcept.models.utils.misc import offset2bincount
from pointcept.models.utils.structure import Point
from pointcept.models.modules import PointModule, PointSequential

# --- LoRA 相关代码 ---
import math

class LoRALinear(nn.Module):
    """LoRA 适配层，用于替换 PTv3 中的线性层"""
    def __init__(self, linear_layer, rank=4, alpha=16, dropout=0.0):
        super().__init__()
        self.linear = linear_layer
        d, k = linear_layer.weight.shape

        # 冻结原始权重（关键步骤）
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        # LoRA 参数
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank  # 重要缩放因子

        # 低秩分解矩阵
        self.lora_A = nn.Parameter(torch.zeros((k, rank)))
        self.lora_B = nn.Parameter(torch.zeros((rank, d)))

        # 初始化 LoRA 参数
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # 原始线性变换
        original_output = self.linear(x)

        # LoRA 变换：x @ (A @ B) * scaling
        lora_output = self.dropout(x) @ self.lora_A @ self.lora_B
        lora_output = lora_output * self.scaling

        return original_output + lora_output

class LoRAQKVLinear(nn.Module):
    """专门用于 QKV 线性层的 LoRA，只对 Q 和 V 进行 LoRA 适配"""
    def __init__(self, qkv_linear_layer, rank=4, alpha=16, dropout=0.0):
        super().__init__()
        self.qkv_linear = qkv_linear_layer
        out_features, in_features = qkv_linear_layer.weight.shape
        assert out_features % 3 == 0, "QKV linear layer output features must be divisible by 3"
        self.dim = out_features // 3

        # 冻结原始权重
        self.qkv_linear.weight.requires_grad = False
        if self.qkv_linear.bias is not None:
            self.qkv_linear.bias.requires_grad = False

        # LoRA 参数 for Q
        self.rank_q = rank
        self.alpha_q = alpha
        self.scaling_q = alpha / rank
        self.lora_A_q = nn.Parameter(torch.zeros((in_features, rank)))
        self.lora_B_q = nn.Parameter(torch.zeros((rank, self.dim)))
        nn.init.kaiming_uniform_(self.lora_A_q, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_q)

        # LoRA 参数 for V
        self.rank_v = rank
        self.alpha_v = alpha
        self.scaling_v = alpha / rank
        self.lora_A_v = nn.Parameter(torch.zeros((in_features, rank)))
        self.lora_B_v = nn.Parameter(torch.zeros((rank, self.dim)))
        nn.init.kaiming_uniform_(self.lora_A_v, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_v)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # 原始 QKV 变换
        qkv = self.qkv_linear(x) # [..., 3*dim]
        q, k, v = qkv.chunk(3, dim=-1) # [..., dim] each

        # LoRA 变换 for Q
        lora_q = self.dropout(x) @ self.lora_A_q @ self.lora_B_q
        lora_q = lora_q * self.scaling_q
        q = q + lora_q

        # LoRA 变换 for V (K 保持不变)
        lora_v = self.dropout(x) @ self.lora_A_v @ self.lora_B_v
        lora_v = lora_v * self.scaling_v
        v = v + lora_v

        # 合并 Q, K, V
        qkv_lora = torch.cat([q, k, v], dim=-1)
        return qkv_lora


# --- 原有模块代码 ---
# (LayerScale, RPE, SerializedAttention, MLP, Block, GridPooling, GridUnpooling, Embedding 保持不变)
# ... [此处省略 LayerScale 到 Embedding 的原有代码，假设它们没有变化] ...

class SerializedAttention(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        super().__init__()
        # ... [原有初始化代码不变] ...
        self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = torch.nn.Linear(channels, channels)
        # ... [其余初始化代码不变] ...

    # ... [forward, get_rel_pos, get_padding_and_inverse 方法不变] ...


class Block(PointModule):
    def __init__(
        self,
        # ... [所有原有参数不变] ...
    ):
        super().__init__()
        # ... [原有初始化代码不变，直到 self.attn] ...
        self.attn = SerializedAttention(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
        )
        # ... [其余初始化代码不变] ...

    # ... [forward 方法不变] ...

# ... [GridPooling, GridUnpooling, Embedding 类保持不变] ...

@MODELS.register_module("PT-v3m3") # 注意：模块名已改为 PT-v3m3
class PointTransformerV3(PointModule):
    def __init__(
        self,
        in_channels=6,
        order=("z", "z-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(48, 48, 48, 48, 48),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(48, 48, 48, 48),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        layer_scale=None,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        traceable=False,
        mask_token=False,
        enc_mode=False,
        freeze_encoder=False,

        # LoRA 配置参数
        lora_rank=None,          # LoRA 的秩，None 表示不使用 LoRA
        lora_alpha=16,           # LoRA 缩放因子
        lora_dropout=0.0,        # LoRA dropout
        # 注意：这里移除了 lora_target_modules，因为我们硬编码了目标
    ):
        super().__init__()
        # ... [原有初始化代码不变] ...

        # 保存 LoRA 配置
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        # self.lora_target_modules = lora_target_modules # 不再需要，硬编码处理

        # ... [原有模型结构初始化代码不变] ...

        # 应用权重初始化
        self.apply(self._init_weights)

        # 在模型初始化完成后应用 LoRA
        if self.lora_rank is not None:
            self._apply_lora()


    @staticmethod
    def _init_weights(module):
        # ... [原有 _init_weights 代码不变] ...

    def _apply_lora_to_module(self, module, target_name):
        """递归地将指定名称的线性层替换为 LoRALinear"""
        if isinstance(module, nn.Linear) and target_name in ['proj', 'fc1', 'fc2']:
             # 对于 proj, fc1, fc2 等普通线性层，直接替换
            lora_layer = LoRALinear(
                module,
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                dropout=self.lora_dropout
            )
            return lora_layer
        elif isinstance(module, nn.Linear) and target_name == 'qkv':
            # 对于 qkv 线性层，使用专门的 LoRAQKVLinear (只对 Q,V 应用)
            lora_layer = LoRAQKVLinear(
                module,
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                dropout=self.lora_dropout
            )
            return lora_layer
        elif isinstance(module, PointSequential):
            # 递归处理 PointSequential 容器
            for name, child in module.named_children():
                # 注意：PointSequential 的子模块可能有特定的命名，如 '0' 对应 MLP
                # 我们需要根据其在 Block 中的角色来判断
                # 这里简化处理，直接传递子模块名称
                replaced_child = self._apply_lora_to_module(child, name)
                if replaced_child is not child:
                    # 替换子模块
                    module._modules[name] = replaced_child
        elif hasattr(module, '_modules'):
             # 递归处理一般的 nn.Module 容器
            for name, child in module.named_children():
                 replaced_child = self._apply_lora_to_module(child, name)
                 if replaced_child is not child:
                     setattr(module, name, replaced_child)
        return module # 返回可能被替换的模块

    def _apply_lora(self):
        """应用 LoRA 到模型的指定层"""
        print(f"Applying LoRA with rank={self.lora_rank}, alpha={self.lora_alpha}")

        # 1. 处理 Embedding 层 (如果需要)
        # self.embedding = self._apply_lora_to_module(self.embedding, 'embedding') # 通常不对 embedding 应用

        # 2. 处理 Encoder 部分
        if hasattr(self, 'enc'):
            for s in range(self.num_stages):
                enc_name = f"enc{s}"
                if enc_name in self.enc:
                    enc = self.enc[enc_name]
                    # 处理每个 block
                    for i in range(len([n for n in enc if n.startswith("block")])):
                        block_name = f"block{i}"
                        if block_name in enc:
                            block = enc[block_name]

                            # 处理注意力层的 qkv
                            if hasattr(block.attn, 'qkv'):
                                 # 直接替换 qkv 线性层
                                block.attn.qkv = LoRAQKVLinear(
                                    block.attn.qkv,
                                    rank=self.lora_rank,
                                    alpha=self.lora_alpha,
                                    dropout=self.lora_dropout
                                )

                            # 处理注意力层的 proj
                            if hasattr(block.attn, 'proj'):
                                block.attn.proj = LoRALinear(
                                    block.attn.proj,
                                    rank=self.lora_rank,
                                    alpha=self.lora_alpha,
                                    dropout=self.lora_dropout
                                )

                            # 处理 MLP 层 (在 PointSequential 'mlp' -> '0' -> MLP -> fc1/fc2)
                            if hasattr(block.mlp, '0'): # 获取 MLP 模块
                                mlp = block.mlp[0]
                                if hasattr(mlp, 'fc1'):
                                    mlp.fc1 = LoRALinear(
                                        mlp.fc1,
                                        rank=self.lora_rank,
                                        alpha=self.lora_alpha,
                                        dropout=self.lora_dropout
                                    )
                                if hasattr(mlp, 'fc2'):
                                    mlp.fc2 = LoRALinear(
                                        mlp.fc2,
                                        rank=self.lora_rank,
                                        alpha=self.lora_alpha,
                                        dropout=self.lora_dropout
                                    )

        # 3. 处理 Decoder 部分（如果存在且需要应用 LoRA）
        if not self.enc_mode and hasattr(self, 'dec'):
            for s in reversed(range(self.num_stages - 1)):
                dec_name = f"dec{s}"
                if dec_name in self.dec:
                    dec = self.dec[dec_name]

                    # 处理上采样层的 proj 和 proj_skip (如果需要)
                    # if "up" in dec:
                    #     up = dec.up
                    #     if hasattr(up, 'proj') and hasattr(up.proj, 'linear'): # 检查结构
                    #         up.proj.linear = self._apply_lora_to_module(up.proj.linear, 'proj')
                    #     if hasattr(up, 'proj_skip') and hasattr(up.proj_skip, 'linear'):
                    #         up.proj_skip.linear = self._apply_lora_to_module(up.proj_skip.linear, 'proj')

                    # 处理每个 block
                    for i in range(len([n for n in dec if n.startswith("block")])):
                        block_name = f"block{i}"
                        if block_name in dec:
                            block = dec[block_name]

                            # 处理注意力层的 qkv
                            if hasattr(block.attn, 'qkv'):
                                block.attn.qkv = LoRAQKVLinear(
                                    block.attn.qkv,
                                    rank=self.lora_rank,
                                    alpha=self.lora_alpha,
                                    dropout=self.lora_dropout
                                )

                            # 处理注意力层的 proj
                            if hasattr(block.attn, 'proj'):
                                block.attn.proj = LoRALinear(
                                    block.attn.proj,
                                    rank=self.lora_rank,
                                    alpha=self.lora_alpha,
                                    dropout=self.lora_dropout
                                )

                            # 处理 MLP 层
                            if hasattr(block.mlp, '0'): # 获取 MLP 模块
                                mlp = block.mlp[0]
                                if hasattr(mlp, 'fc1'):
                                    mlp.fc1 = LoRALinear(
                                        mlp.fc1,
                                        rank=self.lora_rank,
                                        alpha=self.lora_alpha,
                                        dropout=self.lora_dropout
                                    )
                                if hasattr(mlp, 'fc2'):
                                    mlp.fc2 = LoRALinear(
                                        mlp.fc2,
                                        rank=self.lora_rank,
                                        alpha=self.lora_alpha,
                                        dropout=self.lora_dropout
                                    )

        print("LoRA applied successfully.")

    def forward(self, data_dict):
        # ... [原有 forward 代码不变] ...
        point = Point(data_dict)
        point = self.embedding(point)

        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        point = self.enc(point)
        if not self.enc_mode:
            point = self.dec(point)
        return point
