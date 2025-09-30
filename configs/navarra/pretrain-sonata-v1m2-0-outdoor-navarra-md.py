"""
Default configuration for pretraining a SONATA model
Dataset: ScanNet v2, ScanNet++, S3DIS, HM3D, ArkitScene, Structured3D
"""

_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 1  # bs: total bs in all gpus
num_worker = 1
mix_prob = 0  # 混合增强的应用概率，控制在训练过程中是否应用点云混合增强技术
clip_grad = 3.0 # 梯度裁剪的阈值参数，用于防止训练过程中的梯度爆炸问题
empty_cache = False
enable_amp = True # 控制是否在训练过程中使用混合精度计算
amp_dtype = "bfloat16"  #enable_amp联动  bfloat16比float32节省50%内存
evaluate = False
find_unused_parameters = False
num_points_per_step = 300  # 65536
grid_size = 0.1 # 0.02
dataset_type = "NavarraDataset"
data_root = "/datasets/navarra-small/"
save_path =  "/datasets/exp/pretrain_outdoor_01_ep2000_sonata_v1m2_md"
# exp-0801\server_data\exp\default\model
#weight = "/datasets/exp-0801/server_data/exp/default/model/epoch_5.pth"
#resume = True
# model settings
model = dict(
    type="Sonata-v1m2-MD-Generic",
    # num_density_views=3,        # 生成3个不同密度视图
    density_min_ratio=0.2,      # 最小为原始密度的20%
    density_max_ratio=3.0,      # 最大为原始密度的400%
    # density_anisotropic_prob=0.4,  # 40%概率使用各向异性采样
    #cross_density_weight_start=0.2,
    # cross_density_weight=1.2,   # 密度一致性损失权重
    # backbone - student & teacher
    backbone=dict(
        type="PT-v3m2",
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),  # 编码器阶段的下采样步长
        enc_depths=(3, 3, 3, 12, 3), # 每个编码器阶段的Transformer块深度  中间阶段(第4阶段)深度最大，以捕获最复杂的特征交互
        enc_channels=(48, 96, 192, 384, 512), # 每个编码器阶段的特征通道数
        enc_num_head=(3, 6, 12, 24, 32),  # 编码器阶段的注意力头数量
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        mlp_ratio=4, # MLP隐藏层与输入维度的比例
        qkv_bias=True, # pt v3 参数
        qk_scale=None, # 自动计算为1/sqrt(head_dim)
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        traceable=True,
        enc_mode=True,
        mask_token=True,
    ),
    teacher_custom=dict(
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
    ),
    head_in_channels=1088,
    head_hidden_channels=4096,
    head_embed_channels=256,
    head_num_prototypes=4096,
    num_global_view=2,
    num_local_view=4,
    mask_size_start=0.4,  # 0.1
    mask_size_base=1.2,   # 0.4
    mask_size_warmup_ratio=0.05,
    mask_ratio_start=0.3,
    mask_ratio_base=0.7,
    mask_ratio_warmup_ratio=0.05,
    mask_jitter=0.01,
    teacher_temp_start=0.04,
    teacher_temp_base=0.07,
    teacher_temp_warmup_ratio=0.05,
    student_temp=0.1,
    mask_loss_weight=2 / 8,
    roll_mask_loss_weight=2 / 8,
    unmask_loss_weight=4 / 8,
    momentum_base=0.994,
    momentum_final=1,
    match_max_k=8,
    match_max_r=0.32,
    up_cast_level=2,
)

# scheduler settings
epoch = 2000
eval_epoch = 200
base_lr = 0.004
lr_decay = 0.9  # layer-wise lr decay

base_wd = 0.04  # wd scheduler enable in hooks
final_wd = 0.2  # wd scheduler enable in hooks

dec_depths = model["backbone"]["enc_depths"]
param_dicts = [
    dict(
        keyword=f"enc{e}.block{b}.",
        lr=base_lr * lr_decay ** (sum(dec_depths) - sum(dec_depths[:e]) - b - 1),
    )
    for e in range(len(dec_depths))
    for b in range(dec_depths[e])
]
del dec_depths

optimizer = dict(type="AdamW", lr=base_lr, weight_decay=base_wd)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[base_lr] + [g["lr"] for g in param_dicts],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

# dataset settings
transform = [
    dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="train"),
    dict(type="Copy", keys_dict={"coord": "origin_coord"}),
    dict(
        # type="MultiViewGenerator",
        # global_view_scale=(0.4, 1.0),
        # local_view_scale=(0.1, 0.4),
        type="DensityPerturbationViewGenerator",
        global_view_size=(15, 45),  # (40, 50)
        local_view_size=(3, 15),  # (20, 30)

        view_keys=("coord", "origin_coord", "color"),
        global_view_num=2,
        local_view_num=4,
        global_shared_transform=[
            dict(
                type="RandomColorJitter",
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.02,
                p=0.8,
            ),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="HeightNormalization",
                         base_level="ground",
                         max_height=50.0,
                         ground_percentile=0.03),
            # dict(type="ChromaticJitter", p=0.95, std=0.05),
            # dict(type="NormalizeColor"),
        ],
        global_transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomScale", scale=[0.8, 1.2]),
            # dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8), [-0.1, 0.1]
            dict(type="RandomRotate", angle=[-0.1, 0.1], axis="z", center=[0, 0, 0], p=0.8),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.8),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.8),
            dict(type="RandomFlip", p=0.5),
            # dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="RandomJitter", sigma=0.05, clip=0.5),
            # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
        ],
        local_transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomScale", scale=[0.8, 1.2]),
            # dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8), [-0.1, 0.1]
            dict(type="RandomRotate", angle=[-0.1, 0.1], axis="z", center=[0, 0, 0], p=0.8),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.8),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.8),
            dict(type="RandomFlip", p=0.5),
            # dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="RandomJitter", sigma=0.05, clip=0.5),

            # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            # dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(
                type="RandomColorJitter",
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.02,
                p=0.8,
            ),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            # dict(type="ChromaticJitter", p=0.95, std=0.05),
            # dict(type="NormalizeColor"),
        ],
        max_size=num_points_per_step,
    ),
    dict(type="ToTensor"),
    dict(type="Update", keys_dict={"grid_size": grid_size}),
    dict(
        type="Collect",
        keys=(
            "global_origin_coord",
            "global_coord",
            "global_color",
            "global_offset",
            "local_origin_coord",
            "local_coord",
            "local_color",
            "local_offset",
            "grid_size",
            "name",
        ),
        offset_keys_dict=dict(),
        global_feat_keys=("global_coord", "global_color"),
        local_feat_keys=("local_coord", "local_color"),
    ),
]
# dataset settings

data = dict(
    train=dict(
        type = dataset_type,
        split = ("train", "val"),
        data_root = data_root,
        transform = transform,
        test_mode = False,
    )
)

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="WeightDecaySchedular", base_value=base_wd, final_value=final_wd),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaver", save_freq=1),
]
