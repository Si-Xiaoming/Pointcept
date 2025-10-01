_base_ = ["../_base_/default_runtime.py"]

# misc custom settings
batch_size = 1
num_workers = 0 # shm
mix_prob = 0.8 #
empty_cache = False
enable_amp = True

num_points_per_block = 65536

weight = "/datasets/exp/default-1.0/model/model_best.pth"  # path to model weight
save_path = "/datasets/exp/outputs"
  # model_best_supervised.pth     model_last-ep3.pth
grid_size=0.1
overlap_ratio = 0.1  # 块之间的重叠比例


# dataset settings
dataset_type = "LAZDatasetVote"
laz_file="/datasets/navarra-small/raw/test/0401.laz"

# model settings
model = dict(
    type='DefaultSegmentorV2',
    num_classes=4,
    backbone_out_channels=64,
    backbone=dict(
        type='PT-v3m2',
        in_channels=6,
        order=('z', 'z-trans', 'hilbert', 'hilbert-trans'),
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
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        traceable=False,
        mask_token=False,
        enc_mode=False,
        freeze_encoder=True),
    criteria=[
        dict(type='CrossEntropyLoss', loss_weight=1.0, ignore_index=-1),
        dict(
            type='LovaszLoss',
            mode='multiclass',
            loss_weight=1.0,
            ignore_index=-1)
    ],
    freeze_backbone=False)


# scheduler settings

optimizer = dict(type="AdamW", lr=0.006, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.006, 0.0006],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0006)]


data = dict(
    num_classes = 4,
    ignore_index = -1,
    names = [
        "ground",
        "vegetation",
        "building",
        "others",
    ],
    train=dict(
        type=dataset_type,
        split="train",

    ),
    test=dict(
        type=dataset_type,
        split="test",
        laz_file=laz_file,
        has_ground_truth=True,
        transform=[
            dict(type="CenterShift", apply_z=True),
            # dict(type="NormalizeColor"),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "color", "inverse", "segment"),
                feat_keys=("coord", "color"),
            )
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                # keys=("coord", "color"),
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("coord", "color"),
                ),
            ],
            aug_transform=[
                [dict(type="RandomScale", scale=[0.9, 0.9])],
                [dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomScale", scale=[1, 1])],
                [dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomScale", scale=[1.1, 1.1])],
            ],
        num_points_per_block=num_points_per_block,
        overlap_ratio=overlap_ratio,
        grid_size=grid_size,

    ),
    ),
)

# hooks
hooks = [
    dict(
        type="CheckpointLoader",
        keywords="module.student.backbone",
        replacement="module.backbone",
    ),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

# tester settings
test = dict(
    type="LAZSemiSegTesterSimple")