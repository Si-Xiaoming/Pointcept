DefaultSegmentorV2(
  (seg_head): Linear(in_features=64, out_features=4, bias=True)
  (backbone): PointTransformerV3(
    (embedding): Embedding(
      (stem): PointSequential(
        (linear): Linear(in_features=6, out_features=48, bias=True)
        (norm): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
        (act): GELU(approximate='none')
      )
    )
    (enc): PointSequential(
      (enc0): PointSequential(
        (block0): Block(
          (cpe): PointSequential(
            (0): SubMConv3d(48, 48, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm)
            (1): Linear(in_features=48, out_features=48, bias=True)
            (2): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
          )
          (norm1): PointSequential(
            (0): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
          )
          (ls1): PointSequential(
            (0): Identity()
          )
          (attn): SerializedAttention(
            (qkv): Linear(in_features=48, out_features=144, bias=True)
            (proj): Linear(in_features=48, out_features=48, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (norm2): PointSequential(
            (0): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
          )
          (ls2): PointSequential(
            (0): Identity()
          )
          (mlp): PointSequential(
            (0): MLP(
              (fc1): Linear(in_features=48, out_features=192, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=192, out_features=48, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (drop_path): PointSequential(
            (0): Identity()
          )
        )
        (block1): Block(
          (cpe): PointSequential(
            (0): SubMConv3d(48, 48, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm)
            (1): Linear(in_features=48, out_features=48, bias=True)
            (2): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
          )
          (norm1): PointSequential(
            (0): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
          )
          (ls1): PointSequential(
            (0): Identity()
          )
          (attn): SerializedAttention(
            (qkv): Linear(in_features=48, out_features=144, bias=True)
            (proj): Linear(in_features=48, out_features=48, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (norm2): PointSequential(
            (0): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
          )
          (ls2): PointSequential(
            (0): Identity()
          )
          (mlp): PointSequential(
            (0): MLP(
              (fc1): Linear(in_features=48, out_features=192, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=192, out_features=48, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (drop_path): PointSequential(
            (0): DropPath(drop_prob=0.013)
          )
        )
        (block2): Block(
          (cpe): PointSequential(
            (0): SubMConv3d(48, 48, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm)
            (1): Linear(in_features=48, out_features=48, bias=True)
            (2): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
          )
          (norm1): PointSequential(
            (0): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
          )
          (ls1): PointSequential(
            (0): Identity()
          )
          (attn): SerializedAttention(
            (qkv): Linear(in_features=48, out_features=144, bias=True)
            (proj): Linear(in_features=48, out_features=48, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (norm2): PointSequential(
            (0): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
          )
          (ls2): PointSequential(
            (0): Identity()
          )
          (mlp): PointSequential(
            (0): MLP(
              (fc1): Linear(in_features=48, out_features=192, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=192, out_features=48, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (drop_path): PointSequential(
            (0): DropPath(drop_prob=0.026)
          )
        )
      )
      (enc1): PointSequential(
        (down): GridPooling(
          (proj): Linear(in_features=48, out_features=96, bias=True)
          (norm): PointSequential(
            (0): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
          )
          (act): PointSequential(
            (0): GELU(approximate='none')
          )
        )
        (block0): Block(
          (cpe): PointSequential(
            (0): SubMConv3d(96, 96, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm)
            (1): Linear(in_features=96, out_features=96, bias=True)
            (2): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
          )
          (norm1): PointSequential(
            (0): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
          )
          (ls1): PointSequential(
            (0): Identity()
          )
          (attn): SerializedAttention(
            (qkv): Linear(in_features=96, out_features=288, bias=True)
            (proj): Linear(in_features=96, out_features=96, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (norm2): PointSequential(
            (0): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
          )
          (ls2): PointSequential(
            (0): Identity()
          )
          (mlp): PointSequential(
            (0): MLP(
              (fc1): Linear(in_features=96, out_features=384, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=384, out_features=96, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (drop_path): PointSequential(
            (0): DropPath(drop_prob=0.039)
          )
        )
        (block1): Block(
          (cpe): PointSequential(
            (0): SubMConv3d(96, 96, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm)
            (1): Linear(in_features=96, out_features=96, bias=True)
            (2): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
          )
          (norm1): PointSequential(
            (0): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
          )
          (ls1): PointSequential(
            (0): Identity()
          )
          (attn): SerializedAttention(
            (qkv): Linear(in_features=96, out_features=288, bias=True)
            (proj): Linear(in_features=96, out_features=96, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (norm2): PointSequential(
            (0): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
          )
          (ls2): PointSequential(
            (0): Identity()
          )
          (mlp): PointSequential(
            (0): MLP(
              (fc1): Linear(in_features=96, out_features=384, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=384, out_features=96, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (drop_path): PointSequential(
            (0): DropPath(drop_prob=0.052)
          )
        )
        (block2): Block(
          (cpe): PointSequential(
            (0): SubMConv3d(96, 96, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm)
            (1): Linear(in_features=96, out_features=96, bias=True)
            (2): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
          )
          (norm1): PointSequential(
            (0): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
          )
          (ls1): PointSequential(
            (0): Identity()
          )
          (attn): SerializedAttention(
            (qkv): Linear(in_features=96, out_features=288, bias=True)
            (proj): Linear(in_features=96, out_features=96, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (norm2): PointSequential(
            (0): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
          )
          (ls2): PointSequential(
            (0): Identity()
          )
          (mlp): PointSequential(
            (0): MLP(
              (fc1): Linear(in_features=96, out_features=384, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=384, out_features=96, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (drop_path): PointSequential(
            (0): DropPath(drop_prob=0.065)
          )
        )
      )
      (enc2): PointSequential(
        (down): GridPooling(
          (proj): Linear(in_features=96, out_features=192, bias=True)
          (norm): PointSequential(
            (0): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
          )
          (act): PointSequential(
            (0): GELU(approximate='none')
          )
        )
        (block0): Block(
          (cpe): PointSequential(
            (0): SubMConv3d(192, 192, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm)
            (1): Linear(in_features=192, out_features=192, bias=True)
            (2): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
          )
          (norm1): PointSequential(
            (0): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
          )
          (ls1): PointSequential(
            (0): Identity()
          )
          (attn): SerializedAttention(
            (qkv): Linear(in_features=192, out_features=576, bias=True)
            (proj): Linear(in_features=192, out_features=192, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (norm2): PointSequential(
            (0): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
          )
          (ls2): PointSequential(
            (0): Identity()
          )
          (mlp): PointSequential(
            (0): MLP(
              (fc1): Linear(in_features=192, out_features=768, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=768, out_features=192, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (drop_path): PointSequential(
            (0): DropPath(drop_prob=0.078)
          )
        )
        (block1): Block(
          (cpe): PointSequential(
            (0): SubMConv3d(192, 192, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm)
            (1): Linear(in_features=192, out_features=192, bias=True)
            (2): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
          )
          (norm1): PointSequential(
            (0): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
          )
          (ls1): PointSequential(
            (0): Identity()
          )
          (attn): SerializedAttention(
            (qkv): Linear(in_features=192, out_features=576, bias=True)
            (proj): Linear(in_features=192, out_features=192, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (norm2): PointSequential(
            (0): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
          )
          (ls2): PointSequential(
            (0): Identity()
          )
          (mlp): PointSequential(
            (0): MLP(
              (fc1): Linear(in_features=192, out_features=768, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=768, out_features=192, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (drop_path): PointSequential(
            (0): DropPath(drop_prob=0.091)
          )
        )
        (block2): Block(
          (cpe): PointSequential(
            (0): SubMConv3d(192, 192, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm)
            (1): Linear(in_features=192, out_features=192, bias=True)
            (2): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
          )
          (norm1): PointSequential(
            (0): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
          )
          (ls1): PointSequential(
            (0): Identity()
          )
          (attn): SerializedAttention(
            (qkv): Linear(in_features=192, out_features=576, bias=True)
            (proj): Linear(in_features=192, out_features=192, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (norm2): PointSequential(
            (0): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
          )
          (ls2): PointSequential(
            (0): Identity()
          )
          (mlp): PointSequential(
            (0): MLP(
              (fc1): Linear(in_features=192, out_features=768, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=768, out_features=192, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (drop_path): PointSequential(
            (0): DropPath(drop_prob=0.104)
          )
        )
      )
      (enc3): PointSequential(
        (down): GridPooling(
          (proj): Linear(in_features=192, out_features=384, bias=True)
          (norm): PointSequential(
            (0): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          )
          (act): PointSequential(
            (0): GELU(approximate='none')
          )
        )
        (block0): Block(
          (cpe): PointSequential(
            (0): SubMConv3d(384, 384, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm)
            (1): Linear(in_features=384, out_features=384, bias=True)
            (2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          )
          (norm1): PointSequential(
            (0): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          )
          (ls1): PointSequential(
            (0): Identity()
          )
          (attn): SerializedAttention(
            (qkv): Linear(in_features=384, out_features=1152, bias=True)
            (proj): Linear(in_features=384, out_features=384, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (norm2): PointSequential(
            (0): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          )
          (ls2): PointSequential(
            (0): Identity()
          )
          (mlp): PointSequential(
            (0): MLP(
              (fc1): Linear(in_features=384, out_features=1536, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=1536, out_features=384, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (drop_path): PointSequential(
            (0): DropPath(drop_prob=0.117)
          )
        )
        (block1): Block(
          (cpe): PointSequential(
            (0): SubMConv3d(384, 384, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm)
            (1): Linear(in_features=384, out_features=384, bias=True)
            (2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          )
          (norm1): PointSequential(
            (0): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          )
          (ls1): PointSequential(
            (0): Identity()
          )
          (attn): SerializedAttention(
            (qkv): Linear(in_features=384, out_features=1152, bias=True)
            (proj): Linear(in_features=384, out_features=384, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (norm2): PointSequential(
            (0): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          )
          (ls2): PointSequential(
            (0): Identity()
          )
          (mlp): PointSequential(
            (0): MLP(
              (fc1): Linear(in_features=384, out_features=1536, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=1536, out_features=384, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (drop_path): PointSequential(
            (0): DropPath(drop_prob=0.130)
          )
        )
        (block2): Block(
          (cpe): PointSequential(
            (0): SubMConv3d(384, 384, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm)
            (1): Linear(in_features=384, out_features=384, bias=True)
            (2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          )
          (norm1): PointSequential(
            (0): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          )
          (ls1): PointSequential(
            (0): Identity()
          )
          (attn): SerializedAttention(
            (qkv): Linear(in_features=384, out_features=1152, bias=True)
            (proj): Linear(in_features=384, out_features=384, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (norm2): PointSequential(
            (0): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          )
          (ls2): PointSequential(
            (0): Identity()
          )
          (mlp): PointSequential(
            (0): MLP(
              (fc1): Linear(in_features=384, out_features=1536, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=1536, out_features=384, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (drop_path): PointSequential(
            (0): DropPath(drop_prob=0.143)
          )
        )
        (block3): Block(
          (cpe): PointSequential(
            (0): SubMConv3d(384, 384, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm)
            (1): Linear(in_features=384, out_features=384, bias=True)
            (2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          )
          (norm1): PointSequential(
            (0): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          )
          (ls1): PointSequential(
            (0): Identity()
          )
          (attn): SerializedAttention(
            (qkv): Linear(in_features=384, out_features=1152, bias=True)
            (proj): Linear(in_features=384, out_features=384, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (norm2): PointSequential(
            (0): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          )
          (ls2): PointSequential(
            (0): Identity()
          )
          (mlp): PointSequential(
            (0): MLP(
              (fc1): Linear(in_features=384, out_features=1536, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=1536, out_features=384, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (drop_path): PointSequential(
            (0): DropPath(drop_prob=0.157)
          )
        )
        (block4): Block(
          (cpe): PointSequential(
            (0): SubMConv3d(384, 384, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm)
            (1): Linear(in_features=384, out_features=384, bias=True)
            (2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          )
          (norm1): PointSequential(
            (0): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          )
          (ls1): PointSequential(
            (0): Identity()
          )
          (attn): SerializedAttention(
            (qkv): Linear(in_features=384, out_features=1152, bias=True)
            (proj): Linear(in_features=384, out_features=384, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (norm2): PointSequential(
            (0): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          )
          (ls2): PointSequential(
            (0): Identity()
          )
          (mlp): PointSequential(
            (0): MLP(
              (fc1): Linear(in_features=384, out_features=1536, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=1536, out_features=384, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (drop_path): PointSequential(
            (0): DropPath(drop_prob=0.170)
          )
        )
        (block5): Block(
          (cpe): PointSequential(
            (0): SubMConv3d(384, 384, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm)
            (1): Linear(in_features=384, out_features=384, bias=True)
            (2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          )
          (norm1): PointSequential(
            (0): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          )
          (ls1): PointSequential(
            (0): Identity()
          )
          (attn): SerializedAttention(
            (qkv): Linear(in_features=384, out_features=1152, bias=True)
            (proj): Linear(in_features=384, out_features=384, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (norm2): PointSequential(
            (0): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          )
          (ls2): PointSequential(
            (0): Identity()
          )
          (mlp): PointSequential(
            (0): MLP(
              (fc1): Linear(in_features=384, out_features=1536, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=1536, out_features=384, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (drop_path): PointSequential(
            (0): DropPath(drop_prob=0.183)
          )
        )
        (block6): Block(
          (cpe): PointSequential(
            (0): SubMConv3d(384, 384, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm)
            (1): Linear(in_features=384, out_features=384, bias=True)
            (2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          )
          (norm1): PointSequential(
            (0): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          )
          (ls1): PointSequential(
            (0): Identity()
          )
          (attn): SerializedAttention(
            (qkv): Linear(in_features=384, out_features=1152, bias=True)
            (proj): Linear(in_features=384, out_features=384, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (norm2): PointSequential(
            (0): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          )
          (ls2): PointSequential(
            (0): Identity()
          )
          (mlp): PointSequential(
            (0): MLP(
              (fc1): Linear(in_features=384, out_features=1536, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=1536, out_features=384, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (drop_path): PointSequential(
            (0): DropPath(drop_prob=0.196)
          )
        )
        (block7): Block(
          (cpe): PointSequential(
            (0): SubMConv3d(384, 384, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm)
            (1): Linear(in_features=384, out_features=384, bias=True)
            (2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          )
          (norm1): PointSequential(
            (0): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          )
          (ls1): PointSequential(
            (0): Identity()
          )
          (attn): SerializedAttention(
            (qkv): Linear(in_features=384, out_features=1152, bias=True)
            (proj): Linear(in_features=384, out_features=384, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (norm2): PointSequential(
            (0): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          )
          (ls2): PointSequential(
            (0): Identity()
          )
          (mlp): PointSequential(
            (0): MLP(
              (fc1): Linear(in_features=384, out_features=1536, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=1536, out_features=384, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (drop_path): PointSequential(
            (0): DropPath(drop_prob=0.209)
          )
        )
        (block8): Block(
          (cpe): PointSequential(
            (0): SubMConv3d(384, 384, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm)
            (1): Linear(in_features=384, out_features=384, bias=True)
            (2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          )
          (norm1): PointSequential(
            (0): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          )
          (ls1): PointSequential(
            (0): Identity()
          )
          (attn): SerializedAttention(
            (qkv): Linear(in_features=384, out_features=1152, bias=True)
            (proj): Linear(in_features=384, out_features=384, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (norm2): PointSequential(
            (0): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          )
          (ls2): PointSequential(
            (0): Identity()
          )
          (mlp): PointSequential(
            (0): MLP(
              (fc1): Linear(in_features=384, out_features=1536, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=1536, out_features=384, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (drop_path): PointSequential(
            (0): DropPath(drop_prob=0.222)
          )
        )
        (block9): Block(
          (cpe): PointSequential(
            (0): SubMConv3d(384, 384, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm)
            (1): Linear(in_features=384, out_features=384, bias=True)
            (2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          )
          (norm1): PointSequential(
            (0): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          )
          (ls1): PointSequential(
            (0): Identity()
          )
          (attn): SerializedAttention(
            (qkv): Linear(in_features=384, out_features=1152, bias=True)
            (proj): Linear(in_features=384, out_features=384, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (norm2): PointSequential(
            (0): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          )
          (ls2): PointSequential(
            (0): Identity()
          )
          (mlp): PointSequential(
            (0): MLP(
              (fc1): Linear(in_features=384, out_features=1536, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=1536, out_features=384, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (drop_path): PointSequential(
            (0): DropPath(drop_prob=0.235)
          )
        )
        (block10): Block(
          (cpe): PointSequential(
            (0): SubMConv3d(384, 384, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm)
            (1): Linear(in_features=384, out_features=384, bias=True)
            (2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          )
          (norm1): PointSequential(
            (0): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          )
          (ls1): PointSequential(
            (0): Identity()
          )
          (attn): SerializedAttention(
            (qkv): Linear(in_features=384, out_features=1152, bias=True)
            (proj): Linear(in_features=384, out_features=384, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (norm2): PointSequential(
            (0): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          )
          (ls2): PointSequential(
            (0): Identity()
          )
          (mlp): PointSequential(
            (0): MLP(
              (fc1): Linear(in_features=384, out_features=1536, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=1536, out_features=384, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (drop_path): PointSequential(
            (0): DropPath(drop_prob=0.248)
          )
        )
        (block11): Block(
          (cpe): PointSequential(
            (0): SubMConv3d(384, 384, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm)
            (1): Linear(in_features=384, out_features=384, bias=True)
            (2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          )
          (norm1): PointSequential(
            (0): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          )
          (ls1): PointSequential(
            (0): Identity()
          )
          (attn): SerializedAttention(
            (qkv): Linear(in_features=384, out_features=1152, bias=True)
            (proj): Linear(in_features=384, out_features=384, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (norm2): PointSequential(
            (0): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          )
          (ls2): PointSequential(
            (0): Identity()
          )
          (mlp): PointSequential(
            (0): MLP(
              (fc1): Linear(in_features=384, out_features=1536, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=1536, out_features=384, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (drop_path): PointSequential(
            (0): DropPath(drop_prob=0.261)
          )
        )
      )
      (enc4): PointSequential(
        (down): GridPooling(
          (proj): Linear(in_features=384, out_features=512, bias=True)
          (norm): PointSequential(
            (0): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          )
          (act): PointSequential(
            (0): GELU(approximate='none')
          )
        )
        (block0): Block(
          (cpe): PointSequential(
            (0): SubMConv3d(512, 512, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm)
            (1): Linear(in_features=512, out_features=512, bias=True)
            (2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          )
          (norm1): PointSequential(
            (0): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          )
          (ls1): PointSequential(
            (0): Identity()
          )
          (attn): SerializedAttention(
            (qkv): Linear(in_features=512, out_features=1536, bias=True)
            (proj): Linear(in_features=512, out_features=512, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (norm2): PointSequential(
            (0): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          )
          (ls2): PointSequential(
            (0): Identity()
          )
          (mlp): PointSequential(
            (0): MLP(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (drop_path): PointSequential(
            (0): DropPath(drop_prob=0.274)
          )
        )
        (block1): Block(
          (cpe): PointSequential(
            (0): SubMConv3d(512, 512, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm)
            (1): Linear(in_features=512, out_features=512, bias=True)
            (2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          )
          (norm1): PointSequential(
            (0): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          )
          (ls1): PointSequential(
            (0): Identity()
          )
          (attn): SerializedAttention(
            (qkv): Linear(in_features=512, out_features=1536, bias=True)
            (proj): Linear(in_features=512, out_features=512, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (norm2): PointSequential(
            (0): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          )
          (ls2): PointSequential(
            (0): Identity()
          )
          (mlp): PointSequential(
            (0): MLP(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (drop_path): PointSequential(
            (0): DropPath(drop_prob=0.287)
          )
        )
        (block2): Block(
          (cpe): PointSequential(
            (0): SubMConv3d(512, 512, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm)
            (1): Linear(in_features=512, out_features=512, bias=True)
            (2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          )
          (norm1): PointSequential(
            (0): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          )
          (ls1): PointSequential(
            (0): Identity()
          )
          (attn): SerializedAttention(
            (qkv): Linear(in_features=512, out_features=1536, bias=True)
            (proj): Linear(in_features=512, out_features=512, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (norm2): PointSequential(
            (0): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          )
          (ls2): PointSequential(
            (0): Identity()
          )
          (mlp): PointSequential(
            (0): MLP(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (drop_path): PointSequential(
            (0): DropPath(drop_prob=0.300)
          )
        )
      )
    )
    (dec): PointSequential(
      (dec3): PointSequential(
        (up): GridUnpooling(
          (proj): PointSequential(
            (0): Linear(in_features=512, out_features=384, bias=True)
            (1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (2): GELU(approximate='none')
          )
          (proj_skip): PointSequential(
            (0): Linear(in_features=384, out_features=384, bias=True)
            (1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (2): GELU(approximate='none')
          )
        )
        (block0): Block(
          (cpe): PointSequential(
            (0): SubMConv3d(384, 384, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm)
            (1): Linear(in_features=384, out_features=384, bias=True)
            (2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          )
          (norm1): PointSequential(
            (0): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          )
          (ls1): PointSequential(
            (0): Identity()
          )
          (attn): SerializedAttention(
            (qkv): Linear(in_features=384, out_features=1152, bias=True)
            (proj): Linear(in_features=384, out_features=384, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (norm2): PointSequential(
            (0): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          )
          (ls2): PointSequential(
            (0): Identity()
          )
          (mlp): PointSequential(
            (0): MLP(
              (fc1): Linear(in_features=384, out_features=1536, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=1536, out_features=384, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (drop_path): PointSequential(
            (0): DropPath(drop_prob=0.300)
          )
        )
        (block1): Block(
          (cpe): PointSequential(
            (0): SubMConv3d(384, 384, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm)
            (1): Linear(in_features=384, out_features=384, bias=True)
            (2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          )
          (norm1): PointSequential(
            (0): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          )
          (ls1): PointSequential(
            (0): Identity()
          )
          (attn): SerializedAttention(
            (qkv): Linear(in_features=384, out_features=1152, bias=True)
            (proj): Linear(in_features=384, out_features=384, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (norm2): PointSequential(
            (0): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          )
          (ls2): PointSequential(
            (0): Identity()
          )
          (mlp): PointSequential(
            (0): MLP(
              (fc1): Linear(in_features=384, out_features=1536, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=1536, out_features=384, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (drop_path): PointSequential(
            (0): DropPath(drop_prob=0.257)
          )
        )
      )
      (dec2): PointSequential(
        (up): GridUnpooling(
          (proj): PointSequential(
            (0): Linear(in_features=384, out_features=192, bias=True)
            (1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
            (2): GELU(approximate='none')
          )
          (proj_skip): PointSequential(
            (0): Linear(in_features=192, out_features=192, bias=True)
            (1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
            (2): GELU(approximate='none')
          )
        )
        (block0): Block(
          (cpe): PointSequential(
            (0): SubMConv3d(192, 192, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm)
            (1): Linear(in_features=192, out_features=192, bias=True)
            (2): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
          )
          (norm1): PointSequential(
            (0): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
          )
          (ls1): PointSequential(
            (0): Identity()
          )
          (attn): SerializedAttention(
            (qkv): Linear(in_features=192, out_features=576, bias=True)
            (proj): Linear(in_features=192, out_features=192, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (norm2): PointSequential(
            (0): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
          )
          (ls2): PointSequential(
            (0): Identity()
          )
          (mlp): PointSequential(
            (0): MLP(
              (fc1): Linear(in_features=192, out_features=768, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=768, out_features=192, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (drop_path): PointSequential(
            (0): DropPath(drop_prob=0.214)
          )
        )
        (block1): Block(
          (cpe): PointSequential(
            (0): SubMConv3d(192, 192, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm)
            (1): Linear(in_features=192, out_features=192, bias=True)
            (2): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
          )
          (norm1): PointSequential(
            (0): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
          )
          (ls1): PointSequential(
            (0): Identity()
          )
          (attn): SerializedAttention(
            (qkv): Linear(in_features=192, out_features=576, bias=True)
            (proj): Linear(in_features=192, out_features=192, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (norm2): PointSequential(
            (0): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
          )
          (ls2): PointSequential(
            (0): Identity()
          )
          (mlp): PointSequential(
            (0): MLP(
              (fc1): Linear(in_features=192, out_features=768, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=768, out_features=192, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (drop_path): PointSequential(
            (0): DropPath(drop_prob=0.171)
          )
        )
      )
      (dec1): PointSequential(
        (up): GridUnpooling(
          (proj): PointSequential(
            (0): Linear(in_features=192, out_features=96, bias=True)
            (1): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
            (2): GELU(approximate='none')
          )
          (proj_skip): PointSequential(
            (0): Linear(in_features=96, out_features=96, bias=True)
            (1): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
            (2): GELU(approximate='none')
          )
        )
        (block0): Block(
          (cpe): PointSequential(
            (0): SubMConv3d(96, 96, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm)
            (1): Linear(in_features=96, out_features=96, bias=True)
            (2): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
          )
          (norm1): PointSequential(
            (0): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
          )
          (ls1): PointSequential(
            (0): Identity()
          )
          (attn): SerializedAttention(
            (qkv): Linear(in_features=96, out_features=288, bias=True)
            (proj): Linear(in_features=96, out_features=96, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (norm2): PointSequential(
            (0): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
          )
          (ls2): PointSequential(
            (0): Identity()
          )
          (mlp): PointSequential(
            (0): MLP(
              (fc1): Linear(in_features=96, out_features=384, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=384, out_features=96, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (drop_path): PointSequential(
            (0): DropPath(drop_prob=0.129)
          )
        )
        (block1): Block(
          (cpe): PointSequential(
            (0): SubMConv3d(96, 96, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm)
            (1): Linear(in_features=96, out_features=96, bias=True)
            (2): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
          )
          (norm1): PointSequential(
            (0): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
          )
          (ls1): PointSequential(
            (0): Identity()
          )
          (attn): SerializedAttention(
            (qkv): Linear(in_features=96, out_features=288, bias=True)
            (proj): Linear(in_features=96, out_features=96, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (norm2): PointSequential(
            (0): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
          )
          (ls2): PointSequential(
            (0): Identity()
          )
          (mlp): PointSequential(
            (0): MLP(
              (fc1): Linear(in_features=96, out_features=384, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=384, out_features=96, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (drop_path): PointSequential(
            (0): DropPath(drop_prob=0.086)
          )
        )
      )
      (dec0): PointSequential(
        (up): GridUnpooling(
          (proj): PointSequential(
            (0): Linear(in_features=96, out_features=64, bias=True)
            (1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
            (2): GELU(approximate='none')
          )
          (proj_skip): PointSequential(
            (0): Linear(in_features=48, out_features=64, bias=True)
            (1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
            (2): GELU(approximate='none')
          )
        )
        (block0): Block(
          (cpe): PointSequential(
            (0): SubMConv3d(64, 64, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm)
            (1): Linear(in_features=64, out_features=64, bias=True)
            (2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
          )
          (norm1): PointSequential(
            (0): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
          )
          (ls1): PointSequential(
            (0): Identity()
          )
          (attn): SerializedAttention(
            (qkv): Linear(in_features=64, out_features=192, bias=True)
            (proj): Linear(in_features=64, out_features=64, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (norm2): PointSequential(
            (0): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
          )
          (ls2): PointSequential(
            (0): Identity()
          )
          (mlp): PointSequential(
            (0): MLP(
              (fc1): Linear(in_features=64, out_features=256, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=256, out_features=64, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (drop_path): PointSequential(
            (0): DropPath(drop_prob=0.043)
          )
        )
        (block1): Block(
          (cpe): PointSequential(
            (0): SubMConv3d(64, 64, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm)
            (1): Linear(in_features=64, out_features=64, bias=True)
            (2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
          )
          (norm1): PointSequential(
            (0): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
          )
          (ls1): PointSequential(
            (0): Identity()
          )
          (attn): SerializedAttention(
            (qkv): Linear(in_features=64, out_features=192, bias=True)
            (proj): Linear(in_features=64, out_features=64, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (norm2): PointSequential(
            (0): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
          )
          (ls2): PointSequential(
            (0): Identity()
          )
          (mlp): PointSequential(
            (0): MLP(
              (fc1): Linear(in_features=64, out_features=256, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=256, out_features=64, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (drop_path): PointSequential(
            (0): Identity()
          )
        )
      )
    )
  )
)