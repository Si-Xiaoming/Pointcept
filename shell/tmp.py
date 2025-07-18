

def y01():
    import torch_scatter
    import torch
    print(torch_scatter.__version__)
    print(torch_scatter.segment_csr(torch.rand(10, 3).cuda(), torch.tensor([0, 5, 10]).cuda(),
                                    reduce="max"))  # 应输出 CUDA 张量


def y02():
    import torch
    print(torch.__version__)  # 应为 2.5.0
    print(torch.version.cuda)  # 应为 12.4

if __name__ == '__main__':
    y01()