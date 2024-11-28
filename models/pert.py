import argparse
import torch
import torch.nn as nn


# parser = argparse.ArgumentParser()
# parser.add_argument('--batch_size', default=4, type=int, help='number of workers')
# args = parser.parse_args()

class Perturbation(nn.Module):
    def __init__(self):
        super().__init__()
        H = 256
        W = 256
        channel_num = 3
        cls_vectors = torch.empty(1,channel_num,H,W)
        nn.init.normal_(cls_vectors,std=0.01)
        self.pert = nn.Parameter(cls_vectors)

    def forward(self,x,train = True):
        if train is False:
            return x
        else:
            x = 0.01 * self.pert + x
            return x

# if __name__ == '__main__':
#     img = torch.randn(2,3,256,256)
#     model = Perturbation()
#     out = model(img)
#     print(type(out))
