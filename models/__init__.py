from .model import model_s
from .premodel import premodel_s
from .EFN import EFN_model
from .pert import Perturbation
# from .dehazeformer_main import dehazeformer1_t, dehazeformer1_s, dehazeformer1_b, dehazeformer1_d, dehazeformer1_w, dehazeformer1_m, dehazeformer1_l,DehazeFormer1
from .ConvNeXt import ConvNeXtBlock
from .part1 import Denoisy
# __factory = {
#     'dehazeformer-b': dehazeformer_b,
#     'dehazeformer-d': dehazeformer_d,
#     'dehazeformer-l': dehazeformer_l,
#     'dehazeformer-m': dehazeformer_m,
#     'dehazeformer-s': dehazeformer_s,
#     'dehazeformer-t':dehazeformer_t,
#     'dehazeformer-w':dehazeformer_w,
# }
#
# def init_model(name, pre_dir):
#     if name not in __factory.keys():
#         raise KeyError("Unknown model: {}".format(name))
#
#     net = __factory[name]()
#     checkpoint = torch.load(pre_dir)
#     state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
#     change = False
#     for k, v in state_dict.items():
#         if k[:6] == 'module':
#             change = True
#             break
#     if not change:
#         new_state_dict = state_dict
#     else:
#         from collections import OrderedDict
#         new_state_dict = OrderedDict()
#         for k,v in state_dict.items():
#             name = k[7:]
#             new_state_dict[name] = v
#     net.load_state_dict(new_state_dict)
#     net.eval()
#     net.volatile = True
#     return net
