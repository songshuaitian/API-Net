import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
from torch.utils.data import DataLoader
from collections import OrderedDict

from datasets import build_dataset, build_dataloader
from utils import AverageMeter, write_img, chw_to_hwc, parse_options
from datasets.loader import PairLoader
from models import *


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='model-s', type=str, help='model name')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--result_dir', default='./results/', type=str, help='path to results saving')
parser.add_argument('--dataset', default='rgb500', type=str, help='dataset name')
parser.add_argument('--exp', default='RTTS', type=str, help='experiment setting')
parser.add_argument('--hazy', default='hazy', type=str, help='experiment setting')
parser.add_argument('--gpu', default='1', type=str, help='GPUs used for training')


args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

def single(save_dir):
	state_dict = torch.load(save_dir)['state_dict']
	new_state_dict = OrderedDict()

	for k, v in state_dict.items():
		name = k[7:]
		new_state_dict[name] = v

	return new_state_dict


def test(test_loader, network, result_dir):
	PSNR = AverageMeter()
	SSIM = AverageMeter()

	torch.cuda.empty_cache()

	network.eval()

	os.makedirs(os.path.join(result_dir, 'imgs'), exist_ok=True)
	f_result = open(os.path.join(result_dir, 'results.csv'), 'w')

	for idx, batch in enumerate(test_loader):
		input = batch['lq'].cuda()
		target = batch['gt'].cuda()
		seg_img = batch['seg'].cuda()
		filename = os.path.basename(batch['lq_path'][0])
		with torch.no_grad():
			output = network(input, seg_img).clamp_(-1, 1)
			psnr_val = 10 * torch.log10(1 / F.mse_loss(output, target)).item()

			_, _, H, W = output.size()
			down_ratio = max(1, round(min(H, W) / 256))		# Zhou Wang
			ssim_val = ssim(F.adaptive_avg_pool2d(output, (int(H / down_ratio), int(W / down_ratio))), 
							F.adaptive_avg_pool2d(target, (int(H / down_ratio), int(W / down_ratio))), 
							data_range=1, size_average=False).item()				

		PSNR.update(psnr_val)
		SSIM.update(ssim_val)

		print('Test: [{0}]\t'
			  'PSNR: {psnr.val:.02f} ({psnr.avg:.02f})\t'
			  'SSIM: {ssim.val:.04f} ({ssim.avg:.04f})'
			  .format(idx, psnr=PSNR, ssim=SSIM))

		f_result.write('%s,%.02f,%.03f\n'%(filename, psnr_val, ssim_val))
		out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
		out_img_hazy = chw_to_hwc(input.detach().cpu().squeeze(0).numpy())
		write_img(os.path.join(result_dir, 'imgs', filename), out_img)

	f_result.close()

	os.rename(os.path.join(result_dir, 'results.csv'), 
			  os.path.join(result_dir, '%.02f | %.04f.csv'%(PSNR.avg, SSIM.avg)))


if __name__ == '__main__':
	network = eval(args.model.replace('-', '_'))()
	network.cuda()
	saved_model_dir = os.path.join(args.save_dir, args.exp, 'API-Net.pth')

	if os.path.exists(saved_model_dir):
		print('==> Start testing, current model name: ' + args.model)
		network.load_state_dict(single(saved_model_dir))
	else:
		print('==> No existing trained model!')
		exit(0)


	root_path = '../API'#当前项目路径
	opt, _ = parse_options(root_path, is_train=False)
	test_loaders = []
	for _, dataset_opt in sorted(opt['datasets'].items()):
		test_set = build_dataset(dataset_opt)
		test_loader = build_dataloader(
			test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
	result_dir = os.path.join(args.result_dir, args.dataset, args.model)


	test(test_loader, network, result_dir)