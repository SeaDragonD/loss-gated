import torch, sys, os, tqdm, numpy, time, faiss, gc, soundfile
import torch.nn as nn
import torch.nn.functional as F

from tools import *
from loss import *
from encoder import *
from collections import defaultdict
from sklearn.metrics.cluster import normalized_mutual_info_score

class trainer(nn.Module):
	def __init__(self, lr, n_class, **kwargs):
		super(trainer, self).__init__()
		self.Network = ECAPA_TDNN(C = 512).cuda() # Speaker encoder
		self.Loss = LossFunction(num_class = n_class).cuda() # Classification layer
		self.criterion = torch.nn.CrossEntropyLoss()
		self.Optim = torch.optim.Adam(list(self.Network.parameters()) + list(self.Loss.parameters()), lr = lr) # Adam, learning rate is fixed

	def train_network(self, epoch, loader):
		self.train()
		loss, index, top1 = 0, 0, 0
		time_start = time.time()
		for num, (data, label) in enumerate(loader, start = 1):
			data = data.transpose(0, 1)
			feat, res_label = [], []
			for inp in data:
				x_em = self.Network.forward(torch.FloatTensor(inp).cuda())
				x_clas = self.Loss.forward(x_em)
				res_label.append(x_clas)
			res_label = torch.stack(res_label, dim=1).squeeze()

			self.zero_grad()

			res_label = res_label.to('cuda:0')  # 确保模型的输出也在正确的设备上，如果它已经在cuda:0上，则这一行不是必需的
			label = label.to('cuda:0')  # 将真实标签移到cuda:0上

			closs = self.criterion(res_label, label)

			closs.backward()
			self.Optim.step()

			prec1 = accuracy_sup(res_label, label)

			loss += closs.detach().cpu().numpy()
			index += len(label)

			top1 += prec1
			time_used = time.time() - time_start
			sys.stderr.write("[Train] [%2d] %.2f%% (est %.1f mins), Loss: %.3f, ACC: %.2f%%\r" %\
			(epoch, 100 * (num / loader.__len__()), time_used * loader.__len__() / num / 60, \
			loss / num, top1 / num))
			sys.stderr.flush()
		sys.stdout.write("\n")
		torch.cuda.empty_cache()
		return loss / num, top1 / num

	def eval_network(self,loader,is_val = False, **kwargs):
		self.eval()
		loss, index, nselects, top1 = 0, 0, 0, 0
		if is_val:
			test_type = 'Val'
		else:
			test_type = 'Test'
		time_start = time.time()
		with torch.no_grad():
			for num, (data, label) in enumerate(loader, start=1):
				data = data.transpose(0, 1)
				feat, res_label = [], []
				for inp in data:
					x_em = self.Network.forward(torch.FloatTensor(inp).cuda())
					x_clas = self.Loss.forward(x_em)
					res_label.append(x_clas)
				res_label = torch.stack(res_label, dim=1).squeeze()

				res_label = res_label.to('cuda:0')  # 确保模型的输出也在正确的设备上，如果它已经在cuda:0上，则这一行不是必需的
				label = label.to('cuda:0')  # 将真实标签移到cuda:0上

				closs = self.criterion(res_label, label)
				prec1 = accuracy_sup(res_label, label)

				loss += closs.detach().cpu().numpy()
				index += len(label)

				top1 += prec1
				time_used = time.time() - time_start
				sys.stderr.write("[%s]  %.2f%% (est %.1f mins), Loss: %.3f, ACC: %.2f%%\r" % \
								 ( test_type, 100 * (num / loader.__len__()), time_used * loader.__len__() / num / 60, \
								  loss / num, top1 / num))
				sys.stderr.flush()
			sys.stdout.write("\n")
			torch.cuda.empty_cache()

		return loss / num, top1 / num
		# files, feats = [], {}
		# for line in open(val_list).read().splitlines():
		# 	data = line.split()
		# 	files.append(data[1])
		# 	files.append(data[2])
		# setfiles = list(set(files))
		# setfiles.sort()  # Read the list of wav files
		# for idx, file in tqdm.tqdm(enumerate(setfiles), total = len(setfiles)):
		# 	audio, _ = soundfile.read(os.path.join(val_path, file))
		# 	feat = torch.FloatTensor(numpy.stack([audio], axis=0)).cuda()
		# 	with torch.no_grad():
		# 		ref_feat = self.Network.forward(feat).detach().cpu()
		# 	feats[file]     = ref_feat # Extract features for each data, get the feature dict
		# scores, labels  = [], []
		# for line in open(val_list).read().splitlines():
		# 	data = line.split()
		# 	ref_feat = F.normalize(feats[data[1]].cuda(), p=2, dim=1) # feature 1
		# 	com_feat = F.normalize(feats[data[2]].cuda(), p=2, dim=1) # feature 2
		# 	score = numpy.mean(torch.matmul(ref_feat, com_feat.T).detach().cpu().numpy()) # Get the score
		# 	scores.append(score)
		# 	labels.append(int(data[0]))
		# EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
		# fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
		# minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)
		# return [EER, minDCF]

	def save_parameters(self, path):
		torch.save(self.state_dict(), path)

	def load_parameters(self, path):
		self_state = self.state_dict()
		loaded_state = torch.load(path)
		print("Model %s loaded!"%(path))
		for name, param in loaded_state.items():
			origname = name
			if name not in self_state:
				name = name.replace("module.", "")
				if name not in self_state:
					# print("%s is not in the model."%origname)
					continue
			if self_state[name].size() != loaded_state[origname].size():
				print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
				continue
			self_state[name].copy_(param)