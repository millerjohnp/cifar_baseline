import torch
import torch.utils.data as data
import numpy as np
import os
import pdb

class SplitDataset(data.Dataset):
	def __init__(self, joint_set, permute):
		self.joint_set = joint_set
		self.permute = permute

	def __getitem__(self, index):
		return self.joint_set.__getitem__(self.permute[index])

	def __len__(self):
		return len(self.permute)

def split_validation(orset, va_size, seed, permute_path='utils/split_{}.pth'):
	state = np.random.get_state()
	np.random.seed(seed)

	if permute_path is not None:
		if os.path.isfile(permute_path.format(seed)):
			permute = torch.load(permute_path.format(seed))
		else:	
			permute = np.random.permutation(len(orset))
			torch.save(permute, permute_path.format(seed))
			print('Created new dataset permutation!')
	else:
		permute = np.random.permutation(len(orset))

	np.random.set_state(state)

	trset = SplitDataset(orset, permute[:len(orset)-va_size])
	vaset = SplitDataset(orset, permute[len(orset)-va_size:])
	return trset, vaset