import torch
import numpy as np

import matplotlib
matplotlib.use('Agg') # before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

filename = '0906_overfit'
outroot = 'output/'
modes = ['deeper', 'wider']

settings = {}
settings['deeper'] = [26, 50, 98, 194, 386]
settings['wider'] = [1, 2, 4, 6, 8]

for mode in modes:
	te_err = []
	tr_loss = []

	for setting in settings[mode]:
		ckpt = torch.load('%s/%s_%d/ckpt_info.pth' %(outroot, mode, setting))
		te_err.append(ckpt['te_err'] * 100)
		tr_loss.append(ckpt['tr_loss'] * 3)

	plt.plot(settings[mode], te_err, label='test error (%)', color='r')
	plt.plot(settings[mode], tr_loss, label='training loss (scaled)', color='b')
	plt.legend()
	plt.ylim([0, 10])
	plt.savefig('%s/%s_%s.pdf' %(outroot, filename, mode))
	plt.close()