import torch
import numpy as np

import matplotlib
matplotlib.use('Agg') # before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

filename = 'RN110_baseline_runs'
outroot = 'output/RN110_baseline_runs'
outfs = ['default', 'pad0', 'bs64']
colors = ['r', 'b', 'g']

for outf, color in zip(outfs, colors):
	all_loss = torch.load('%s/%s/loss.pth' %(outroot, outf))
	y_tr, _, y_va, _, y_te, _ = zip(*all_loss)

	y_te = np.asarray(y_te)
	plt.plot(y_te*100, label=outf, color=color)

	state = torch.load('%s/%s/ckpt.pth' %(outroot, outf))
	epoch = state['epoch']
	print('best test by validation for %s: %.2f%%' %(outf, y_te[epoch]*100))
	plt.plot([epoch], y_te[epoch]*100, label=outf, marker='x', color=color)

plt.legend()
plt.ylim([3, 20])
plt.savefig('%s/%s.pdf' %(outroot, filename))
plt.close()