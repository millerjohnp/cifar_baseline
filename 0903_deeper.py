from subprocess import call
import sys

gpu = int(sys.argv[1])
print('Running on GPU %d' % (gpu))

Ns = [4, 8, 16, 32, 64]
depths = [6*N+2 for N in Ns]
print(depths)

for depth in depths:
	call(' '.join(['CUDA_VISIBLE_DEVICES=%d' %(gpu), 
					'python', 'main.py',
					'--depth %d' %(depth),
					'--outf output/deeper_%d/' %(depth)]), 
					shell=True)