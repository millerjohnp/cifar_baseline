from subprocess import call
import sys

Ns = [512, 1024]
depths = [6*N+2 for N in Ns]
print(depths)

for depth in depths:
	call(' '.join(['CUDA_VISIBLE_DEVICES=0,1,2,3', 
					'python', 'main.py',
					'--depth %d' %(depth),
					'--outf output/deeper_%d/' %(depth),
					'--parallel']), 
					shell=True)