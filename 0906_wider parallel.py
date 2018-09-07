from subprocess import call
import sys

widths = list(range(18, 24+2, 2))
print(widths)

for width in widths:
	call(' '.join(['CUDA_VISIBLE_DEVICES=0,1,2', 
					'python', 'main.py',
					'--width %d' %(width),
					'--outf output/wider_%d/' %(width),
					'--parallel']), 
					shell=True)