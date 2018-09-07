from subprocess import call
import sys

gpu = int(sys.argv[1])
print('Running on GPU %d' % (gpu))

widths = list(range(10, 16+2, 2))
print(widths)

for width in widths:
	call(' '.join(['CUDA_VISIBLE_DEVICES=%d' %(gpu), 
					'python', 'main.py',
					'--width %d' %(width),
					'--outf output/wider_%d/' %(width)]), 
					shell=True)