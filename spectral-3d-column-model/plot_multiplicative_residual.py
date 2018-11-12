import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('-f1', '--file1', dest='filename1', help='Input file 1.')
parser.add_argument('-f2', '--file2', dest='filename2', help='Input file 2.')
args = parser.parse_args()

Nx, Ny, Nz = 100, 100, 50
record_length = 8  # [bytes]

with open(args.filename1, 'rb') as f:
    data1 = np.fromfile(f, dtype='f8', count=Nx*Ny*Nz)
    a1 = np.reshape(data1, [Nz, Ny, Nx], order='C')

with open(args.filename2, 'rb') as f:
    data2 = np.fromfile(f, dtype='f8', count=Nx*Ny*Nz)
    a2 = np.reshape(data2, [Nz, Ny, Nx], order='C')

res = np.divide(a1, a2)

print('array1: shape={:}'.format(a1.shape))
print('stats1: sum={:g}, mean={:g}, min={:g}, max={:g}, argmin={:}, argmax={:}'
    .format(a1.sum(), a1.mean(), a1.min(), a1.max(),
        np.unravel_index(a1.argmin(), a1.shape), np.unravel_index(a1.argmax(), a1.shape)))
print('')

print('array2: shape={:}'.format(a2.shape))
print('stats2: sum={:g}, mean={:g}, min={:g}, max={:g}, argmin={:}, argmax={:}'
    .format(a2.sum(), a2.mean(), a2.min(), a2.max(),
        np.unravel_index(a2.argmin(), a2.shape), np.unravel_index(a2.argmax(), a2.shape)))
print('')

print('mulres: shape={:}'.format(res.shape))
print('stats : sum={:g}, mean={:g}, min={:g}, max={:g}, argmin={:}, argmax={:}'
    .format(res.sum(), res.mean(), res.min(), res.max(),
        np.unravel_index(res.argmin(), res.shape), np.unravel_index(res.argmax(), res.shape)))
print('')

fig, ax = plt.subplots()

im = ax.imshow(res[0,:,:], vmin=-res.mean(), vmax=res.mean(), cmap='RdBu_r', interpolation='none', animated=True)
plt.colorbar(im)

time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                    bbox=dict(boxstyle='square', ec='black', fc='lightgray'))

def init_fig():
    pass

def animate_fig(i):
    im.set_array(res[i,:,:])
    time_text.set_text('lvl = {:d}'.format(i))
    return im, ax

anim = animation.FuncAnimation(fig, animate_fig, frames=Nz, blit=True)
plt.show()