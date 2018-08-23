import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-f', '--file', dest='filename', help='Input file.')
args = parser.parse_args()

Nx, Ny, Nz = 100, 100, 50
record_length = 8  # [bytes]

with open(args.filename, 'rb') as f:
    data = np.fromfile(f, dtype='f8', count=Nx*Ny*Nz)
    a = np.reshape(data, [Nz, Ny, Nx], order='C')

print('array: shape={:}'.format(a.shape))
print('stats: mean={:g}, min={:g}, max={:g}, argmin={:}, argmax={:}'
    .format(a.mean(), a.min(), a.max(),
        np.unravel_index(a.argmin(), a.shape), np.unravel_index(a.argmax(), a.shape)))
print('')

fig, ax = plt.subplots()

im = ax.imshow(a[0,:,:], vmin=-a.max()/10, vmax=a.max()/10, cmap='RdBu', animated=True)
plt.colorbar(im)

time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
    bbox=dict(boxstyle='square', ec='black', fc='lightgray'))

def init_fig():
    pass

def animate_fig(i):
    im.set_array(a[i,:,:])
    time_text.set_text('lvl = {:d}'.format(i))
    return im, ax


ani = animation.FuncAnimation(fig, animate_fig, frames=Nz, blit=True)
plt.show()