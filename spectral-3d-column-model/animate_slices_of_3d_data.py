import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-f', '--file', dest='filename', help='Data file to animate.')
args = parser.parse_args()

with open(args.filename, 'rb') as f:
    Nx, Ny, Nz = 64, 64, 64

    record_length = 8  # [bytes]

    # f.seek(level * record_length * nx*ny, os.SEEK_SET)

    data = np.fromfile(f, dtype='f8', count=Nx*Ny*Nz)
    array = np.reshape(data, [Nx, Ny, Nz], order='F')

print(array.shape)
print(array.mean())

fig, ax = plt.subplots()

im = ax.imshow(array[0], vmin=-0.1, vmax=0.1, animated=True)
plt.colorbar(im)

time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
    bbox=dict(boxstyle='square', ec='black', fc='lightgray'))

def init_fig():
    pass

def animate_fig(i):
    im.set_array(array[i])
    time_text.set_text('i = {:d}'.format(i))
    return im, ax


ani = animation.FuncAnimation(fig, animate_fig, frames=Nz, blit=True)
plt.show()