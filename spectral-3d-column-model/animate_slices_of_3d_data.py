import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-in', '--in-file', dest='in_filename', help='Input array binary file.')
parser.add_argument('-out', '--out-file', dest='out_filename', help='Output array binary file.')
parser.add_argument('-rec', '--rec-file', dest='rec_filename', help='Reconstructed array binary file.')
args = parser.parse_args()

Nx, Ny, Nz = 32, 32, 32
record_length = 8  # [bytes]

with open(args.in_filename, 'rb') as f:
    data = np.fromfile(f, dtype='f8', count=Nx*Ny*Nz)
    in_array = np.reshape(data, [Nx, Ny, Nz], order='F')

with open(args.out_filename, 'rb') as f:
    data = np.fromfile(f, dtype='f8', count=Nx*Ny*Nz)
    out_array = np.reshape(data, [Nx, Ny, Nz], order='F')

with open(args.rec_filename, 'rb') as f:
    data = np.fromfile(f, dtype='f8', count=Nx*Ny*Nz)
    rec_array = np.reshape(data, [Nx, Ny, Nz], order='F')

rec_array = (rec_array / (Nx*Ny*Nz))  # Scaling reconstruction due to unnormalized IFFT.

residual_array = in_array - rec_array

array_names = ['input array', 'output array', 'reconstructed array', 'residual array']

for i, a in enumerate([in_array, out_array, rec_array, residual_array]):
    print('{:s}: shape={:}'.format(array_names[i], a.shape))
    print('stats: mean={:g}, min={:g}, max={:g}, argmin={:}, argmax={:}'
        .format(a.mean(), a.min(), a.max(),
            np.unravel_index(a.argmin(), a.shape), np.unravel_index(a.argmax(), a.shape)))
    print('')

fig, ax = plt.subplots()

im = ax.imshow(out_array[0], vmin=-1, vmax=1, animated=True)
plt.colorbar(im)

time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
    bbox=dict(boxstyle='square', ec='black', fc='lightgray'))

def init_fig():
    pass

def animate_fig(i):
    im.set_array(out_array[i])
    time_text.set_text('lvl = {:d}'.format(i))
    return im, ax


ani = animation.FuncAnimation(fig, animate_fig, frames=Nz, blit=True)
plt.show()