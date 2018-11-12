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

# # The potential is unique up to a constant, so we pick the "gauge" or normalization that it must integrate to zero.
# print('Before normalization: sum(phi_nh_rec)={:g}, mean(phi_nh_rec)={:g}'.format(np.sum(a), np.mean(np.mean(a))))
# a = a - np.mean(a)
# a = a / (8*Nx*Ny*Nz)
# print('After normalization: sum(phi_nh_rec)={:g}, mean(phi_nh_rec)={:g}'.format(np.sum(a), np.mean(np.mean(a))))

print('array: shape={:}'.format(a.shape))
print('stats: sum={:g}, mean={:g}, min={:g}, max={:g}, argmin={:}, argmax={:}'
    .format(a.sum(), a.mean(), a.min(), a.max(),
        np.unravel_index(a.argmin(), a.shape), np.unravel_index(a.argmax(), a.shape)))
print('')

fig, ax = plt.subplots()

im = ax.imshow(a[0,:,:], vmin=-a.max()/10, vmax=a.max()/10, cmap='RdBu_r', interpolation='none', animated=True)
plt.colorbar(im)

time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                    bbox=dict(boxstyle='square', ec='black', fc='lightgray'))

def init_fig():
    pass

def animate_fig(i):
    im.set_array(a[i,:,:])
    time_text.set_text('lvl = {:d}'.format(i))
    print('i={:d}'.format(i))
    print(a[i,:,:])
    print('({:d},0,0)->{:e}'.format(i, a[i,0,0]))
    print('({:d},20,20)->{:e}'.format(i, a[i,20,20]))
    return im, ax

anim = animation.FuncAnimation(fig, animate_fig, frames=Nz, blit=True)
plt.show()

plt.close('all')

# for i in range(3):
#     fig = plt.figure()
#     ax = fig.gca()

#     im = ax.pcolormesh(a[i,:,:], vmin=-a.max()/10, vmax=a.max()/10, cmap='RdBu_r')
#     fig.colorbar(im)
#     plt.title('z_index = {:d}'.format(i))

#     plt.savefig('phi_nh_reconstruction_' + str(i).zfill(2) + '.png', dpi=300, format='png', transparent=False)

#     plt.close('all')
