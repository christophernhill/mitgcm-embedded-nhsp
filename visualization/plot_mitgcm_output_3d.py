import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import xmitgcm

from joblib import Parallel, delayed

data_dir = '/nobackup1b/users/alir/MITgcm_mpi/verification/tutorial_deep_convection/run'
variables = ['T']

iter_stride = 20
x_offset = 1990.0
y_offset = 990.0
z_offset = -10.0

def process_iters(iter_start, iter_end):
    print('Loading dataset (iter_start={:d}, iter_end={:d})...'.format(iter_start, iter_end))
    ds = xmitgcm.open_mdsdataset(data_dir, geometry='cartesian', iters=np.arange(iter_start, iter_end, iter_stride), prefix=variables)
    # print(ds)

    for i in np.arange(iter_start, iter_end, iter_stride):
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        theta_z_slice = ds['T'].sel(time=i, Z=z_offset, YC=slice(0, 1000))
        theta_x_slice = ds['T'].sel(time=i, XC=x_offset, YC=slice(0, 1000))
        theta_y_slice = ds['T'].sel(time=i, YC=y_offset)  #, XC=slice(0, 1000))

        XC_z, YC_z = np.meshgrid(theta_z_slice.XC.values, theta_z_slice.YC.values)
        YC_x, ZC_x = np.meshgrid(theta_x_slice.YC.values, theta_x_slice.Z.values)
        XC_y, ZC_y = np.meshgrid(theta_y_slice.XC.values, theta_y_slice.Z.values)

        cf1 = ax.contourf(XC_z, YC_z, theta_z_slice.values, zdir='z', offset=z_offset, levels=np.arange(19.89, 20.01, 0.001), cmap='jet')
        cf2 = ax.contourf(theta_x_slice.values, YC_x, ZC_x, zdir='x', offset=x_offset, levels=np.arange(19.89, 20.01, 0.001), cmap='jet')
        cf3 = ax.contourf(XC_y, theta_y_slice.values, ZC_y, zdir='y', offset=y_offset, levels=np.arange(19.89, 20.01, 0.001), cmap='jet')

        clb = fig.colorbar(cf1, ticks=[19.90, 19.92, 19.94, 19.96, 19.98, 20.00])
        clb.ax.set_title(r'$\theta$ (°C)')

        # norm = matplotlib.colors.Normalize(vmin=19.89, vmax=20.01)

        # Z_offset = z_offset*np.ones((50, 100))
        # X_offset = x_offset*np.ones((50, 50))
        # Y_offset = y_offset*np.ones((50, 100))

        # cf1 = ax.plot_surface(XC_z, YC_z, Z_offset, rstride=1, cstride=1, linewidth=0, alpha=1, vmin=19.89, vmax=20.01,
        #                       antialiased=False, shade=False, facecolors=plt.cm.jet(norm(theta_z_slice.values)))
        # cf2 = ax.plot_surface(X_offset, YC_x, ZC_x, rstride=1, cstride=1, linewidth=0, alpha=1, vmin=19.89, vmax=20.01,
        #                       antialiased=False, shade=False, facecolors=plt.cm.jet(norm(theta_x_slice.values)))
        # cf3 = ax.plot_surface(XC_y, Y_offset, ZC_y, rstride=1, cstride=1, linewidth=0, alpha=1, vmin=19.89, vmax=20.01,
        #                       antialiased=False, shade=False, facecolors=plt.cm.jet(norm(theta_y_slice.values)))

        ax.set_xlim3d(0, 2000)
        ax.set_ylim3d(0, 2000)
        ax.set_zlim3d(-1000, 0)

        ax.set_xlabel('XC (m)')
        ax.set_ylabel('YC (m)')
        ax.set_zlabel('Z (m)')

        ax.view_init(elev=30, azim=45)

        ax.set_title('t = {:07d} s'.format(i*30))

        ax.set_xticks([0, 500, 1000, 1500, 2000])
        ax.set_yticks([0, 500, 1000, 1500, 2000])
        ax.set_zticks([0, -200, -400, -600, -800, -1000])

        # plt.show()

        filename = 'surface_temp_3d_' + str(int(i/20)).zfill(4) + '.png'
        plt.savefig(filename, dpi=300, format='png', transparent=False)
        print('Saving: {:s}'.format(filename))

        plt.close('all')

Parallel(n_jobs=28)(delayed(process_iters)(i, i+1001) for i in np.arange(0, 50001, 1000))
# process_iters(35000, 35101)