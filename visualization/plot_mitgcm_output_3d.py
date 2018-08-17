from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import xmitgcm

data_dir = '/home/alir/MITgcm/verification/tutorial_deep_convection/run'
variables = ['U', 'V', 'W', 'S', 'T', 'Eta']

print('Loading dataset...')
ds = xmitgcm.open_mdsdataset(data_dir, geometry='cartesian', iters=np.arange(0, 1001, 5), prefix=variables)
print(ds)

for i in np.arange(0, 1001, 5):
	fig = plt.figure()
	ax = fig.gca(projection='3d')

	theta_z_slice = ds['T'].sel(time=i, Z=-10.0, XC=slice(0, 1000), YC=slice(0, 1000))
	theta_x_slice = ds['T'].sel(time=i, XC=990.0, YC=slice(0, 1000))
	theta_y_slice = ds['T'].sel(time=i, YC=990.0, XC=slice(0, 1000))

	XC_z, YC_z = np.meshgrid(theta_z_slice.XC.values, theta_z_slice.YC.values)
	YC_x, ZC_x = np.meshgrid(theta_x_slice.YC.values, theta_x_slice.Z.values)
	XC_y, ZC_y = np.meshgrid(theta_y_slice.XC.values, theta_y_slice.Z.values)

	cf1 = ax.contourf(XC_z, YC_z, theta_z_slice.values, zdir='z', offset=-10, levels=np.arange(19.89, 20.01, 0.01), cmap='plasma')
	cf2 = ax.contourf(theta_x_slice.values, YC_x, ZC_x, zdir='x', offset=990, levels=np.arange(19.89, 20.01, 0.01), cmap='plasma')
	cf3 = ax.contourf(XC_y, theta_y_slice.values, ZC_y, zdir='y', offset=990, levels=np.arange(19.89, 20.01, 0.01), cmap='plasma')

	clb = fig.colorbar(cf1)
	clb.ax.set_title(r'$\theta$ (°C)')

	ax.set_xlim3d(0, 1000)
	ax.set_ylim3d(0, 1000)
	ax.set_zlim3d(-1000, 0)

	ax.set_xlabel('XC (m)')
	ax.set_ylabel('YC (m)')
	ax.set_zlabel('Z (m)')

	ax.view_init(elev=30, azim=45)

	# plt.show()

	filename = 'surface_temp_3d_' + str(int(i/5)).zfill(3) + '.png'
	plt.savefig(filename, dpi=300, format='png', transparent=False)
	print('Saving: {:s}'.format(filename))

	plt.close('all')