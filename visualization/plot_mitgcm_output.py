import numpy as np
import matplotlib.pyplot as plt
import holoviews as hv
import xmitgcm

hv.extension('matplotlib')

data_dir = '/home/alir/MITgcm/verification/tutorial_deep_convection/run'
variables = ['U', 'V', 'W', 'S', 'T', 'Eta']

ds = xmitgcm.open_mdsdataset(data_dir, geometry='cartesian', iters=np.arange(0, 1001, 5), prefix=variables)
print(ds)

# surface_temp = ds['T'].sel(time=np.arange(0, 101), Z=-10.0)
# images = surface_temp.to(hv.Image, ['XC', 'YC']).options(fig_inches=(10, 10), colorbar=True, cmap='viridis')
# renderer = hv.renderer('matplotlib')
# renderer.save(images, 'hv_anim', 'mp4')

for i in np.arange(0, 1001, 5):
	surface_temp = ds['T'].sel(time=i, Z=-10.0)
	surface_temp.plot()
	
	n = int(i/5)
	plt.savefig('surface_temp_' + str(n).zfill(3) + '.png', dpi=300, format='png', transparent=False)
	print('Saving: surface_temp_' + str(n).zfill(3) + '.png')
	plt.clf()