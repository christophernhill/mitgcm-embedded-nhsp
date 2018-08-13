import numpy as np
import matplotlib.pyplot as plt

import xmitgcm

data_dir = '/home/alir/MITgcm/verification/tutorial_deep_convection/run'
variables = ['U', 'V', 'W', 'S', 'T', 'Eta']

ds = xmitgcm.open_mdsdataset(data_dir, geometry='cartesian', iters=np.arange(0, 101), prefix=variables)
print(ds)

surface_temp = ds['T'].sel(time=3, Z=-10.0)
surface_temp.plot()
plt.show()