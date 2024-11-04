import numpy as np
from metpy.units import units

distance = np.arange(1, 5) * units.meters
time = units.Quantity(np.arange(2, 10, 2), 'sec')


