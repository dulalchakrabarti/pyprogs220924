# Copyright (c) 2016,2017 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""
===========================
Upper Air Sounding Tutorial
===========================

Upper air analysis is a staple of many synoptic and mesoscale analysis
problems. In this tutorial we will gather weather balloon data, plot it,
perform a series of thermodynamic calculations, and summarize the results.
To learn more about the Skew-T diagram and its use in weather analysis and
forecasting, checkout `this <http://www.pmarshwx.com/research/manuals/AF_skewt_manual.pdf>`_
air weather service guide.
"""


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd

import metpy.calc as mpcalc
from metpy.cbook import get_test_data
from metpy.plots import Hodograph, SkewT
from metpy.units import units
# Create a datetime for our request - notice the times are from laregest (year) to smallest (hour)
from datetime import datetime
sl = {'42182':'New Delhi','42809':'Kolkata','43279':'Chennai','43285':'Mangalore','42410':'Guwahati','42867':'Nagpur','43003':'Santacruz','43150':'Visakhapatnam','42971':'Bhubaneswar','42701':'Ranchi','42369':'Ranchi','42101':'Patiala','42724':'Agartala','42101':'Patiala'}
request_time = datetime(2026, 4, 30, 0)
time_str = str(request_time)
# Store the station name in a variable for flexibility and clarity
station = '42101'
station1 = ['42182','42809','43279','43285','42410','42867']
# Import the Wyoming simple web service and request the data
# Don't worry about a possible warning from Pandas - it's related to our handling of units
from siphon.simplewebservice.wyoming import WyomingUpperAir
df = WyomingUpperAir.request_data(request_time, station)
# Drop any rows with all NaN values for T, Td, winds
df = df.dropna(subset=('temperature', 'dewpoint', 'direction', 'speed'
                       ), how='all').reset_index(drop=True)

# We will pull the data out of the example dataset into individual variables and
# assign units.

p = df['pressure'].values * units.hPa
T = df['temperature'].values * units.degC
Td = df['dewpoint'].values * units.degC
wind_speed = df['speed'].values * units.knots
wind_dir = df['direction'].values * units.degrees
u, v = mpcalc.wind_components(wind_speed, wind_dir)

##########################################################################
# Thermodynamic Calculations
# --------------------------
#
# Often times we will want to calculate some thermodynamic parameters of a
# sounding. The MetPy calc module has many such calculations already implemented!
#
# * **Lifting Condensation Level (LCL)** - The level at which an air parcel's
#   relative humidity becomes 100% when lifted along a dry adiabatic path.
# * **Parcel Path** - Path followed by a hypothetical parcel of air, beginning
#   at the surface temperature/pressure and rising dry adiabatically until
#   reaching the LCL, then rising moist adiabatially.

# Calculate the LCL
lcl_pressure, lcl_temperature = mpcalc.lcl(p[0], T[0], Td[0])

#print(lcl_pressure, lcl_temperature)

# Calculate the parcel profile.
parcel_prof = mpcalc.parcel_profile(p, T[0], Td[0]).to('degC')
cape, cin = mpcalc.cape_cin(p, T, Td, parcel_prof)
#print(cape,cin)
# Calculate thermodynamics
lcl_pressure, lcl_temperature = mpcalc.lcl(p[0],
                                           T[0],
                                           Td[0])

lfc_pressure, lfc_temperature = mpcalc.lfc(p,
                                           T,
                                           Td)

el_pressure, el_temperature = mpcalc.el(p,
                                        T,
                                        Td)


##########################################################################
# Adding a Hodograph
# ------------------
#
# A hodograph is a polar representation of the wind profile measured by the rawinsonde.
# Winds at different levels are plotted as vectors with their tails at the origin, the angle
# from the vertical axes representing the direction, and the length representing the speed.
# The line plotted on the hodograph is a line connecting the tips of these vectors,
# which are not drawn.

# Create a new figure. The dimensions here give a good aspect ratio
fig = plt.figure(figsize=(14, 14))
sup = sl[station]+' '+time_str
fig.suptitle(sup, fontsize=20)
skew = SkewT(fig, rotation=30)

# Plot the data using normal plotting functions, in this case using
# log scaling in Y, as dictated by the typical meteorological plot
skew.plot(p, T, 'r')
skew.plot(p, Td, 'g')
skew.plot_barbs(p, u, v)
skew.ax.set_ylim(1000, 50)
skew.ax.set_xlim(-40, 60)

# Plot LCL as black dot
skew.plot(lcl_pressure, lcl_temperature, 'ko', markerfacecolor='black')

# Plot the parcel profile as a black line
skew.plot(p, parcel_prof, 'k', linewidth=2)

# Shade areas of CAPE and CIN
skew.shade_cin(p, T, parcel_prof)
skew.shade_cape(p, T, parcel_prof)

# Plot a zero degree isotherm
skew.ax.axvline(0, color='c', linestyle='--', linewidth=2)

# Add the relevant special lines
skew.plot_dry_adiabats()
skew.plot_moist_adiabats()
skew.plot_mixing_lines()
if lcl_pressure:
    skew.ax.plot(lcl_temperature, lcl_pressure, marker="_", color='black', markersize=30, markeredgewidth=3)
    # Adding the text label
    skew.ax.text(lcl_temperature, lcl_pressure, '         LCL',  verticalalignment='center', 
                 horizontalalignment='left', 
                 fontsize=12, 
                 fontweight='bold', 
                 color='black')    
if lfc_pressure:
    skew.ax.plot(lfc_temperature, lfc_pressure, marker="_", color='brown', markersize=30, markeredgewidth=3)
    skew.ax.text(lfc_temperature, lfc_pressure, '               LFC',  verticalalignment='center', 
                 horizontalalignment='left', 
                 fontsize=12, 
                 fontweight='bold', 
                 color='brown')    
    
if el_pressure:
    skew.ax.plot(el_temperature, el_pressure, marker="_", color='blue', markersize=30, markeredgewidth=3)
    skew.ax.text(el_temperature, el_pressure, '         EL',  verticalalignment='center', 
                 horizontalalignment='left', 
                 fontsize=12, 
                 fontweight='bold', 
                 color='blue')    
    skew.ax.text(el_temperature, (el_pressure+75*units.hPa), '                       CAPE '+str(cape) +'\n'+'                       CIN '+str(cin),  verticalalignment='center', 
                 horizontalalignment='left', 
                 fontsize=12, 
                 fontweight='bold', 
                 color='red')    


# Create a hodograph
# Create an inset axes object that is 50% width and height of the
# figure and put it in the upper right hand corner.
ax_hod = inset_axes(skew.ax, '40%', '40%', loc='upper center')
h = Hodograph(ax_hod, component_range=90.)
h.add_grid(increment=20)
l = h.plot_colormapped(u, v, wind_speed)  # Plot a line colored by wind speed
plt.colorbar(l)
# Show the plot
plt.savefig(sl[station]+'.jpg')
plt.show()
