# This is a script by Kees Nederhoff

# Run this in 'base'

# Libraries needed
import os
import datetime
import numpy as np
import cdsapi
from pathlib import Path

#==============================================================================
# 1. Settings
#==============================================================================
# Time period
years_wanted            = list(range(1950, 2024))  # Note that 2024 is not included, the range stops at 2023

# optionally restrict area to Europe (in N/W/S/E).
# Wadden Sea area:      "50/2/!56/8"
# Hawaii:               "190/14/215/31"
# West Africa:          "12/341/18/344"
# San Fransisco         "36/235/40/240"

# Define working directory
cwd = os.getcwd()
download_path = Path("C:/Users/bruij_kn/Downloads")
os.chdir(download_path)


#==============================================================================
# 2. Loop
#==============================================================================
for years in years_wanted:

    # A. Get new time period
    print(years)
    starttime           = datetime.date(int(years), 1,1)
    variables_wanted    = [
        '10m_u_component_of_wind',
        '10m_v_component_of_wind', 
        '2m_temperature', 
        'sea_ice_cover', 
        'sea_surface_temperature', 
        'mean_surface_net_short_wave_radiation_flux', 
        'mean_surface_net_long_wave_radiation_flux', 
        'mean_surface_latent_heat_flux', 
        'sea_surface_temperature', 
        'snow_depth', 
        'soil_temperature_level_1', 
        'soil_temperature_level_2', 
        'soil_temperature_level_3', 
        'soil_temperature_level_4'
        ]

    # B. Loop over variables we wanted
    for variables_now in variables_wanted:
        
        # Define variable wanted
        targetname             = variables_now + '_' + str(int(years)) + '.nc'

        # C. Contact API
        c = cdsapi.Client()
        c.retrieve('reanalysis-era5-single-levels', {
                'variable'      : variables_now,
                'product_type'  : 'reanalysis',
                'year'          : starttime.strftime("%Y"),
                'month'         : ['01','02','03','04','05','06', '07','08','09','10','11','12'],
                'day'           : ['01','02','03','04','05','06', '07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31'],
                'area'          : [72, -145, 69, -140],      # North, West, South, East. Default: global
                'time'          : ['00:00','01:00','02:00','03:00','04:00','05:00','06:00','07:00','08:00','09:00','10:00','11:00','12:00','13:00','14:00','15:00','16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00'],
                'format'        : 'netcdf'         
            }, targetname)

