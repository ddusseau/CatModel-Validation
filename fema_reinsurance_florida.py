
import geopandas as gpd
import numpy as np
import pandas as pd
import warnings
import cpi
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import glob2



## function to get burn rate for Florida
def burn_rate_comparison():
    # load catastrophe models AAL data
    catmodels = pd.read_csv('./fema_nfip-reinsurance-placement-information_2023/FEMA_reinsurance_2023_AAL.csv',index_col='State')
    # apply scale AAL and Limit values
    catmodels['Katrisk_Storm Surge_AAL'] = catmodels['Katrisk_Storm Surge_AAL'] * 1000
    catmodels['Katrisk_Inland_AAL'] = catmodels['Katrisk_Inland_AAL'] * 1000
    catmodels['Katrisk_Limit'] = catmodels['Katrisk_Limit'] * 1000
    catmodels['AIR_Storm Surge_AAL'] = catmodels['AIR_Storm Surge_AAL'] * 1000
    catmodels['AIR_Inland_AAL'] = catmodels['AIR_Inland_AAL'] * 1000
    catmodels['AIR_Limit'] = catmodels['AIR_Limit'] * 1000
    catmodels['RMS_Storm_Surge_AAL'] = catmodels['RMS_Storm_Surge_AAL'] * 1000
    catmodels['RMS_Limit'] = catmodels['RMS_Limit'] * 1000

    catmodels = catmodels[catmodels.index == 'FL']
    catmodels['Katrisk_burnRate'] = (catmodels['Katrisk_Storm Surge_AAL'] + catmodels['Katrisk_Inland_AAL']) / catmodels['Katrisk_Limit'] * 1000
    catmodels['AIR_burnRate'] = (catmodels['AIR_Storm Surge_AAL'] + catmodels['AIR_Inland_AAL']) / catmodels['AIR_Limit'] * 1000

    print(catmodels['AIR_Inland_AAL'] + catmodels['AIR_Storm Surge_AAL'])
    print(catmodels['Katrisk_Storm Surge_AAL'] + catmodels['Katrisk_Inland_AAL'])

    return



########################################


warnings.filterwarnings('ignore')

burn_rate_comparison()
