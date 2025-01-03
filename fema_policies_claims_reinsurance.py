
import geopandas as gpd
import numpy as np
import pandas as pd
import warnings
import cpi
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import glob2


## function to extract NSI data for each state and for the 100-year FEMA flood zone
def nsi_structures():
    # dictionary of FIPS codes for states
    fips_dict = {'01':'ALABAMA','04': 'ARIZONA','05': 'ARKANSAS','06': 'CALIFORNIA','08': 'COLORADO','09': 'CONNECTICUT','10': 'DELAWARE','11':'DISTRICT OF COLUMBIA','12': 'FLORIDA','13': 'GEORGIA','16': 'IDAHO','17': 'ILLINOIS','18': 'INDIANA','19': 'IOWA','20': 'KANSAS','21': 'KENTUCKY','22': 'LOUISIANA','23':  'MAINE','24': 'MARYLAND','25':'MASSACHUSETTS','26': 'MICHIGAN','27': 'MINNESOTA','28':'MISSISSIPPI','29':'MISSOURI','30': 'MONTANA','31':  'NEBRASKA','32':'NEVADA','33': 'NEW HAMPSHIRE','34':'NEW JERSEY','35': 'NEW MEXICO','36':'NEW YORK','37':'NORTH CAROLINA','38':'NORTH DAKOTA','39': 'OHIO','40': 'OKLAHOMA','41': 'OREGON','42': 'PENNSYLVANIA','44': 'RHODE ISLAND','45':'SOUTH CAROLINA','46':'SOUTH DAKOTA','47':'TENNESSEE','48': 'TEXAS','49':'UTAH','50':'VERMONT','51':'VIRGINIA','53':'WASHINGTON','54':'WEST VIRGINIA','55':'WISCONSIN','56':'WYOMING'}

    # list of FEMA flood zone values for the 100 year floodplain
    fld_zone_100yr = ['A', 'AE', 'A99', 'AH', 'AHB', 'AO', 'AOB', 'V', 'VE', 'AR', 'AR/AE', 'AR/AH', 'AR/AO', 'AR/A']
    for i in range(1,31):
        fld_zone_100yr.append('A'+str(i))
        fld_zone_100yr.append('V'+str(i))
        fld_zone_100yr.append('AR/A'+str(i))

    # create dataframe to save
    out_pd = pd.DataFrame([],index=list(fips_dict.values()),columns=["Num_Res_1-4units","Total_structures","Total_structures_SFHA" ,"Num_res_1-4units_SFHA","Average_Value_res_1-4units","Average_Value_res_1-4units_SFHA"])

    # loop through state dictionary
    for f in fips_dict.keys():
        print(f)
        # read in NSI dataset for each state
        nsi_data = gpd.read_file(f'./NSI_data/nsi_2022_{f}.gpkg', engine='pyogrio', use_arrow=True, columns=['occtype','sqft','val_struct','geometry'])
        nfhl_file = glob2.glob(f"./NFHL/NFHL_{f}_*")[0] # get filename of NFHL for each state
        # read in NFHL, just flood zone layer, for each state
        nfhl_data = gpd.read_file(nfhl_file, engine='pyogrio', layer='S_FLD_HAZ_AR', columns=['FLD_ZONE','geometry'])

        # filter for 100-year floodplain flood zones
        nfhl_data = nfhl_data[nfhl_data['FLD_ZONE'].isin(fld_zone_100yr)]
        nfhl_data_reproject = nfhl_data.to_crs(4326) # reproject to NSI CRS

        # list of occupancy types for residential (1-4 units) in NSI dataset
        occupancy_res_1_4 =  ['RES1-1SNB', 'RES1-1SWB', 'RES1-2SNB', 'RES1-2SWB', 'RES1-3SNB', 'RES1-3SWB', 'RES1-SLNB', 'RES1-SLWB', 'RES2', 'RES3A', 'RES3B']

        # save the total number of structures in each state
        out_pd.at[fips_dict[f],'Total_structures'] = nsi_data.shape[0]

        # clip the NSI data to the NFHL 100-year flood zones
        nsi_data_sfha = nsi_data.sjoin(nfhl_data_reproject, how='inner')
        # save the total number of structures in the NFHL 100-year flood zone for each state
        out_pd.at[fips_dict[f],'Total_structures_SFHA'] = nsi_data_sfha.shape[0]

        # filter the NSI dataset for residential (1-4 units)
        nsi_data_res_1_4 = nsi_data[nsi_data['occtype'].isin(occupancy_res_1_4)]
        # save the total number of residential (1-4 units) for each state
        out_pd.at[fips_dict[f],'Num_Res_1-4units'] = nsi_data_res_1_4.shape[0]

        # calculate average replacement cost
        out_pd.at[fips_dict[f],'Average_Value_res_1-4units'] = nsi_data_res_1_4['val_struct'].mean(skipna=True)

        # filter for residential (1-4 units) structures that are in the NFHL 100-year floodplain flood zone
        nsi_data_res_1_4_sfha = nsi_data_sfha[nsi_data_sfha['occtype'].isin(occupancy_res_1_4)]
        # save the total number of residential (1-4 units) structures that are in the NFHL 100-year floodplain flood zone
        out_pd.at[fips_dict[f],'Num_res_1-4units_SFHA'] = nsi_data_res_1_4_sfha.shape[0]

        # calculate average replacement cost just in SFHA
        out_pd.at[fips_dict[f],'Average_Value_res_1-4units_SFHA'] = nsi_data_res_1_4_sfha['val_struct'].mean(skipna=True)

    # save dataframe to file
    out_pd.to_csv('./NSI_data/NSI_residential_1-4units.csv')

    return

## function to read large CSV file
def read_csv(file_name, fields):
    for chunk in pd.read_csv(file_name, chunksize=1000000, usecols=fields, engine="c"):
        yield chunk

## function to calculate NFIP coverage from
def total_coverage():
    # list of state abbreviations for CONUS
    states = ['AL','AR','AZ','CA','CO','CT','DC','DE','FL','GA','IA','ID','IL','IN','KS','KY','LA','MA','MD','ME','MI','MN','MO','MS','MT','NC','ND','NE','NH','NJ','NM','NV','NY','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VA','VT','WA','WI','WV','WY']
    # list of fields to extract from CSV
    fields = ['propertyState', 'policyEffectiveDate', 'policyTerminationDate', 'totalBuildingInsuranceCoverage', 'totalContentsInsuranceCoverage','floodZoneCurrent','occupancyType','buildingReplacementCost']

    occupancy_res = [1, 2, 3, 11, 12, 13, 14, 15, 16] # residential occupancy codes
    occupancy_res1_4 = [1, 2, 11, 12, 14] # residential (1-4 units) occupancy codes

    years = range(2009,2024) # list of years with data
    # create the dataframe to save output
    coverage_pd_out = pd.DataFrame([],index=years,columns=['Total_coverage','Res_1-4_coverage','Res_1-4_SFHAcoverage'])

    # loop through years
    for y in years:
        # create empty list to hold dataframes
        fl_dataframes = []
        # loop through chunks of dataframe to extract data
        for df in read_csv('FimaNfipPolicies.csv', fields=fields):
            # filter for policies that are in CONUS
            df = df[df['propertyState'].isin(states)]
            # extract year for each policy
            df['year_date'] = df['policyEffectiveDate'].str.split('-').str[0]
            # filter out bad years
            df = df[df['year_date'].str.len() == 4]
            df = df[(df['year_date'].astype(int) <= 2100) & (df['year_date'].astype(int) > 1968)]

            # extract data for specific year
            df = df[df['year_date'] == str(y)]

            # append dataframe to list
            fl_dataframes.append(df)

        # merge dataframes
        data = pd.concat(fl_dataframes)

        # calculate contents and building total coverage
        contents_coverage = data['totalContentsInsuranceCoverage'].sum()
        building_coverage = data['totalBuildingInsuranceCoverage'].sum()
        coverage_pd_out.at[y,'Total_coverage'] = contents_coverage+building_coverage

        # filter for residential (1-4 units) policies
        data = data[data['occupancyType'].isin(occupancy_res1_4)]

        # calculate contents and building total coverage for residential (1-4 units) policies
        contents_coverage = data['totalContentsInsuranceCoverage'].sum()
        building_coverage = data['totalBuildingInsuranceCoverage'].sum()
        print(y, contents_coverage+building_coverage)
        coverage_pd_out.at[y,'Res_1-4_coverage'] = contents_coverage+building_coverage

    # save dataframe to file
    coverage_pd_out.to_csv('FEMA_total_ResSFHA_coverage_2009_2023.csv')

    return

## function to calculate NFIP coverage for residential properties
def residential_coverage_cat_models():
    # list of state abbreviations for CONUS
    states = ['AL','AR','AZ','CA','CO','CT','DC','DE','FL','GA','IA','ID','IL','IN','KS','KY','LA','MA','MD','ME','MI','MN','MO','MS','MT','NC','ND','NE','NH','NJ','NM','NV','NY','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VA','VT','WA','WI','WV','WY']
    # list of fields to extract from CSV
    fields = ['propertyState', 'policyEffectiveDate', 'totalBuildingInsuranceCoverage', 'totalContentsInsuranceCoverage','occupancyType','buildingReplacementCost','floodZoneCurrent','policyTerminationDate']

    # create the dataframe to save output
    pd_out = pd.DataFrame([],index=states, columns=['Locations','Coverage','Building_Value','SFHA_Coverage','SFHA_Locations'])
    # date of reinsurnace catastrophe model data publication
    doi = '2022-05-31'

    occupancy_res1_4 = [1, 2, 11, 12, 14] # residential (1-4 units) occupancy codes

    # create empty list to hold dataframes
    pd_dataframes = []
    # loop through chunks of dataframe to extract data
    for df in read_csv('FimaNfipPolicies.csv', fields=fields):
        # clean policy effective year and filter out bad values
        df['year_date'] = df['policyEffectiveDate'].str.split('-').str[0]
        df = df[df['year_date'].str.len() == 4]
        df = df[(df['year_date'].astype(int) <= 2100) & (df['year_date'].astype(int) > 1968)]

        # clean policy termination year and filter out bad values
        df['year_date'] = df['policyTerminationDate'].str.split('-').str[0]
        df = df[df['year_date'].str.len() == 4]
        df = df[(df['year_date'].astype(int) <= 2100) & (df['year_date'].astype(int) > 1968)]

        # convert fields to datetime
        df['policyEffectiveDate'] = pd.to_datetime(df['policyEffectiveDate'])
        df['policyTerminationDate'] = pd.to_datetime(df['policyTerminationDate'])

        # filter for in-force policies for date of interest
        df = df[(df['policyEffectiveDate']  <= doi) & (df['policyTerminationDate'] > doi)]
        # filter for CONUS states
        df = df[df['propertyState'].isin(states)]
        # append dataframe to list
        pd_dataframes.append(df)

    # merge dataframes
    data_doi = pd.concat(pd_dataframes)

    # list of FEMA flood zone values for the 100 year floodplain
    fld_zone_100yr = ['A', 'AE', 'A99', 'AH', 'AHB', 'AO', 'AOB', 'V', 'VE', 'AR', 'AR/AE', 'AR/AH', 'AR/AO', 'AR/A']
    for i in range(1,31):
        fld_zone_100yr.append('A'+str(i))
        fld_zone_100yr.append('V'+str(i))
        fld_zone_100yr.append('AR/A'+str(i))

    print(data_doi['totalBuildingInsuranceCoverage'].sum() + data_doi['totalContentsInsuranceCoverage'].sum())
    print(data_doi.shape[0])
    # fileter for residential (1-4 units) policies
    data_res = data_doi[data_doi['occupancyType'].isin(occupancy_res1_4)]

    # loop through CONUS states
    for s in states:
        # filter for specific state
        state_data = data_res[data_res['propertyState']==s]
        # calculate building and contents coverage for residential (1-4 units)
        building_coverage_res = state_data['totalBuildingInsuranceCoverage'].sum() + state_data['totalContentsInsuranceCoverage'].sum()
        # calculate building replacement cost for residential (1-4 units)
        building_value = state_data['buildingReplacementCost'].sum()
        # save total residential (1-4 units) locations
        pd_out.at[s,'Locations'] = state_data.shape[0]
        # save total residential (1-4 units) coverage
        pd_out.at[s,'Coverage'] = building_coverage_res
        # save total residential (1-4 units) replacement value
        pd_out.at[s,'Building_Value'] = building_value

        # filter for residential (1-4 units) in FEMA 100-year flood zones
        state_data_sfha = state_data[state_data['floodZoneCurrent'].isin(fld_zone_100yr)]
        # calculate building and contents coverage for residential (1-4 units) in 100-year flood zone
        building_coverage_res = state_data_sfha['totalBuildingInsuranceCoverage'].sum() + state_data_sfha['totalContentsInsuranceCoverage'].sum()
        # save total residential (1-4 units) in 100-year flood zone coverage
        pd_out.at[s,'SFHA_Coverage'] = building_coverage_res
        # save total residential (1-4 units) in 100-year flood zone locations
        pd_out.at[s,'SFHA_Locations'] = state_data_sfha.shape[0]

    # save dataframe to file
    pd_out.to_csv('NFIP_state_residential.csv')

    return

## function to compare burn rates between First Street and reinsurance catastrophe models
def burn_rate_comparison():
    first_street = pd.read_csv('./FirstStreet_check/FirstStreet_AAL.csv') # load First Street AAL and locations data
    nfip_coverage = pd.read_csv('NFIP_state_residential.csv') # load NFIP residential (1-4 units) coverage and locations data
    nsi_data = pd.read_csv('./NSI_data/NSI_residential_1-4units.csv') # load NSI data
    # load catastrophe models AAL data
    catmodels = pd.read_csv('./fema_nfip-reinsurance-placement-information_2023/FEMA_reinsurance_2023_AAL.csv',index_col='State')
    # apply scale AAL and Limit values
    catmodels['Katrisk_Storm Surge_AAL'] = catmodels['Katrisk_Storm Surge_AAL'] * 1000
    catmodels['Katrisk_Inland_AAL'] = catmodels['Katrisk_Inland_AAL'] * 1000
    catmodels['AIR_Storm Surge_AAL'] = catmodels['AIR_Storm Surge_AAL'] * 1000
    catmodels['AIR_Inland_AAL'] = catmodels['AIR_Inland_AAL'] * 1000
    catmodels['RMS_Storm_Surge_AAL'] = catmodels['RMS_Storm_Surge_AAL'] * 1000

    # calculate mean of storm surge AAL across catastrophe models for each state
    catmodels['StormSurge_AAL_mean'] = catmodels[['Katrisk_Storm Surge_AAL', 'AIR_Storm Surge_AAL', 'RMS_Storm_Surge_AAL']].mean(axis='columns')
    # calculate mean of inland AAL across catastrophe models
    catmodels['Inland_AAL_mean'] = catmodels[['AIR_Inland_AAL', 'Katrisk_Inland_AAL']].mean(axis='columns',skipna=True)

    # set index to each state in NFIP coverage dataframe
    nfip_coverage.set_index('Unnamed: 0', inplace=True)
    # loop through states
    for index, row in catmodels.iterrows():
        # limit = catmodels.at[index,'Limit_mean']
        # for each state set coverage to the NFIP coverage for residential (1-4 units) in FEMA 100-year flood zone
        limit = nfip_coverage.at[index,'SFHA_Coverage']
        catmodels.at[index,'SFHA_Coverage'] = limit
        # for each state calculate catastrophe model burn rate: add mean storm surge and inland AAL then divide by $1,000 of coverage
        catmodels.at[index,'AAL_per_1000_limit'] = (catmodels.at[index,'StormSurge_AAL_mean'] + catmodels.at[index,'Inland_AAL_mean']) / (limit / 1000)

    # create state column from index
    catmodels['State'] = catmodels.index

    # calculate CONUS storm surge AAL for each cat model and then take mean
    cat_models_ss_aal = np.mean([catmodels['Katrisk_Storm Surge_AAL'].sum(), catmodels['AIR_Storm Surge_AAL'].sum(), catmodels['RMS_Storm_Surge_AAL'].sum()])
    # calculate CONUS inland AAL for each cat model and then take mean
    cat_models_inland_aal = np.mean([catmodels['Katrisk_Inland_AAL'].sum(), catmodels['AIR_Inland_AAL'].sum()])
    # calculate CONUS burn rate using the mean storm surge and inland AAL and CONUS NFIP coverage for residential (1-4 units) in FEMA 100-year flood zone per $1,000 of coverage
    cat_models_burn_rate_mean = (cat_models_ss_aal + cat_models_inland_aal) / (nfip_coverage['SFHA_Coverage'].sum() / 1000)
    # calculate KatRisk burn rate
    katrisk_burn_rate = (catmodels['Katrisk_Storm Surge_AAL'].sum() + catmodels['Katrisk_Inland_AAL'].sum()) / (nfip_coverage['SFHA_Coverage'].sum() / 1000)
    # calculate Verisk burn rate
    verisk_burn_rate = (catmodels['AIR_Storm Surge_AAL'].sum() + catmodels['AIR_Inland_AAL'].sum()) / (nfip_coverage['SFHA_Coverage'].sum() / 1000)
    # print cat model burn rates for CONUS
    print(f"Cat Models Average Burn Rate: {cat_models_burn_rate_mean}; Katrisk Burn Rate: {katrisk_burn_rate}; AIR Burn Rate: {verisk_burn_rate}")
    print(f"Cat Models Average AAL: {cat_models_ss_aal + cat_models_inland_aal}; Katrisk AAL: {catmodels['Katrisk_Storm Surge_AAL'].sum() + catmodels['Katrisk_Inland_AAL'].sum()}; AIR AAL: {catmodels['AIR_Storm Surge_AAL'].sum() + catmodels['AIR_Inland_AAL'].sum()}")
    print(f"Cat Models Average coverage: {nfip_coverage['SFHA_Coverage'].sum()}; Katrisk coverage: {nfip_coverage['SFHA_Coverage'].sum()}; AIR coverage: {nfip_coverage['SFHA_Coverage'].sum()}")

    # use Ginto Normal font
    font_path = '/Users/ddusseau/Documents/Fonts/GintoNormal/GintoNormal-Regular.ttf'  # the location of the font file
    my_font = fm.FontProperties(fname=font_path, size=9)  # get the font based on the font_path
    # set DPI parameter
    plt.rcParams['savefig.dpi'] = 300
    # creating an empty chart and set size
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111)

    # convert state name in First Street data to all caps
    first_street['State_Name_Upper'] = first_street['State_Name'].str.upper()
    # set NSI index as the state name
    nsi_data.set_index('Unnamed: 0', inplace=True)

    fs_aal_total = 0 # initialize total First Street AAL
    fs_coverage_total = 0 # initialize total First Street coverage
    # loop through First Street data by state
    for index, row in first_street.iterrows():
        # calculate First Street coverage for a specific state: multiply the number of residential (1-4 units) structures in FEMA 100-year flood zone by the max coverage amount of $250,000
        fs_coverage = nsi_data.at[row['State_Name_Upper'],"Num_res_1-4units_SFHA"] * 250000

        # calculate burn rate using AAL from First Street 100-year floodplain and FEMA 100-year floodplain
        aal_fs = row['Inside SFHA AAL'] / (fs_coverage / 1000)
        # add state coverage to CONUS total coverage
        fs_coverage_total = fs_coverage_total+ fs_coverage
        # add state AAL to CONUS total AAL
        fs_aal_total = fs_aal_total + row['Inside SFHA AAL']

        catmodels.fillna(0, inplace=True)
        # extract cat model mean burn rate
        kat_burn = (catmodels.at[row['State'],'Katrisk_Storm Surge_AAL'] + catmodels.at[row['State'],'Katrisk_Inland_AAL']) / (catmodels.at[row['State'],'SFHA_Coverage'] / 1000)
        verisk_burn = (catmodels.at[row['State'],'AIR_Storm Surge_AAL'] + catmodels.at[row['State'],'AIR_Inland_AAL']) / (catmodels.at[row['State'],'SFHA_Coverage'] / 1000)

        # aal_cat = catmodels.at[row['State'],'AAL_per_1000_limit']

        # append First Street and cat model mean burn rate to output list
        # aal_fs_cat.append([row['State'],aal_fs,aal_cat])

        # print(row['State'],aal_fs,aal_cat)
        # plot burn rates in stem plot and assign color depending on which burn rate is greater
        ax.vlines(row['State'], 0, np.max([aal_fs, kat_burn, verisk_burn]), linestyles='dashed', color='black', linewidths=0.5) #
        ax.scatter(row['State'], aal_fs, label='First Street', color='#fa7921')
        ax.scatter(row['State'], kat_burn, color='#0c4767', label='KatRisk')
        ax.scatter(row['State'], verisk_burn, color='#b9a44c', label='Verisk')

    # convert list of burn rates into dataframe
    # aal_fs_cat_pd = pd.DataFrame(aal_fs_cat,columns=['State','FSF_burn_rate','CatModel_mean_burn_rate'])
    # aal_fs_cat_pd.to_csv('FSF_CatModel_BurnRate.csv',index=False)

    plt.tick_params(left = False)
    plt.box(False)
    plt.grid(visible=True,axis='y')
    plt.yticks(fontproperties=my_font, fontsize=13)
    plt.xticks(fontproperties=my_font, fontsize=10)
    plt.text(-2.5, 50, s="Catastrophe Models Burn Rate Comparison", fontproperties=my_font, fontsize=16)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.ylabel("Burn rate ($)" ,fontproperties=my_font, fontsize=13)

    # plt.show()

    plt.savefig("./figures/FSF_minus_CatModels_BurnRate_Residential.png",bbox_inches='tight')

    # print CONUS First Street burn rate
    print(f"First Street 100-year burn rate: ${fs_aal_total / (fs_coverage_total / 1000)}")
    print(f"First Street AAL total: {fs_aal_total}. First Street coverage total: {fs_coverage_total}")

    return


## function to compare reinsurance cat model AAL to NFIP claims data
def aal_validation():
    # load in NFIP claims data
    data = pd.read_csv('FimaNfipClaims.csv', engine='c')

    # get unique years from NFIP claims data and sort them
    years = np.sort(pd.unique(data['yearOfLoss']))

    # use Ginto Normal font
    font_path = '/Users/ddusseau/Documents/Fonts/GintoNormal/GintoNormal-Regular.ttf'  # the location of the font file
    my_font = fm.FontProperties(fname=font_path, size=9)  # get the font based on the font_path
    plt.rcParams['savefig.dpi'] = 300

################################
## NFIP: Average recorded yearly damages as a proportion of the yearly total value of NFIP policies.
## Cat Models: Average annual relative damage according to reinsurance placement data
    nfip_policies_data = pd.read_csv('./NFIP_total_exposure_policies.csv')
    nfip_policies_data.set_index('Year', inplace=True)

    total_coverage = 0
    total_paid = 0
    moving = []
    for y in years:
        if y < 1978 or y > 2023: #1992
            continue
        data_year = data[data['yearOfLoss'] == y]
        data_year['total_paid'] = data_year['netBuildingPaymentAmount'] + data_year['netContentsPaymentAmount']

        total_paid = total_paid + data_year['total_paid'].sum()
        total_coverage = total_coverage + nfip_policies_data.at[y,'Exposure']
        moving.append(total_paid / (total_coverage / 1000))


    plt.plot(np.arange(1978,2023+1),moving)
    plt.title("NFIP Moving Burn Rate",fontproperties=my_font, fontsize=15)
    plt.ylabel("Burn Rate ($)" ,fontproperties=my_font, fontsize=13)
    plt.xlabel("Years Used in Analysis" ,fontproperties=my_font, fontsize=13)
    plt.yticks(fontproperties=my_font, fontsize=12)
    plt.xticks(fontproperties=my_font, fontsize=12)
    plt.ylim(0, 6)
    plt.savefig("./figures/NFIP_moving_burn_rate.png",bbox_inches='tight')
    # plt.show()

    print(f"Burn rate of NFIP ${total_paid / (total_coverage / 1000)}")
    print(f"Total claims ${total_paid} and total coverage ${total_coverage}")

    years_flip = []
    total_coverage = 0
    total_paid = 0
    moving_burn_rate = []
    for y in np.flip(years):
        if y < 1978 or y > 2023:
            continue
        data_year = data[data['yearOfLoss'] == y]
        data_year['total_paid'] = data_year['netBuildingPaymentAmount'] + data_year['netContentsPaymentAmount']

        total_paid = total_paid + data_year['total_paid'].sum()
        total_coverage = total_coverage + nfip_policies_data.at[y,'Exposure']
        moving_burn_rate.append(total_paid/(total_coverage / 1000))
        years_flip.append(y)

    plt.plot(years_flip,moving_burn_rate)
    plt.title("NFIP Moving Burn Rate - Backwards in Time",fontproperties=my_font, fontsize=15)
    plt.ylabel("Burn Rate ($)" ,fontproperties=my_font, fontsize=13)
    plt.xlabel("Years Used in Analysis" ,fontproperties=my_font, fontsize=13)
    plt.gca().invert_xaxis()
    plt.yticks(fontproperties=my_font, fontsize=12)
    plt.xticks(fontproperties=my_font, fontsize=12)
    plt.ylim(0, 6)
    plt.savefig("./figures/NFIP_moving_burn_rate_backwards.png",bbox_inches='tight')
    # plt.show()


################################
## Comparison graph of average replacement value between First Street and NSI

    fs_aal_data = pd.read_csv('./FirstStreet_check/FirstStreet_AAL.csv')
    # convert state name in First Street data to all caps
    fs_aal_data['State_Name_Upper'] = fs_aal_data['State_Name'].str.upper()

    nsi_data = pd.read_csv('./NSI_data/NSI_residential_1-4units.csv')

    nsi_data.set_index('Unnamed: 0', inplace=True)

    # creating an empty chart and set size
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111)

    for index, row in fs_aal_data.iterrows():
        fs_value = row['Avg_structure_value']
        nsi_value = nsi_data.at[row['State_Name_Upper'],'Average_Value_res_1-4units']

        if nsi_value / fs_value > 1:
            ax.stem(row['State'], nsi_value / fs_value, linefmt='#233E99', markerfmt='#233E99', basefmt=' ', label= 'NSI > First Street')
        if nsi_value / fs_value < 1:
            ax.stem(row['State'], nsi_value / fs_value, linefmt='#FF5700', markerfmt='#FF5700', basefmt=' ', label= 'NSI < First Street')

    plt.tick_params(left = False)
    plt.box(False)
    plt.grid(visible=True,axis='y')
    plt.yticks(fontproperties=my_font, fontsize=13)
    plt.xticks(fontproperties=my_font, fontsize=10)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.ylabel("NSI / First Street (ratio)" ,fontproperties=my_font, fontsize=13)

    # plt.show()
    plt.savefig("./figures/NSI_divided_FSF_Average_Structure_Value.png",bbox_inches='tight')


################################
## NFIP: Relative average number of flooded buildings in the NFIP.
    # load in NFIP claims data
    data = pd.read_csv('FimaNfipClaims.csv', engine='c')
    nfip_policies_data = pd.read_csv('./NFIP_total_exposure_policies.csv')
    flood_denied = [1, 15] # denied due to deductible or damage before start of policy

    data_flood_denied = data[data['nonPaymentReasonBuilding'].isin(flood_denied)] # denied flooded buildings
    data_notdenied = data[data['nonPaymentReasonBuilding'].isna()] # not denied flooded buildings
    total_flooded = data_notdenied.shape[0] + data_flood_denied.shape[0] # total flooded buildings
    total_policies = nfip_policies_data['Policies'].sum() # total policies
    print(f"Relative average number of flooded buildings in NFIP: {(total_flooded / total_policies) * 100}%")

    return




########################################


warnings.filterwarnings('ignore')
# cpi.update()

# nsi_structures()

# total_coverage()

# residential_coverage_cat_models()

burn_rate_comparison()

# aal_validation()
