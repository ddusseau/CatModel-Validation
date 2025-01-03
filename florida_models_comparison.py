import matplotlib.pyplot as plt
import tabula
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

## function to convert pfds into csv
def pdf_to_csv():

    new_cols = ['RatingArea', 'GeographicZone', 'ZIPCode', 'CountyName', 'FIPSCode', 'FrameOwners', 'MasonryOwners', 'ManufacturedHomes']

    # Read PDF into a list of DataFrames
    dfs = tabula.read_pdf("if2021formaf1_09202024.pdf", pages='all')
    df1 = dfs[0].iloc[2:,:6]
    df1[['FrameOwners', 'MasonryOwners', 'ManufacturedHomes']] = df1['Standard Flood Lost Cost per Policy Type'].str.split(' ', expand=True)
    df1.drop(columns='Standard Flood Lost Cost per Policy Type', inplace=True)
    df1.columns = new_cols

    dfs = [df.iloc[:,:8] for df in dfs[1:]]
    dfs = [df.reset_index().T.reset_index().T.iloc[:,1:] for df in dfs]

    new_dfs = [df1]
    for d in dfs:
        d.columns = new_cols
        new_dfs.append(d)

    # Convert the list of DataFrames to a single DataFrame
    aon = pd.concat(new_dfs)
    aon.to_csv('AON_2024.csv', index=False)


    # Read PDF into a list of DataFrames
    new_cols = ['RatingArea', 'ZIPCode', 'CountyName', 'FIPSCode', 'FrameOwners', 'MasonryOwners', 'ManufacturedHomes']
    dfs = tabula.read_pdf("kcc21_formaf1_20240130.pdf", pages='all')
    df1 = dfs[0].iloc[1:,:5]
    df1[['FrameOwners', 'MasonryOwners', 'ManufacturedHomes']] = df1['Standard Flood Loss Cost per Policy Type'].str.split(' ', expand=True)
    df1.drop(columns='Standard Flood Loss Cost per Policy Type', inplace=True)
    df1.columns = new_cols

    # select rows and columns
    dfs = [df.iloc[1:,:5] for df in dfs[1:]]

    new_dfs = [df1]
    for d in dfs:
        d[['FrameOwners', 'MasonryOwners', 'ManufacturedHomes']] = d['Standard Flood Loss Cost per Policy Type'].str.split(' ', expand=True)
        d.drop(columns='Standard Flood Loss Cost per Policy Type', inplace=True)
        d.columns = new_cols
        new_dfs.append(d)

    # Convert the list of DataFrames to a single DataFrame
    kcc = pd.concat(new_dfs)
    kcc.to_csv('KCC_2024.csv', index=False)



    # Read PDF into a list of DataFrames
    new_cols = ['RatingArea', 'GeographicZone', 'ZIPCode', 'CountyName', 'FIPSCode', 'FrameOwners', 'MasonryOwners', 'ManufacturedHomes']
    dfs = tabula.read_pdf("fiu21formaf1.pdf", pages='all')
    df1 = dfs[0].iloc[1:,:6]
    df1[['FrameOwners', 'MasonryOwners', 'ManufacturedHomes']] = df1['Standard Flood Loss Cost per Policy Type'].str.split(' ', expand=True)
    df1.drop(columns='Standard Flood Loss Cost per Policy Type', inplace=True)
    df1.columns = new_cols

    # select rows and columns
    dfs = [df.iloc[1:,:8] for df in dfs[1:]]

    new_dfs = [df1]
    for d in dfs:
        d.columns = new_cols
        new_dfs.append(d)

    # Convert the list of DataFrames to a single DataFrame
    fiu = pd.concat(new_dfs)
    fiu.to_csv('FIU_2024.csv', index=False)

    return

## function to compare loss rates by zip code
def compare_loss():
    aon = pd.read_csv('AON_2024.csv')
    kcc = pd.read_csv('KCC_2024.csv')
    fiu = pd.read_csv('FIU_2024.csv')

    for index, row in aon.iterrows():
        fips = str(row['FIPSCode'])
        if len(fips) == 1:
            aon.at[index,'FIPSCode'] = '1200'+fips
        if len(fips) == 2:
            aon.at[index,'FIPSCode'] = '120'+fips
        if len(fips) == 3:
            aon.at[index,'FIPSCode'] = '12'+fips
    aon['ZIPCode'] = aon['ZIPCode'].astype(int)

    fiu['FIPSCode'] = fiu['FIPSCode'].astype(int)
    for index, row in fiu.iterrows():
        fips = str(row['FIPSCode'])
        if len(fips) == 1:
            fiu.at[index,'FIPSCode'] = '1200'+fips
        if len(fips) == 2:
            fiu.at[index,'FIPSCode'] = '120'+fips
        if len(fips) == 3:
            fiu.at[index,'FIPSCode'] = '12'+fips
    fiu['ZIPCode'] = fiu['ZIPCode'].astype(int)

    aon['CountyZIP'] = aon['FIPSCode'].astype(str)+'_'+aon['ZIPCode'].astype(str)
    kcc['CountyZIP'] = kcc['FIPSCode'].astype(str)+'_'+kcc['ZIPCode'].astype(str)
    fiu['CountyZIP'] = fiu['FIPSCode'].astype(str)+'_'+fiu['ZIPCode'].astype(str)

    frame = fiu.groupby(['CountyZIP'])['FrameOwners'].mean()
    masonry = fiu.groupby(['CountyZIP'])['MasonryOwners'].mean()
    manufactured = fiu.groupby(['CountyZIP'])['ManufacturedHomes'].mean()
    fiu = pd.concat([frame,masonry,manufactured], axis=1)
    fiu.reset_index(inplace=True)

    aon_kcc = pd.merge(aon, kcc, on='CountyZIP')
    aon_kcc_fiu = pd.merge(aon_kcc, fiu, on='CountyZIP')

    aon_kcc_fiu = aon_kcc_fiu[['CountyZIP', 'ZIPCode_x', 'CountyName_x', 'FIPSCode_x', 'FrameOwners', 'MasonryOwners', 'ManufacturedHomes', 'FrameOwners_x', 'MasonryOwners_x', 'ManufacturedHomes_x', 'FrameOwners_y', 'MasonryOwners_y', 'ManufacturedHomes_y']]
    aon_kcc_fiu.rename(columns={'ZIPCode_x':'ZIPCode', 'CountyName_x':'CountyName', 'FIPSCode_x':'FIPSCode', 'FrameOwners':'FIU_FrameOwners', 'MasonryOwners':'FIU_MasonryOwners', 'ManufacturedHomes':'FIU_ManufacturedHomes', 'FrameOwners_x':'AON_FrameOwners', 'MasonryOwners_x':'AON_MasonryOwners', 'ManufacturedHomes_x':'AON_ManufacturedHomes', 'FrameOwners_y':'KCC_FrameOwners', 'MasonryOwners_y':'KCC_MasonryOwners', 'ManufacturedHomes_y':'KCC_ManufacturedHomes'}, inplace=True)

    frames = aon_kcc_fiu[['FIU_FrameOwners', 'KCC_FrameOwners', 'AON_FrameOwners']]
    manufacture = aon_kcc_fiu[['FIU_ManufacturedHomes', 'KCC_ManufacturedHomes', 'AON_ManufacturedHomes']]
    masonry = aon_kcc_fiu[['FIU_MasonryOwners', 'KCC_MasonryOwners', 'AON_MasonryOwners']]
    fig, ax = plt.subplots(3, 3, figsize=(10, 10))
    fig.subplots_adjust(wspace=0.6, hspace=0.6)

    plt.text(-155, 430, 'Frame Structures', fontsize=14)
    plt.text(-160, 270, 'Masonry Structures', fontsize=14)
    plt.text(-180, 110, 'Manufactured Structures', fontsize=14)

    ax[0,0].scatter(frames['KCC_FrameOwners'],frames['AON_FrameOwners'])
    ax[0,0].set_xlim([0,100])
    ax[0,0].set_ylim([0,100])
    ax[0,0].set_xticks(np.arange(0, 125, 25))
    ax[0,0].set_yticks(np.arange(0, 125, 25))
    ax[0,0].set_xlabel('Karen Clark & Company Loss Rate')
    ax[0,0].set_ylabel('AON Loss Rate')
    correlation_coefficient, p_value = pearsonr(frames['KCC_FrameOwners'], frames['AON_FrameOwners'])
    ax[0,0].text(60, 80, f"r\u00b2: {correlation_coefficient ** 2:.2f}")

    ax[0,1].scatter(frames['KCC_FrameOwners'],frames['FIU_FrameOwners'])
    ax[0,1].set_xlim([0,100])
    ax[0,1].set_ylim([0,100])
    ax[0,1].set_xticks(np.arange(0, 125, 25))
    ax[0,1].set_yticks(np.arange(0, 125, 25))
    ax[0,1].set_xlabel('Karen Clark & Company Loss Rate')
    ax[0,1].set_ylabel('FPFLM Loss Rate')
    correlation_coefficient, p_value = pearsonr(frames['KCC_FrameOwners'],frames['FIU_FrameOwners'])
    ax[0,1].text(60, 80, f"r\u00b2: {correlation_coefficient ** 2:.2f}")

    ax[0,2].scatter(frames['AON_FrameOwners'],frames['FIU_FrameOwners'])
    ax[0,2].set_xlim([0,100])
    ax[0,2].set_ylim([0,100])
    ax[0,2].set_xticks(np.arange(0, 125, 25))
    ax[0,2].set_yticks(np.arange(0, 125, 25))
    ax[0,2].set_xlabel('AON Loss Rate')
    ax[0,2].set_ylabel('FPFLM Loss Rate')
    correlation_coefficient, p_value = pearsonr(frames['AON_FrameOwners'],frames['FIU_FrameOwners'])
    ax[0,2].text(60, 80, f"r\u00b2: {correlation_coefficient ** 2:.2f}")

    ax[1,0].scatter(masonry['KCC_MasonryOwners'],masonry['AON_MasonryOwners'])
    ax[1,0].set_xlim([0,100])
    ax[1,0].set_ylim([0,100])
    ax[1,0].set_xticks(np.arange(0, 125, 25))
    ax[1,0].set_yticks(np.arange(0, 125, 25))
    ax[1,0].set_xlabel('Karen Clark & Company Loss Rate')
    ax[1,0].set_ylabel('AON Loss Rate')
    correlation_coefficient, p_value = pearsonr(masonry['KCC_MasonryOwners'],masonry['AON_MasonryOwners'])
    ax[1,0].text(60, 80, f"r\u00b2: {correlation_coefficient ** 2:.2f}")

    ax[1,1].scatter(masonry['KCC_MasonryOwners'],masonry['FIU_MasonryOwners'])
    ax[1,1].set_xlim([0,100])
    ax[1,1].set_ylim([0,100])
    ax[1,1].set_xticks(np.arange(0, 125, 25))
    ax[1,1].set_yticks(np.arange(0, 125, 25))
    ax[1,1].set_xlabel('Karen Clark & Company Loss Rate')
    ax[1,1].set_ylabel('FPFLM Loss Rate')
    correlation_coefficient, p_value = pearsonr(masonry['KCC_MasonryOwners'],masonry['FIU_MasonryOwners'])
    ax[1,1].text(60, 80, f"r\u00b2: {correlation_coefficient ** 2:.2f}")

    ax[1,2].scatter(masonry['AON_MasonryOwners'],masonry['FIU_MasonryOwners'])
    ax[1,2].set_xlim([0,100])
    ax[1,2].set_ylim([0,100])
    ax[1,2].set_xticks(np.arange(0, 125, 25))
    ax[1,2].set_yticks(np.arange(0, 125, 25))
    ax[1,2].set_xlabel('AON Loss Rate')
    ax[1,2].set_ylabel('FPFLM Loss Rate')
    correlation_coefficient, p_value = pearsonr(masonry['AON_MasonryOwners'],masonry['FIU_MasonryOwners'])
    ax[1,2].text(60, 80, f"r\u00b2: {correlation_coefficient ** 2:.2f}")

    ax[2,0].scatter(manufacture['KCC_ManufacturedHomes'],manufacture['AON_ManufacturedHomes'])
    ax[2,0].set_xlim([0,100])
    ax[2,0].set_ylim([0,100])
    ax[2,0].set_xticks(np.arange(0, 125, 25))
    ax[2,0].set_yticks(np.arange(0, 125, 25))
    ax[2,0].set_xlabel('Karen Clark & Company Loss Rate')
    ax[2,0].set_ylabel('AON Loss Rate')
    correlation_coefficient, p_value = pearsonr(manufacture['KCC_ManufacturedHomes'],manufacture['AON_ManufacturedHomes'])
    ax[2,0].text(60, 80, f"r\u00b2: {correlation_coefficient ** 2:.2f}")

    ax[2,1].scatter(manufacture['KCC_ManufacturedHomes'],manufacture['FIU_ManufacturedHomes'])
    ax[2,1].set_xlim([0,100])
    ax[2,1].set_ylim([0,100])
    ax[2,1].set_xticks(np.arange(0, 125, 25))
    ax[2,1].set_yticks(np.arange(0, 125, 25))
    ax[2,1].set_xlabel('Karen Clark & Company Loss Rate')
    ax[2,1].set_ylabel('FPFLM Loss Rate')
    correlation_coefficient, p_value = pearsonr(manufacture['KCC_ManufacturedHomes'],manufacture['FIU_ManufacturedHomes'])
    ax[2,1].text(60, 80, f"r\u00b2: {correlation_coefficient ** 2:.2f}")

    ax[2,2].scatter(manufacture['AON_ManufacturedHomes'],manufacture['FIU_ManufacturedHomes'])
    ax[2,2].set_xlim([0,100])
    ax[2,2].set_ylim([0,100])
    ax[2,2].set_xticks(np.arange(0, 125, 25))
    ax[2,2].set_yticks(np.arange(0, 125, 25))
    ax[2,2].set_xlabel('AON Loss Rate')
    ax[2,2].set_ylabel('FPFLM Loss Rate')
    correlation_coefficient, p_value = pearsonr(manufacture['AON_ManufacturedHomes'],manufacture['FIU_ManufacturedHomes'])
    ax[2,2].text(60, 80, f"r\u00b2: {correlation_coefficient ** 2:.2f}")

    fig.savefig('FCHLPM_Flood_Loss_Rate_Compare.png', dpi=300)
    # plt.show()

    return

## function to compare vulnerability functions
def vulnerability_compare():
    data = pd.read_csv('FCHLPM_flood_vulnerability_functions.csv')

    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax[0].plot(data['Depth (ft)'], data['FPFLM_Inland'], label='FPFLM', color='#ff5700')
    ax[0].plot(data['Depth (ft)'], data['Aon_Inland'], label='Aon', color='#6a994e')
    ax[0].plot(data['Depth (ft)'], data['KCC_Inland'], label='KCC', color='#2a4195')

    ax[1].plot(data['Depth (ft)'], data['FPFLM_Coastal'], label='FPFLM', color='#ff5700')
    ax[1].plot(data['Depth (ft)'], data['Aon_Coastal'], label='Aon', color='#6a994e')
    ax[1].plot(data['Depth (ft)'], data['KCC_Coastal'], label='KCC', color='#2a4195')

    plt.legend()
    ax[0].set_ylabel('Ratio Damage/Exposure')
    plt.text(-9, -0.19, 'Water Depth (ft)')
    ax[0].set_title('Inland Depth-Damage Function')
    ax[1].set_title('Coastal Depth-Damage Function')
    plt.tight_layout()
    plt.savefig('FCHLPM_Flood_Vulnerability_Functions.png', dpi=300)

    return


#############################


# pdf_to_csv()

compare_loss()

# vulnerability_compare()
