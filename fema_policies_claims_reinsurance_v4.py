import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt


STATES = [
    "AL",
    "AR",
    "AZ",
    "CA",
    "CO",
    "CT",
    "DC",
    "DE",
    "FL",
    "GA",
    "IA",
    "ID",
    "IL",
    "IN",
    "KS",
    "KY",
    "LA",
    "MA",
    "MD",
    "ME",
    "MI",
    "MN",
    "MO",
    "MS",
    "MT",
    "NC",
    "ND",
    "NE",
    "NH",
    "NJ",
    "NM",
    "NV",
    "NY",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VA",
    "VT",
    "WA",
    "WI",
    "WV",
    "WY",
]

FIPS_TO_STATE = {
    1: "AL", 2: "AK", 4: "AZ", 5: "AR", 6: "CA", 8: "CO",
    9: "CT", 10: "DE", 11: "DC", 12: "FL", 13: "GA", 15: "HI",
    16: "ID", 17: "IL", 18: "IN", 19: "IA", 20: "KS", 21: "KY",
    22: "LA", 23: "ME", 24: "MD", 25: "MA", 26: "MI", 27: "MN",
    28: "MS", 29: "MO", 30: "MT", 31: "NE", 32: "NV", 33: "NH",
    34: "NJ", 35: "NM", 36: "NY", 37: "NC", 38: "ND", 39: "OH",
    40: "OK", 41: "OR", 42: "PA", 44: "RI", 45: "SC", 46: "SD",
    47: "TN", 48: "TX", 49: "UT", 50: "VT", 51: "VA", 53: "WA",
    54: "WV", 55: "WI", 56: "WY",
    # territories
    60: "AS", 66: "GU", 69: "MP", 72: "PR", 78: "VI",
}

RES1_4 = [1, 2, 11, 12, 14]  # residential (1-4 units) occupancy codes


def weighted_median(values, weights):
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    cw = np.cumsum(w) / w.sum()
    return v[np.searchsorted(cw, 0.5)]

def nfip_firststreet_losscost():
    DMG_MAX = 2e6   # nominal res 1-4 building damage above this is treated as an entry error
    RC_CEIL = 1e7 # replacement cost above this is treated as missing
    START_YEAR = 1978
    END_YEAR = 2023
    states_noDC = [s for s in STATES if s != 'DC'] # NSI data doesn't have any valid SFHA fields for properties in DC

    policy_data = pd.read_parquet("../NfipPoliciesV3.parquet",columns=["reportedZipCode", "policyEffectiveDate", "policyCount", "ratedFloodZone", "propertyState", "occupancyType", "buildingReplacementCost"])
    policy_data = policy_data.dropna(how='all')
    policy_data = policy_data[policy_data['propertyState'].isin(states_noDC)]
    policy_data = policy_data[policy_data['occupancyType'].isin(RES1_4)]
    policy_data = policy_data[
        policy_data["ratedFloodZone"]
        .str.strip()
        .str.upper()
        .str.contains(r"^[AV]", na=False)
    ]
    policy_data["reportedZipCode"] = pd.to_numeric(policy_data["reportedZipCode"].astype(str).str[:5], errors="coerce")
    policy_data["year"] = pd.to_datetime(policy_data['policyEffectiveDate']).dt.year
    pc = pd.to_numeric(policy_data["policyCount"], errors="coerce")
    bad = pc.isna() | (pc < 1) | (pc % 1 != 0)
    policy_data["count"] = pc.mask(bad, 1).astype(int)

    nfip_riskcount = (
        policy_data.groupby(["reportedZipCode", "year"])["count"]
        .sum()
        .reset_index()
    )

    nfip_policies_data = pd.read_csv("./NFIP_total_exposure_policies.csv")
    policies_2009 = nfip_policies_data[nfip_policies_data["Year"] == 2009][
        "Policies"
    ].to_numpy()[0]
    nfip_policies_data["trendFactor2009"] = (
        policies_2009 / nfip_policies_data["Policies"]
    )

    # 2020 count per zip
    c2020 = (
        nfip_riskcount.loc[nfip_riskcount["year"] == 2020]
        .drop_duplicates("reportedZipCode")
        .set_index("reportedZipCode")["count"]
    )

    nfip_riskcount["trendFactor2020"] = (
        nfip_riskcount["reportedZipCode"].map(c2020) / nfip_riskcount["count"]
    )

    nfip_riskcount.dropna(
        inplace=True
    )  # remove any zip codes where there are no 2020 policies or simply bad data
    nfip_riskcount = nfip_riskcount.astype({"reportedZipCode": int, "year": int})

    construction_index = pd.read_excel('price_uc_cust.xlsx', sheet_name='Fixed', skiprows=5, nrows=63, usecols=['Year', 'Annual index'])
    construction_index2020 = construction_index[construction_index["Year"] == 2020]["Annual index"].to_numpy()[
        0
    ]
    construction_index["constructionCost2020Factor"] = construction_index2020 / construction_index["Annual index"]

    total_inflation_trend = dict(
        zip(construction_index["Year"], construction_index["constructionCost2020Factor"].astype(float))
    )

    path = "../NfipClaimsV3.parquet"
    COLS = [
        "yearOfLoss",
        "buildingDamageAmount",
        "state",
        "reportedZipCode",
        "occupancyType",
        "ratedFloodZone",
        "buildingPropertyValue",
        "buildingReplacementCost",
    ]
    cols = set(pq.read_schema(path).names)
    want = [c for c in COLS if c in cols]
    claims = pd.read_parquet(path, columns=want)
    claims = claims[(claims["yearOfLoss"] >= START_YEAR) & (claims["yearOfLoss"] <= END_YEAR)]
    claims = claims[
        claims["ratedFloodZone"]
        .str.strip()
        .str.upper()
        .str.contains(r"^[AV]", na=False)
    ]

    drop = claims["buildingDamageAmount"] > DMG_MAX
    claims = claims[~drop].copy()

    #Claims in CONUS
    claims = claims[claims['state'].isin(states_noDC)]
    claims = claims[claims["occupancyType"].isin(RES1_4)]

    claims["reportedZipCode"] = pd.to_numeric(claims["reportedZipCode"].astype(str).str[:5], errors="coerce")
    claims.dropna(inplace=True, subset=["reportedZipCode", "yearOfLoss"])
    claims = claims[
        (claims["reportedZipCode"] > 1000) & (claims["reportedZipCode"] < 99999)
    ]  # remove remaining zip codes in PR or invalid ones
    claims["reportedZipCode"] = claims["reportedZipCode"].astype(int)

    # convert ACV to RCV
    min_year=2000
    r_bounds=(0.8, 3.0)
    v = claims[
        (claims["yearOfLoss"] >= min_year)
        & (claims["buildingPropertyValue"] > 0)
        & (claims["buildingReplacementCost"] > 0)
        & (claims["buildingDamageAmount"] > 0)
    ].copy()
    v["r"] = v["buildingReplacementCost"] / v["buildingPropertyValue"]
    v = v[v["r"].between(*r_bounds)]
    is125 = np.isclose(v["r"], 1.25, atol=1e-3)

    uplift = weighted_median(v["r"], v["buildingDamageAmount"])
    uplift_low = weighted_median(v.loc[~is125, "r"], v.loc[~is125, "buildingDamageAmount"])

    print(f"n={len(v):,} | share exactly 1.25: {is125.mean():.1%}")
    print(f"uplift (all): {uplift:.4f} | uplift (excl. 1.25 convention): {uplift_low:.4f}")

    claims["buildingDamageAmount"] *= uplift

    rc_c, pv_c = claims["buildingReplacementCost"], claims["buildingPropertyValue"]
    val = rc_c.where((rc_c > 0) & (rc_c <= RC_CEIL), pv_c.where((pv_c > 0) & (pv_c <= RC_CEIL)) * uplift)
    claims["buildingDamageAmount"] = np.minimum(claims["buildingDamageAmount"], val.fillna(np.inf))

    claims["buildingDamageAmount"] = claims["buildingDamageAmount"].clip(lower=0)

    f = claims["yearOfLoss"].map(total_inflation_trend)
    assert f.notna().all(), f"missing inflation years: {sorted(claims.loc[f.isna(), 'yearOfLoss'].unique())}"

    claims["buildingDamageAmount"] = claims[
        "buildingDamageAmount"
    ] * claims["yearOfLoss"].map(total_inflation_trend)

    claims = (
        claims.groupby(["reportedZipCode", "state", "yearOfLoss"])["buildingDamageAmount"]
        .sum()
        .reset_index()
    )

    claims["yearOfLoss"] = claims["yearOfLoss"].astype(int)
    nat = nfip_policies_data.set_index("Year")["trendFactor2009"]
    pre = claims["yearOfLoss"] < 2009
    pre_factor = claims.loc[pre, "yearOfLoss"].map(nat)

    assert pre_factor.notna().all(), "missing pre-2009 national policy years"

    claims.loc[pre, "buildingDamageAmount"] *= pre_factor

    rt = nfip_riskcount.set_index(["reportedZipCode", "year"])["trendFactor2020"]
    idx = pd.MultiIndex.from_arrays(
        [claims["reportedZipCode"], claims["yearOfLoss"].clip(lower=2009)]
    )
    claims["riskTrend"] = rt.reindex(idx).to_numpy()

    miss = claims["riskTrend"].isna()
    print(f"No exposure match: {miss.mean():.1%} of zip-years, "
          f"{claims.loc[miss, 'buildingDamageAmount'].sum() / claims['buildingDamageAmount'].sum():.1%} of loss")

    claims["buildingDamageAmount"] *= claims["riskTrend"].fillna(0)
    claims_post2000 = claims[claims["yearOfLoss"] >= 2000]
    print(f"Percent of building loss occurring in 2000 or later: {claims_post2000['buildingDamageAmount'].sum() / claims['buildingDamageAmount'].sum() * 100}")

    n_years = END_YEAR - START_YEAR + 1
    annual_loss = claims["buildingDamageAmount"].sum() / n_years

    policy_data = policy_data[policy_data["year"] == 2020].copy()

    rc = policy_data["buildingReplacementCost"]
    missing = rc.isna() | (rc <= 0)
    too_high = rc > RC_CEIL
    print(f"RC missing/zero: {missing.mean():.1%} | RC > ${RC_CEIL:,.0f}: "
          f"{too_high.sum()} policies, {rc[too_high].sum() / rc.sum():.1%} of raw RC")

    bad = missing | too_high
    med = policy_data.loc[~bad].groupby("propertyState")["buildingReplacementCost"].median()
    policy_data.loc[bad, "buildingReplacementCost"] = policy_data.loc[bad, "propertyState"].map(med)
    assert policy_data["buildingReplacementCost"].notna().all()

    total_replacement_cost = policy_data["buildingReplacementCost"].sum()

    print(f"NFIP annual loss: {annual_loss} and replacement cost: {total_replacement_cost}")

    print(f"NFIP SFHA loss cost: ${annual_loss / total_replacement_cost * 1000:.2f}")

    fs = pd.read_csv("./FirstStreet/FirstStreet_AAL.csv")
    nsi_data = pd.read_csv("../NSI/nsi_sfha_by_state.csv")

    fs = fs.merge(nsi_data, on="State", how="left")

    rc_state = policy_data.groupby("propertyState")["buildingReplacementCost"].sum()

    fs["share_count"] = fs["Inside SFHA # Properties"] / fs["n_sfha_res14"]
    print(fs)
    fs["d"] = fs["Inside SFHA LossRatio"] * fs["share_count"]          # within-dataset only
    assert (fs["share_count"].dropna() <= 1).all(), fs.loc[fs.share_count > 1, ["State", "share_count"]]

    fs["w_nfip"] = fs["State"].map(rc_state)
    ok = fs["d"].notna() & fs["w_nfip"].notna()
    burn_fs_nfipwt = (fs.loc[ok, "d"] * fs.loc[ok, "w_nfip"]).sum() / fs.loc[ok, "w_nfip"].sum() * 1000

    print(f"First Street-implied AAL on NFIP book: {(fs.loc[ok, 'd'] * fs.loc[ok, 'w_nfip']).sum()}, Coverage: {fs.loc[ok, 'w_nfip'].sum()}")
    print(f"First Street Substantial Risk SFHA burn rate: ${burn_fs_nfipwt:.2f}")

    return


def nfip_policies_count(first_year=2009, last_year=2023):
    p = pd.read_parquet("../NfipPoliciesV3.parquet", columns=["policyEffectiveDate", "policyCount"])
    p["year"] = pd.to_datetime(p["policyEffectiveDate"], errors="coerce").dt.year
    pc = pd.to_numeric(p["policyCount"], errors="coerce")
    p["count"] = pc.mask(pc.isna() | (pc < 1) | (pc % 1 != 0), 1).astype(int)

    post = (p[p["year"].between(first_year, last_year)]
            .groupby("year")["count"].sum()
            .rename_axis("Year").rename("Policies").reset_index())
    pre = pd.read_csv("./NFIP_total_exposure_policies_pre2009.csv")
    pre = pre[pre["Year"] < first_year]

    out = pd.concat([pre, post], ignore_index=True).astype({"Year": int})
    assert out["Year"].is_unique and out["Year"].is_monotonic_increasing
    out.to_csv("NFIP_total_exposure_policies.csv", index=False)

    return

def std_catmodel(model_all_rp):
    d = model_all_rp.sort_values("Probability")
    p = np.r_[d["Probability"].to_numpy(), 1.0]   # anchor high-frequency end
    L = np.r_[d["AEP_Coastal_millions"].to_numpy(), 0.0]             # lower bound; tail beyond max RP still truncated
    m1, m2 = np.trapezoid(L, p), np.trapezoid(L**2, p)
    var_coastal = m2 - m1**2

    p = np.r_[d["Probability"].to_numpy(), 1.0]   # anchor high-frequency end
    L = np.r_[d["AEP_Inland_millions"].to_numpy(), 0.0]             # lower bound; tail beyond max RP still truncated
    m1, m2 = np.trapezoid(L, p), np.trapezoid(L**2, p)
    var_inland = m2 - m1**2

    return np.sqrt(var_coastal + var_inland)


## function to compare reinsurance cat model AAL to NFIP claims data
def aal_catmodel_validation():
    catmodels = pd.read_csv(
        "./fema_nfip-reinsurance-placement-information_2023/FEMA_reinsurance_2023_AAL.csv",
        index_col="State",
    )

    # apply scale AAL and Limit values
    catmodels["Katrisk_Storm Surge_AAL"] = catmodels["Katrisk_Storm Surge_AAL"] * 1000
    catmodels["Katrisk_Inland_AAL"] = catmodels["Katrisk_Inland_AAL"] * 1000
    catmodels["Katrisk_Limit"] = catmodels["Katrisk_Limit"] * 1000
    catmodels["AIR_Storm Surge_AAL"] = catmodels["AIR_Storm Surge_AAL"] * 1000
    catmodels["AIR_Inland_AAL"] = catmodels["AIR_Inland_AAL"] * 1000
    catmodels["AIR_Limit"] = catmodels["AIR_Limit"] * 1000
    catmodels["RMS_Storm_Surge_AAL"] = catmodels["RMS_Storm_Surge_AAL"] * 1000
    catmodels["RMS_Limit"] = catmodels["RMS_Limit"] * 1000

    # calculate mean CONUS storm surge AAL
    cat_models_ss_aal = np.mean(
        [
            catmodels["Katrisk_Storm Surge_AAL"].sum(),
            catmodels["AIR_Storm Surge_AAL"].sum(),
            catmodels["RMS_Storm_Surge_AAL"].sum(),
        ]
    )

    # calculate mean CONUS inland AAL
    cat_models_inland_aal = np.mean(
        [catmodels["Katrisk_Inland_AAL"].sum(), catmodels["AIR_Inland_AAL"].sum()]
    )

    # calculate mean CONUS limit
    catmodels['Mean_Limit'] = catmodels.filter(like='Limit').mean(axis=1)
    cat_models_limit = catmodels['Mean_Limit'].sum()

    # calculate CONUS burn rate using the mean storm surge and inland AAL and CONUS NFIP coverage for residential (1-4 units) in FEMA 100-year flood zone per $1,000 of coverage
    cat_models_burn_rate_mean = (cat_models_ss_aal + cat_models_inland_aal) / (
        cat_models_limit  / 1000
    )
    # calculate KatRisk burn rate
    katrisk_burn_rate = (
        catmodels["Katrisk_Storm Surge_AAL"].sum()
        + catmodels["Katrisk_Inland_AAL"].sum()
    ) / (catmodels["Katrisk_Limit"].sum() / 1000)
    # calculate Verisk burn rate
    verisk_burn_rate = (
        catmodels["AIR_Storm Surge_AAL"].sum() + catmodels["AIR_Inland_AAL"].sum()
    ) / (catmodels["AIR_Limit"].sum() / 1000)
    # print cat model burn rates for CONUS
    print(
        f"Cat Models Average Burn Rate: {cat_models_burn_rate_mean}; Katrisk Burn Rate: {katrisk_burn_rate}; AIR Burn Rate: {verisk_burn_rate}"
    )
    print(
        f"Cat Models Average AAL: {cat_models_ss_aal + cat_models_inland_aal}; Katrisk AAL: {catmodels['Katrisk_Storm Surge_AAL'].sum() + catmodels['Katrisk_Inland_AAL'].sum()}; AIR AAL: {catmodels['AIR_Storm Surge_AAL'].sum() + catmodels['AIR_Inland_AAL'].sum()}"
    )
    print(
        f"Cat Models Average coverage: {cat_models_limit}; Katrisk coverage: {catmodels['Katrisk_Limit'].sum()}; AIR coverage: {catmodels['AIR_Limit'].sum()}"
    )

    catmodels_copy = catmodels.reset_index()
    catmodels_fl = catmodels_copy[catmodels_copy["State"] == "FL"]
    katrisk_fl_burn_rate = (catmodels_fl["Katrisk_Storm Surge_AAL"] + catmodels_fl["Katrisk_Inland_AAL"]) / (catmodels_fl["Katrisk_Limit"] / 1000)
    verisk_fl_burn_rate = (catmodels_fl["AIR_Storm Surge_AAL"] + catmodels_fl["AIR_Inland_AAL"]) / (catmodels_fl["AIR_Limit"] / 1000)
    print(f"Verisk Florida burn rate: {katrisk_fl_burn_rate.values[0]:.2f}")
    print(f"Verisk Florida burn rate: {verisk_fl_burn_rate.values[0]:.2f}")
    print(stop)

    catmodels_all_rp = pd.read_csv("./fema_nfip-reinsurance-placement-information_2023/fema_reinsurance_2023_AAL_returnperiod.csv")
    catmodels_all_rp['Probability'] = 1.0 / catmodels_all_rp["ReturnPeriod"]
    catmodels_all_rp = catmodels_all_rp.fillna(0)

    # standard deviation of AAL for cat models
    verisk_all_rp = catmodels_all_rp[catmodels_all_rp["Model"] == 'Verisk']
    verisk_std = std_catmodel(verisk_all_rp)
    print(f"Verisk standard deviation of loss: ${verisk_std} millions")

    katrisk_all_rp = catmodels_all_rp[catmodels_all_rp["Model"] == 'KatRisk']
    katrisk_std = std_catmodel(katrisk_all_rp)
    print(f"KatRisk standard deviation of loss: ${katrisk_std} millions")

    START_YEAR = 1978
    END_YEAR = 2023

    policy_data = pd.read_parquet("../NfipPoliciesV3.parquet",columns=["reportedZipCode", "policyEffectiveDate", "policyCount", "ratedFloodZone", "propertyState", "occupancyType", "totalBuildingInsuranceCoverage", "totalContentsInsuranceCoverage"])
    policy_data = policy_data.dropna(how='all')
    policy_data = policy_data[policy_data['propertyState'].isin(STATES)]

    policy_data["year"] = pd.to_datetime(policy_data['policyEffectiveDate']).dt.year
    policy_data = policy_data[(policy_data["year"] >= START_YEAR) & (policy_data["year"] <= END_YEAR)]

    policy_data["reportedZipCode"] = pd.to_numeric(policy_data["reportedZipCode"].astype(str).str[:5], errors="coerce")

    pc = pd.to_numeric(policy_data["policyCount"], errors="coerce")
    bad = pc.isna() | (pc < 1) | (pc % 1 != 0)
    policy_data["count"] = pc.mask(bad, 1).astype(int)

    nfip_riskcount = (
        policy_data.groupby(["reportedZipCode", "year"])["count"]
        .sum()
        .reset_index()
    )

    nfip_policies_data = pd.read_csv("./NFIP_total_exposure_policies.csv")
    policies_2009 = nfip_policies_data[nfip_policies_data["Year"] == 2009][
        "Policies"
    ].to_numpy()[0]
    nfip_policies_data["trendFactor2009"] = (
        policies_2009 / nfip_policies_data["Policies"]
    )

    # 2022 count per zip
    c2022 = (
        nfip_riskcount.loc[nfip_riskcount["year"] == 2022]
        .drop_duplicates("reportedZipCode")
        .set_index("reportedZipCode")["count"]
    )

    nfip_riskcount["trendFactor2022"] = (
        nfip_riskcount["reportedZipCode"].map(c2022) / nfip_riskcount["count"]
    )

    nfip_riskcount.dropna(
        inplace=True
    )  # remove any zip codes where there are no 2022 policies or simply bad data
    nfip_riskcount = nfip_riskcount.astype({"reportedZipCode": int, "year": int})

    construction_index = pd.read_excel('price_uc_cust.xlsx', sheet_name='Fixed', skiprows=5, nrows=63, usecols=['Year', 'Annual index'])
    construction_index2022 = construction_index[construction_index["Year"] == 2022]["Annual index"].to_numpy()[
        0
    ]
    construction_index["constructionCost2022Factor"] = construction_index2022 / construction_index["Annual index"]

    construction_trend = dict(
        zip(construction_index["Year"], construction_index["constructionCost2022Factor"].astype(float))
    )

    cpi_data = pd.read_csv("USCPI_1978-2025_MinnFed.csv", skiprows=1)

    cpi2022 = cpi_data[cpi_data["Year"] == 2022][
        "U.S. Consumer Price Index *"
    ].to_numpy()[0]
    cpi_data["cpi2022factor"] = cpi2022 / cpi_data["U.S. Consumer Price Index *"]

    cpi_trend = dict(
        zip(cpi_data["Year"], cpi_data["cpi2022factor"].astype(float))
    )

    # load in NFIP claims data
    loss_data = pd.read_parquet(
        "../NfipClaimsV3.parquet",
        columns=[
            "reportedZipCode",
            "netContentsPaymentAmount",
            "netBuildingPaymentAmount",
            "yearOfLoss",
            "occupancyType",
            "state",
            "totalBuildingInsuranceCoverage",
            "totalContentsInsuranceCoverage"
        ],
    )
    loss_data.dropna(inplace=True, subset=["reportedZipCode", "yearOfLoss"])
    loss_data = loss_data[loss_data['state'].isin(STATES)]
    loss_data["yearOfLoss"] = loss_data["yearOfLoss"].astype(int)
    loss_data["netBuildingPaymentAmount"] = loss_data["netBuildingPaymentAmount"].astype(float)
    loss_data["netContentsPaymentAmount"] = loss_data["netContentsPaymentAmount"].astype(float)
    loss_data = loss_data[(loss_data["yearOfLoss"] >= START_YEAR) & (loss_data["yearOfLoss"] <= END_YEAR)]

    loss_data["reportedZipCode"] = pd.to_numeric(loss_data["reportedZipCode"].astype(str).str[:5], errors="coerce")

    loss_data = loss_data[
        (loss_data["reportedZipCode"] > 1000) & (loss_data["reportedZipCode"] < 99999)
    ]  # remove remaining zip codes in PR or invalid ones
    loss_data["reportedZipCode"] = loss_data["reportedZipCode"].astype(int)

    loss_data["netBuildingPaymentAmount"] = loss_data["netBuildingPaymentAmount"].fillna(0).clip(lower=0)
    loss_data["netContentsPaymentAmount"] = loss_data["netContentsPaymentAmount"].fillna(0).clip(lower=0)

    fb = loss_data["yearOfLoss"].map(construction_trend)
    fc = loss_data["yearOfLoss"].map(cpi_trend)

    missing = sorted(set(loss_data.loc[fb.isna() | fc.isna(), "yearOfLoss"]))
    assert not missing, f"missing inflation years: {missing}"

    loss_data["netBuildingPaymentAmount"] *= fb
    loss_data["netContentsPaymentAmount"] *= fc

    # Statutory maximums per occupancy (current program limits, in 2022 dollars as-is)
    BLDG_MAX = {
        **dict.fromkeys([1, 11, 14, 16], 250_000),   # single-family / unit / manufactured
        **dict.fromkeys([2, 12], 1_000_000),         # 2-4 units; small condo buildings on RCBAP: 4 x $250k
        **dict.fromkeys([3, 13], 500_000),           # other residential / 5+ units (general property form)
        15: np.inf,                                  # RCBAP: $250k x units, no unit count in claims
        **dict.fromkeys([4, 6, 17, 18, 19], 500_000), # non-residential
    }
    CONT_MAX = {
        **dict.fromkeys([1, 2, 3, 11, 12, 13, 14, 15, 16], 100_000),
        **dict.fromkeys([4, 6, 17, 18, 19], 500_000),
    }

    def cap_to_trended_coverage(pay, cov, factor, occ, max_by_occ, nan_occ_max):
        """Cap each trended payment at its policy's coverage, trended by the same factor,
        bounded by the statutory max. Missing/zero coverage falls back to the statutory max."""
        stat = occ.map(max_by_occ).where(occ.notna(), nan_occ_max).fillna(np.inf)  # unlisted codes: uncapped
        cap = (cov * factor).clip(upper=stat)
        cap = cap.where(cov > 0, stat)
        return np.minimum(pay, cap)

    before_b = loss_data["netBuildingPaymentAmount"].sum()
    before_c = loss_data["netContentsPaymentAmount"].sum()

    loss_data["netBuildingPaymentAmount"] = cap_to_trended_coverage(
        loss_data["netBuildingPaymentAmount"], loss_data["totalBuildingInsuranceCoverage"],
        fb, loss_data["occupancyType"], BLDG_MAX, nan_occ_max=250_000)
    loss_data["netContentsPaymentAmount"] = cap_to_trended_coverage(
        loss_data["netContentsPaymentAmount"], loss_data["totalContentsInsuranceCoverage"],
        fc, loss_data["occupancyType"], CONT_MAX, nan_occ_max=100_000)

    print(f"Coverage caps removed: building ${(before_b - loss_data['netBuildingPaymentAmount'].sum())/1e9:.2f}B "
          f"({(before_b - loss_data['netBuildingPaymentAmount'].sum())/before_b:.1%}), "
          f"contents ${(before_c - loss_data['netContentsPaymentAmount'].sum())/1e9:.2f}B "
          f"({(before_c - loss_data['netContentsPaymentAmount'].sum())/before_c:.1%})")

    loss_data["totalPayment"] = (
        loss_data["netBuildingPaymentAmount"] + loss_data["netContentsPaymentAmount"]
    )

    loss_data = (
        loss_data.groupby(["reportedZipCode", "yearOfLoss"])["totalPayment"]
        .sum()
        .reset_index()
    )
    nat = nfip_policies_data.set_index("Year")["trendFactor2009"]
    pre = loss_data["yearOfLoss"] < 2009
    pre_factor = loss_data.loc[pre, "yearOfLoss"].map(nat)

    assert pre_factor.notna().all(), "missing pre-2009 national policy years"

    loss_data.loc[pre, "totalPayment"] *= pre_factor
    rt = nfip_riskcount.set_index(["reportedZipCode", "year"])["trendFactor2022"]

    idx = pd.MultiIndex.from_arrays(
        [loss_data["reportedZipCode"], loss_data["yearOfLoss"].clip(lower=2009)]
    )
    loss_data["riskTrend"] = rt.reindex(idx).to_numpy()

    miss = loss_data["riskTrend"].isna()
    print(f"No exposure match: {miss.mean():.1%} of zip-years, "
          f"{loss_data.loc[miss, 'totalPayment'].sum() / loss_data['totalPayment'].sum():.1%} of loss")

    loss_data["totalPayment"] *= loss_data["riskTrend"].fillna(0)

    loss_data_year = loss_data.groupby('yearOfLoss')['totalPayment'].sum()

    n_years = END_YEAR - START_YEAR + 1
    annual_loss = loss_data["totalPayment"].sum() / n_years

    policy_data = policy_data[policy_data['year'] == 2022]

    building_coverage = policy_data["totalBuildingInsuranceCoverage"].fillna(0)
    contents_coverage = policy_data["totalContentsInsuranceCoverage"].fillna(0)
    b, c = policy_data["totalBuildingInsuranceCoverage"], policy_data["totalContentsInsuranceCoverage"]
    print(f"missing/zero building & contents: {((b.isna() | (b <= 0)) & (c.isna() | (c <= 0))).mean():.1%}")

    policy_data["totalCoverage"] = building_coverage + contents_coverage
    total_coverage = policy_data["totalCoverage"].sum()

    print(f"NFIP AAL: {annual_loss}")
    print(f"NFIP coverage: {total_coverage}")
    print(f"NFIP loss rate: {annual_loss / (total_coverage / 1000)}")
    loss_by_year_2000 = loss_data_year[loss_data_year.index >= 2000]
    print(
        f"NFIP loss rate for 2000 to present: {loss_by_year_2000.mean() / (total_coverage / 1000)}"
    )

    return


def nsi_shfa():
    import duckdb
    from pathlib import Path

    con = duckdb.connect()
    con.install_extension("spatial")
    con.load_extension("spatial")

    NSI = (Path(__file__).parent / "../NSI").resolve()

    # One-time: convert each state GeoPackage to Parquet (skips states already converted)
    for gpkg in sorted(NSI.glob("nsi_2022_*.gpkg")):
        fips = gpkg.stem.split("_")[-1]            # 'nsi_2022_06' -> '06'
        out = NSI / f"st={fips}.parquet"
        if out.exists():
            continue
        print(f"Converting {gpkg.name}")
        con.sql(f"""
            COPY (
                SELECT fd_id, cbfips, firmzone, occtype, st_damcat, val_struct, x, y
                FROM ST_Read('{gpkg}')
            ) TO '{out}' (FORMAT parquet)
        """)

    SFHA_PRED = ("regexp_matches(upper(trim(firmzone)), '^(A|V)') "
                 "AND upper(trim(firmzone)) NOT LIKE 'AREA%'")
    RES1_4 = "regexp_matches(occtype, '^(RES1|RES2|RES3A|RES3B)')"   # 1–4 unit residential

    df = con.sql(f"""
        SELECT CAST(substr(cbfips, 1, 2) AS INTEGER)                                        AS state_fips,
               count(*)                    FILTER (WHERE {SFHA_PRED})                       AS n_sfha,
               count(*)                                                                     AS n_total,
               coalesce(sum(val_struct)    FILTER (WHERE {SFHA_PRED}), 0)                   AS val_sfha,
               sum(val_struct)                                                              AS val_total,
               count(*)                    FILTER (WHERE {SFHA_PRED} AND {RES1_4})          AS n_sfha_res14,
               coalesce(sum(val_struct)    FILTER (WHERE {SFHA_PRED} AND {RES1_4}), 0)      AS val_sfha_res14,
               avg((firmzone IS NULL OR trim(firmzone) = '')::INT)                          AS null_share
        FROM '{NSI}/st=*.parquet'
        GROUP BY 1
        ORDER BY 1
    """).df()

    df["State"] = df["state_fips"].map(FIPS_TO_STATE)
    df.loc[df["null_share"] == 1.0, ["n_sfha", "val_sfha", "n_sfha_res14", "val_sfha_res14"]] = np.nan  # DC: zone join failed

    print(df.to_string(index=False))
    print(f"National SFHA: {df.n_sfha.sum():,.0f} structures | ${df.val_sfha.sum()/1e9:,.1f}B")
    print(f"National SFHA res 1-4: {df.n_sfha_res14.sum():,.0f} | ${df.val_sfha_res14.sum()/1e9:,.1f}B")

    df.to_csv(NSI / "nsi_sfha_by_state.csv", index=False)


    return


## function to compare vulnerability functions
def vulnerability_compare():
    data = pd.read_csv('./FCHLPM/FCHLPM_flood_vulnerability_functions.csv')

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
    plt.savefig('../FCHLPM/FCHLPM_Flood_Vulnerability_Functions.png', dpi=300)

    return



########################################


# nfip_firststreet_losscost()

# nsi_shfa()

# nfip_policies_count()

aal_catmodel_validation()

# vulnerability_compare()
