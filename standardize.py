"""
Direct standardization of FCHLPM flood-model loss costs (Form AF-1 only) to a common NFIP exposure basis.

Form AF-1 (2021 flood standards) reports, by ZIP, the zero-deductible standard flood loss cost per
$1,000 of Coverage A for a common notional structure (Notional Set 3: built 1989, 1 story;
owners A=$100k / B=$40k; manufactured homes A=$50k / B=$25k). Each model's ZIP values are
combined into one loss cost per ZIP and re-aggregated to the state, weighted by NFIP policies in
force by ZIP and occupancy:
  single-family   occupancyType 1 (legacy; includes manufactured homes) and 11 (RR2.0)
                  -> FRAME_SHARE x frame + (1 - FRAME_SHARE) x masonry AF-1 loss cost
  manufactured    occupancyType 14 (RR2.0 residential manufactured home) -> manufactured-home loss cost
The NFIP policies file has no wall-construction field, so the frame/masonry split is an assumption
(FRAME_SHARE). Legacy policies do not separate manufactured homes, so the manufactured share is
understated for snapshots before RR2.0 (Oct 2021 new business, Apr 2022 renewals) and rises with
later snapshots for coding reasons alone.

Inputs
  af1_fpflm.csv, af1_if.csv, af1_kcc.csv   AF-1 loss costs by ZIP (std_fr, std_ma, std_mh)
  NfipPoliciesV3.parquet                   OpenFEMA policies file, found in this folder or any parent;
                                           only read (needs pyarrow) when the weights file for SNAP
                                           does not exist yet or REBUILD is True

Outputs
  nfip_fl_inforce_<snap>_by_zip.csv        weights: FL policies in force on <snap> by
                                           zip x county x occupancyType x SFHA (cached, reused)
  results_af1_standardized_<snap>.csv      statewide weighted loss cost per model
  results_zip_merged_<snap>.csv            ZIP-level weights, AF-1 values and combined loss cost per model
  FCHLPM_Flood_Loss_Rate_Compare.png       unweighted log-log ZIP scatter, model pair x construction type

Caveats (not removed by reweighting)
  - Each modeler placed the notional structures at its own locations within a ZIP
    (FPFLM: its NFIP-2012 exposure locations; IF/KCC: not fully disclosed).
  - Attributes not specified by Notional Set 3 (e.g., first-floor height) were set by each modeler.
  - FPFLM reports some ZIPs in several rating areas without weights; the mean is used.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr

D = Path(__file__).resolve().parent
CONS = ["fr", "ma", "mh"]
MODELS = ["FPFLM", "IF", "KCC"]
SF_OCC = [1, 11]     # single-family: legacy (includes manufactured homes), RR2.0
MH_OCC = [14]        # RR2.0 residential manufactured/mobile home
FRAME_SHARE = 0.5    # assumed frame share of single-family policies (no construction field in NFIP data)
SNAP = "20240101"    # NFIP policies-in-force snapshot date, YYYYMMDD
REBUILD = False      # True: rebuild the weights file for SNAP from the parquet even if it exists


# ---------------------------------------------------------------- NFIP weights
def aggregate_inforce(d, snap, geo):
    """Policy records -> policies in force on snap, by zip x fips x occupancyType x SFHA."""
    snap = pd.Timestamp(snap, tz="UTC")
    eff = pd.to_datetime(d["policyEffectiveDate"], utc=True, errors="coerce")
    term = pd.to_datetime(d["policyTerminationDate"], utc=True, errors="coerce")
    d = d[(eff <= snap) & (term > snap)].copy()
    d["zip"] = d["reportedZipCode"].astype(str).str[:5]
    d["fips"] = d[geo].astype(str).str[:5] if geo else ""
    d["sfha"] = d["ratedFloodZone"].astype(str).str.upper().str.match(r"^[AV]").astype(int)
    d["policyCount"] = d["policyCount"].fillna(1)
    return (d.groupby(["zip", "fips", "occupancyType", "sfha"], dropna=False)
              .agg(records=("policyCount", "size"), policies=("policyCount", "sum"),
                   bldg_cov=("totalBuildingInsuranceCoverage", "sum"),
                   cont_cov=("totalContentsInsuranceCoverage", "sum"))
              .reset_index())


def build_weights(snap, out):
    """Read Florida records from NfipPoliciesV3.parquet and write the in-force weights file."""
    src = next((p / "NfipPoliciesV3.parquet" for p in [D, *D.parents]
                if (p / "NfipPoliciesV3.parquet").exists()), None)
    if src is None:
        sys.exit("NfipPoliciesV3.parquet not found in this folder or any parent")
    import pyarrow.parquet as pq
    names = set(pq.ParquetFile(src).schema_arrow.names)
    geo = next((c for c in ["censusGeoid", "censusTract", "countyCode"] if c in names), None)
    cols = ["propertyState", "reportedZipCode", "occupancyType", "policyEffectiveDate",
            "policyTerminationDate", "policyCount", "ratedFloodZone",
            "totalBuildingInsuranceCoverage", "totalContentsInsuranceCoverage"] + ([geo] if geo else [])
    print(f"building weights for {snap} from {src}")
    d = pd.read_parquet(src, columns=cols, filters=[("propertyState", "==", "FL")])
    print(f"  FL policy records: {len(d):,}")
    agg = aggregate_inforce(d, snap, geo)
    agg.to_csv(out, index=False)
    print(f"  in force on {pd.Timestamp(snap).date()}: {int(agg['policies'].sum()):,} policies "
          f"-> {out.name} ({len(agg):,} rows)")


def load_nfip(snap, rebuild=False):
    fn = D / f"nfip_fl_inforce_{snap}_by_zip.csv"
    if rebuild or not fn.exists():
        build_weights(snap, fn)
    else:
        print(f"using existing {fn.name} (set REBUILD = True to regenerate)")
    z = pd.read_csv(fn, dtype={"zip": str, "fips": str})
    z = z[z["zip"].str.match(r"^3[234]\d{3}$", na=False)]
    return z.rename(columns={"occupancyType": "occ", "policies": "n", "bldg_cov": "bcov"})


def zip_weights(nf):
    """Policies by ZIP for each AF-1 construction class: frame, masonry, manufactured."""
    sf = nf[nf["occ"].isin(SF_OCC)].groupby("zip")["n"].sum()
    mh = nf[nf["occ"].isin(MH_OCC)].groupby("zip")["n"].sum()
    W = pd.concat({"sf": sf, "mh": mh}, axis=1).fillna(0)
    return pd.DataFrame({"w_fr": FRAME_SHARE * W["sf"], "w_ma": (1 - FRAME_SHARE) * W["sf"], "w_mh": W["mh"]})


def zip_county(nf):
    """Dominant county (by policies) of each ZIP."""
    d = nf.dropna(subset=["fips"]).groupby(["zip", "fips"])["n"].sum().reset_index()
    return d.sort_values("n").drop_duplicates("zip", keep="last").set_index("zip")["fips"]



def merge(af1, W):
    """ZIP table: class weights, each model's AF-1 loss costs, and each model's combined loss cost
    (MODEL_all = weight-averaged over frame, masonry, manufactured). Only ZIPs all models report."""
    J = W.copy()
    J["w"] = W.sum(axis=1)
    J = J[J["w"] > 0]
    for m in MODELS:
        J = J.join(af1[m].add_prefix(f"{m}_"))
        J[f"{m}_all"] = sum(J[f"w_{c}"] * J[f"{m}_{c}"] for c in CONS) / J["w"]
    return J


def standardized(J, total_w):
    """Statewide weighted average of each model's combined loss cost."""
    return pd.DataFrame([{"model": m, "ZIPs": len(J), "weight covered": J["w"].sum() / total_w,
                          "loss cost": np.average(J[f"{m}_all"], weights=J["w"])} for m in MODELS]).set_index("model")


## function to compare loss rates by zip code
def load_af1_merged(af1):
    """AF-1 loss costs on ZIPs reported by all three models; columns <AON|FIU|KCC>_<construction>."""
    src = {"KCC": "KCC", "AON": "IF", "FIU": "FPFLM"}
    cols = {"fr": "FrameOwners", "ma": "MasonryOwners", "mh": "ManufacturedHomes"}
    return pd.concat([af1[v].rename(columns=cols).add_prefix(f"{k}_") for k, v in src.items()],
                     axis=1, join="inner")


def plot_loss_compare(af1, floor=1e-3, out=D / "FCHLPM_Flood_Loss_Rate_Compare.png"):
    """Log-log scatter of loss costs for each model pair and construction type."""
    m = load_af1_merged(af1)
    pairs = [("KCC", "AON"), ("KCC", "FIU"), ("AON", "FIU")]
    types = [("FrameOwners", "Frame"), ("MasonryOwners", "Masonry"), ("ManufacturedHomes", "Manufactured")]
    label = {"KCC": "KCC", "AON": "Aon", "FIU": "FPFLM"}

    # Common axis range across all panels so slopes and offsets are comparable
    allv = m.filter(regex="_(FrameOwners|MasonryOwners|ManufacturedHomes)$").to_numpy().ravel()
    lo, hi = floor, 10 ** np.ceil(np.log10(allv.max()))

    fig, ax = plt.subplots(3, 3, figsize=(10, 10), sharex=True, sharey=True, constrained_layout=True)
    for i, (col, tname) in enumerate(types):
        for j, (a, b) in enumerate(pairs):
            x, y = m[f"{a}_{col}"].to_numpy(), m[f"{b}_{col}"].to_numpy()
            pos = (x > 0) & (y > 0)                       # log scale can't show zeros
            xa = ax[i, j]
            xa.scatter(x[pos], y[pos], s=8, alpha=0.4, edgecolor="none")
            xa.plot([lo, hi], [lo, hi], color="0.4", lw=1, ls="--")   # 1:1 line
            xa.set_xscale("log"); xa.set_yscale("log")
            xa.set_xlim(lo, hi); xa.set_ylim(lo, hi)
            xa.set_aspect("equal")

            r2_log = pearsonr(np.log10(x[pos]), np.log10(y[pos]))[0] ** 2
            xa.text(0.04, 0.96, f"r²(log) = {r2_log:.2f}\nn = {pos.sum()} ({(~pos).sum()} zero)",
                    transform=xa.transAxes, va="top", fontsize=8)
            if i == 2: xa.set_xlabel(f"{label[a]} loss cost")
            if j == 0: xa.set_ylabel(f"{tname}\n{label[b]} loss cost")
            else: xa.set_ylabel(f"{label[b]}")
    fig.savefig(out, dpi=300)
    return fig


def main(snap=SNAP, rebuild=REBUILD):
    pd.set_option("display.width", 250, "display.precision", 3, "display.max_columns", 20)
    af1, nrows = load_af1()
    nf = load_nfip(snap, rebuild)
    W = zip_weights(nf)
    tot = W.sum()
    print(f"AF-1 rows {nrows}; unique ZIPs { {m: len(af1[m]) for m in MODELS} }")
    print(f"NFIP snapshot {snap}: {int(nf['n'].sum()):,} policies in force, {int(tot.sum()):,} used: "
          f"frame {100 * tot['w_fr'] / tot.sum():.1f}%, masonry {100 * tot['w_ma'] / tot.sum():.1f}%, "
          f"manufactured {100 * tot['w_mh'] / tot.sum():.1f}% (frame share of single-family assumed {FRAME_SHARE})")

    print("\nA. NFIP-weighted AF-1 loss cost")
    J = merge(af1, W).dropna(subset=[f"{m}_all" for m in MODELS])
    A = standardized(J, tot.sum())
    print(A)
    A.to_csv(D / f"results_af1_standardized_{snap}.csv")
    J = J.join(zip_county(nf).rename("fips"))
    J.to_csv(D / f"results_zip_merged_{snap}.csv")

    plot_loss_compare(af1)
    print("\nsaved FCHLPM_Flood_Loss_Rate_Compare.png")


def load_af1():
    """AF-1 loss costs by ZIP. Returns {model: DataFrame[fr, ma, mh]} (mean over FPFLM multi-row ZIPs)."""
    files = {"FPFLM": "af1_fpflm.csv", "IF": "af1_if.csv", "KCC": "af1_kcc.csv"}
    out, nrows = {}, {}
    for m, fn in files.items():
        a = pd.read_csv(D / fn, dtype={"zip": str})
        a = a.rename(columns={f"std_{c}": c for c in CONS})
        nrows[m] = len(a)
        out[m] = a.groupby("zip")[CONS].mean()
    return out, nrows


if __name__ == "__main__":
    main()
