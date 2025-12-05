import pandas as pd
import numpy as np

ACE_SUMMARY_PATH = r"E:\Research\Instrument-Data\Plasma_Beta_Summary_Statistics.csv"
OMNI_CSV_PATH = r"E:\Research\Infrastructure-build\plasma-data\omni\omni_complete_corrected.csv"

# Mapping from ACE "Period" to actual month ranges
ISO_MONTH_WINDOWS = {
    "1I/Oumuamua": ("2017-09-01", "2017-09-30"),
    "2I/Borisov":  ("2019-12-01", "2019-12-31"),
    "3I/ATLAS":    ("2025-10-01", "2025-10-31"),
}

def compute_beta_from_omni(df):
    # Assumes columns: Np [cm^-3], Tp [K], Bmag [nT]
    k_B = 1.380649e-23
    mu0 = 4e-7 * np.pi

    n = df["Np"].values * 1e6       # cm^-3 -> m^-3
    T = df["Tp"].values             # K
    B = df["Bmag"].values * 1e-9    # nT -> T

    p_th = n * k_B * T
    p_mag = B**2 / (2 * mu0)

    with np.errstate(divide="ignore", invalid="ignore"):
        beta = p_th / p_mag

    return beta

def main():
    # Load ACE summary
    ace = pd.read_csv(ACE_SUMMARY_PATH)

    # Load OMNI
    df = pd.read_csv(OMNI_CSV_PATH)
    # Make sure datetime column name matches your file (adjust if needed)
    time_col = "datetime" if "datetime" in df.columns else "time"
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col).sort_index()

    # Compute beta for OMNI
    df["beta"] = compute_beta_from_omni(df)

    rows = []
    for obj, (start, end) in ISO_MONTH_WINDOWS.items():
        ace_row = ace.loc[ace["Object"] == obj].iloc[0]

        # OMNI subset for the same calendar month
        omni_slice = df.loc[start:end, "beta"].dropna()
        omni_med = float(omni_slice.median()) if len(omni_slice) else np.nan

        rows.append({
            "Object": obj,
            "ACE_median_beta": float(ace_row["Median_Beta"]),
            "OMNI_median_beta": omni_med,
            "OMNI/ACE_ratio": float(omni_med / ace_row["Median_Beta"]) if ace_row["Median_Beta"] != 0 else np.nan,
            "ACE_classification": ace_row["Classification"],
            "ACE_sample_size": int(ace_row["Sample_Size"]),
            "OMNI_n": int(len(omni_slice)),
        })

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()