import pandas as pd
import numpy as np

OMNI_CSV_PATH = r"E:\Research\Infrastructure-build\plasma-data\omni\omni_complete_corrected.csv"
ACE_SUMMARY_PATH = r"E:\Research\Instrument-Data\Plasma_Beta_Summary_Statistics.csv"

# Same center dates as analyze_iso_beta_windows
ISO_WINDOWS = {
    "1I/Oumuamua": "2017-09-09",   # center of Sep 2–16
    "2I/Borisov":  "2019-12-08",   # center of Dec 1–15
    "3I/ATLAS":    "2025-10-19",   # center of Oct 12–26
}

def compute_plasma_beta(df, n_col="Np", T_col="Tp", B_col="Bmag"):
    """
    Same as PlasmaAnalyzer.compute_plasma_beta, but simplified.
    n in cm^-3, T in K, B in nT -> beta dimensionless.
    """
    k_B = 1.380649e-23      # J/K
    mu0 = 4e-7 * np.pi      # H/m

    n = df[n_col].astype(float).values * 1e6      # cm^-3 -> m^-3
    T = df[T_col].astype(float).values
    B = df[B_col].astype(float).values * 1e-9     # nT -> T

    p_th = n * k_B * T
    p_mag = (B ** 2) / (2.0 * mu0)

    with np.errstate(divide="ignore", invalid="ignore"):
        beta = p_th / p_mag

    return pd.Series(beta, index=df.index, name="beta")

def beta_window_stats(df, center, baseline_days=30, perihelion_half_width_days=7):
    """
    Same logic as PlasmaAnalyzer.beta_window_stats.
    df must have DateTimeIndex and 'beta' column.
    """
    import scipy.stats as stats

    center_ts = pd.to_datetime(center)

    baseline_start = center_ts - pd.Timedelta(days=baseline_days)
    baseline_end   = center_ts - pd.Timedelta(days=1)

    peri_start = center_ts - pd.Timedelta(days=perihelion_half_width_days)
    peri_end   = center_ts + pd.Timedelta(days=perihelion_half_width_days)

    base = df.loc[baseline_start:baseline_end, "beta"].dropna()
    peri = df.loc[peri_start:peri_end, "beta"].dropna()

    # 3σ outlier rejection on baseline
    if len(base) > 0:
        mu = base.mean()
        sigma = base.std()
        base = base[(base > mu - 3 * sigma) & (base < mu + 3 * sigma)]

    if len(base) == 0 or len(peri) == 0:
        return {
            "baseline_beta": np.nan,
            "perihelion_beta": np.nan,
            "delta_beta_frac": np.nan,
            "p_value": np.nan,
            "cohen_d": np.nan,
            "n_base": int(len(base)),
            "n_peri": int(len(peri)),
        }

    base_mu = base.mean()
    peri_mu = peri.mean()
    delta_frac = (peri_mu - base_mu) / base_mu

    # Mann–Whitney U (two-sided)
    u_stat, p_val = stats.mannwhitneyu(base.values, peri.values, alternative="two-sided")

    # Cohen's d
    pooled_sigma = np.sqrt(
        ((len(base) - 1) * base.var() + (len(peri) - 1) * peri.var())
        / (len(base) + len(peri) - 2)
    )
    d = (peri_mu - base_mu) / pooled_sigma if pooled_sigma > 0 else np.nan

    return {
        "baseline_beta": float(base_mu),
        "perihelion_beta": float(peri_mu),
        "delta_beta_frac": float(delta_frac),
        "p_value": float(p_val),
        "cohen_d": float(d),
        "n_base": int(len(base)),
        "n_peri": int(len(peri)),
    }

def main():
    # Load ACE month-level medians
    ace = pd.read_csv(ACE_SUMMARY_PATH)

    # Load OMNI and prepare DateTimeIndex
    df = pd.read_csv(OMNI_CSV_PATH)
    time_col = "datetime" if "datetime" in df.columns else "time"
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col).sort_index()

    # Compute beta from OMNI
    df["beta"] = compute_plasma_beta(df, n_col="Np", T_col="Tp", B_col="Bmag")

    rows = []
    for iso_label, center in ISO_WINDOWS.items():
        # OMNI baseline vs perihelion using your exact window logic
        stats_ = beta_window_stats(
            df,
            center=center,
            baseline_days=30,
            perihelion_half_width_days=7,
        )

        # Find corresponding ACE summary row
        ace_row = ace.loc[ace["Object"] == iso_label].iloc[0]

        rows.append({
            "Object": iso_label,
            "center_date": center,
            "OMNI_baseline_beta": stats_["baseline_beta"],
            "OMNI_perihelion_beta": stats_["perihelion_beta"],
            "OMNI_delta_beta_frac": stats_["delta_beta_frac"],
            "OMNI_p_value": stats_["p_value"],
            "OMNI_cohen_d": stats_["cohen_d"],
            "OMNI_n_base": stats_["n_base"],
            "OMNI_n_peri": stats_["n_peri"],
            "ACE_median_beta_month": float(ace_row["Median_Beta"]),
            "ACE_classification": ace_row["Classification"],
            "ACE_sample_size": int(ace_row["Sample_Size"]),
        })

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()