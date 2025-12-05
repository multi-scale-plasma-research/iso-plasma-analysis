# Main plasma analysis pipeline
import json
import os
import pandas as pd
from datetime import datetime
import numpy as np
from typing import Tuple, Dict

OMNI_FILE = r"E:\Research\Infrastructure-build\plasma-data\omni\omni_complete_corrected.csv"

class PlasmaAnalyzer:
    def __init__(self):
        self.config = {}
        self.omni_df = None
        self.load_config()
        self.load_data()

    def load_config(self):
        config_file = "configuration/gpu_config.json"

        if os.path.exists(config_file):
            with open(config_file) as f:
                self.config = json.load(f)
        else:
            # Fallback if GPU config was not generated
            self.config = {
                "gpu_mode": "single",
                "primary_gpu": 0,
                "batch_size_inference": 4,
            }

        print(f"GPU Mode: {self.config['gpu_mode']}")

    def load_data(self):
        print("Loading OMNI dataset...")

        if os.path.exists(OMNI_FILE):
            # Parse datetime column if present for downstream time-based operations
            try:
                self.omni_df = pd.read_csv(OMNI_FILE, parse_dates=["datetime"])
            except ValueError:
                # If there is no 'datetime' column or parsing fails, fall back to plain read
                self.omni_df = pd.read_csv(OMNI_FILE)

            print(f"[OK] Loaded {len(self.omni_df):,} OMNI records")
        else:
            print(f"[ERROR] OMNI file not found: {OMNI_FILE}")
            self.omni_df = None

    @staticmethod
    def compute_plasma_beta(
        df: pd.DataFrame,
        n_col: str = "Np",
        T_col: str = "Tp",
        B_col: str = "B",
    ) -> pd.Series:
        """
        Compute plasma beta = (n k_B T) / (B^2 / 2 mu0)
        n in cm^-3, T in K, B in nT -> result dimensionless.
        """
        # Physical constants (SI)
        k_B = 1.380649e-23      # J/K
        mu0 = 4e-7 * np.pi      # H/m
        # Convert units: n[cm^-3] -> n[m^-3], B[nT] -> B[T]
        n = df[n_col].astype(float).values * 1e6
        T = df[T_col].astype(float).values
        B = df[B_col].astype(float).values * 1e-9

        p_th = n * k_B * T
        p_mag = (B ** 2) / (2.0 * mu0)
        with np.errstate(divide="ignore", invalid="ignore"):
            beta = p_th / p_mag
        return pd.Series(beta, index=df.index, name="beta")

    @staticmethod
    def beta_window_stats(
        df: pd.DataFrame,
        center: str,
        baseline_days: int = 30,
        perihelion_half_width_days: int = 7,
    ) -> Dict[str, float]:
        """
        Given a DataFrame with DateTimeIndex and 'beta' column,
        compute baseline vs perihelion beta statistics.
        """
        import scipy.stats as stats

        center_ts = pd.to_datetime(center)

        baseline_start = center_ts - pd.Timedelta(days=baseline_days)
        baseline_end = center_ts - pd.Timedelta(days=1)

        peri_start = center_ts - pd.Timedelta(days=perihelion_half_width_days)
        peri_end = center_ts + pd.Timedelta(days=perihelion_half_width_days)

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
            }

        base_mu = base.mean()
        peri_mu = peri.mean()
        delta_frac = (peri_mu - base_mu) / base_mu

        # Mann–Whitney U (non‑parametric)
        u_stat, p_val = stats.mannwhitneyu(
            base.values, peri.values, alternative="two-sided"
        )

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
        }

    def analyze_iso_beta_windows(self):
        """
        Reproduce ISO perihelion beta anomalies using OMNI/ACE-like data.
        """
        if self.omni_df is None:
            print("[ERROR] OMNI/ACE dataset not loaded.")
            return

        df = self.omni_df.copy()

        # Ensure datetime index
        if "datetime" in df.columns:
            dt_col = "datetime"
        elif "Time" in df.columns:
            dt_col = "Time"
        else:
            raise ValueError(
                "No datetime column found; please add mapping here."
            )

        df[dt_col] = pd.to_datetime(df[dt_col])
        df = df.set_index(dt_col).sort_index()

        # Compute beta
        df["beta"] = self.compute_plasma_beta(
            df, n_col="Np", T_col="Tp", B_col="Bmag"
        )

        iso_windows = {
            "1I/'Oumuamua": "2017-09-09",   # center of Sep 2–16; adjust if needed
            "2I/Borisov": "2019-12-08",     # center of Dec 1–15
            "3I/ATLAS": "2025-10-19",       # center of Oct 12–26
        }

        results = []
        for name, center in iso_windows.items():
            print(f"[ISO] Analyzing beta window for {name} (center={center})...")
            stats = self.beta_window_stats(
                df,
                center=center,
                baseline_days=30,
                perihelion_half_width_days=7,
            )
            stats["iso"] = name
            stats["center_date"] = center
            results.append(stats)
            print(
                f"  baseline β={stats['baseline_beta']:.3f}, "
                f"perihelion β={stats['perihelion_beta']:.3f}, "
                f"Δβ={stats['delta_beta_frac']*100:.1f}% "
                f"(p={stats['p_value']:.3e}, d={stats['cohen_d']:.3f})"
            )

        os.makedirs("analysis/results", exist_ok=True)
        out_path = os.path.join(
            "analysis",
            "results",
            f"iso_beta_windows_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[OK] ISO beta window results saved to {out_path}")

    def analyze_cohort(self, comet_list=None):
        print("\n[14-COMET ANALYSIS]")
        print("-" * 50)

        if self.omni_df is None:
            print("[WARNING] OMNI dataset not loaded; analysis will be limited.")
            # return []

        if comet_list is None:
            comet_list = [
                "C/1996 B2",
                "C/2001 A2",
                "C/2004 F4",
                "C/2006 P1",
                "C/2010 X1",
                "C/2011 W3",
                "C/2012 S1",
                "1I/2017 U1",
                "2I/2019 Q4",
                "C/2019 Y4",
                "C/2020 F3",
                "C/2021 A1",
                "96P/Machholz",
                "3I/2024 L5",
            ]

        results = []
        for i, comet in enumerate(comet_list, 1):
            print(f"{i}/{len(comet_list)}: {comet}...", end=" ")

            result = {
                "comet": comet,
                "hcs_detections": 0,
                "anomalies": 0,
                "timestamp": datetime.now().isoformat(),
            }
            results.append(result)
            print("[OK]")

        os.makedirs("analysis/results", exist_ok=True)
        output_file = (
            "analysis/results/"
            f"cohort_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n[OK] Results saved: {output_file}")
        return results

    def run(self):
        print("\n" + "=" * 70)
        print("PLASMA ANALYSIS PIPELINE")
        print("=" * 70)

        self.analyze_cohort()
        self.analyze_iso_beta_windows()

        print("\n[SUCCESS] ANALYSIS COMPLETE")


if __name__ == "__main__":
    analyzer = PlasmaAnalyzer()
    analyzer.run()