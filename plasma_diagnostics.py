# ============================================================================
# ISO Plasma Research: GPU-Accelerated Plasma Diagnostics (TIER 4 Parameters)
# ============================================================================
# Author: Chris (Independent Researcher)
# Description: Compute multi-scale plasma physics parameters from SPDF data
#              using CuPy for GPU acceleration on RTX 3090 Ti
# ============================================================================

import numpy as np
import cupy as cp
from pathlib import Path
import cdflib
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Physical constants (CGS units where applicable, SI for temperatures)
CONSTANTS = {
    'k_B': 1.38064852e-16,      # Boltzmann constant [erg/K]
    'm_p': 1.672621898e-24,     # Proton mass [g]
    'm_e': 9.10938356e-28,      # Electron mass [g]
    'c': 2.99792458e10,         # Speed of light [cm/s]
    'e': 4.803204257e-10,       # Elementary charge [esu]
    'mu_0': 1.25663706212e-6,   # Permeability [H/m] → [nT²·cm³/erg]
}

@dataclass
class PlasmaState:
    """Container for plasma state variables and derived parameters"""
    time: np.ndarray                    # Time array [UTC]
    B_gse: cp.ndarray                   # Magnetic field [nT] shape (N, 3)
    n_p: cp.ndarray                     # Proton density [cm^-3] shape (N,)
    T_p: cp.ndarray                     # Proton temperature [K] shape (N,)
    V_bulk: cp.ndarray                  # Bulk velocity [km/s] shape (N, 3)
    n_he_ratio: Optional[cp.ndarray]    # He/H density ratio shape (N,)
    
    # Derived TIER 4 parameters
    B_mag: Optional[cp.ndarray] = None
    B_gse_normalized: Optional[cp.ndarray] = None
    rho: Optional[cp.ndarray] = None
    P_thermal: Optional[cp.ndarray] = None
    P_magnetic: Optional[cp.ndarray] = None
    beta: Optional[cp.ndarray] = None
    Va: Optional[cp.ndarray] = None
    Cs: Optional[cp.ndarray] = None
    V_mag: Optional[cp.ndarray] = None
    Ma: Optional[cp.ndarray] = None
    gyroradius_p: Optional[cp.ndarray] = None
    gyrofreq_p: Optional[cp.ndarray] = None
    debye_length: Optional[cp.ndarray] = None
    mfp_p: Optional[cp.ndarray] = None
    reynolds_magnetic: Optional[cp.ndarray] = None
    dispersion_parameter: Optional[cp.ndarray] = None


class GPUPlasmaAnalyzer:
    """
    GPU-accelerated plasma parameter computation for multi-scale EM physics
    Designed for rapid analysis of SPDF solar wind data
    """
    
    def __init__(self, device: int = 0):
        """Initialize GPU device"""
        self.device = cp.cuda.Device(device)
        self.device.use()
        logger.info(f"GPU Device initialized: {device}")
        logger.info(f"GPU Compute Capability: {self.device.compute_capability}")
    
    def load_ace_mag(self, cdf_file: Path) -> Dict:
        """Load ACE MAG data from CDF file"""
        cdf = cdflib.CDF(str(cdf_file))
        
        data = {
            'time': cdf.varget('Epoch'),
            'B_gse': cdf.varget('BGSEc'),        # [Bx, By, Bz]
            'B_mag': cdf.varget('Magnitude'),
        }
        cdf.close()
        return data
    
    def load_ace_swepam(self, cdf_file: Path) -> Dict:
        """Load ACE SWEPAM plasma data from CDF file"""
        cdf = cdflib.CDF(str(cdf_file))
        
        data = {
            'time': cdf.varget('Epoch'),
            'n_p': cdf.varget('Np'),             # Proton density [cm^-3]
            'T_p': cdf.varget('Tpr'),            # Temperature [K]
            'V_x': cdf.varget('Vp_x'),           # Velocity components [km/s]
            'V_y': cdf.varget('Vp_y'),
            'V_z': cdf.varget('Vp_z'),
            'n_he_np': cdf.varget('He_qual'),    # He/H ratio (if available)
        }
        cdf.close()
        return data
    
    def align_datasets(self, mag_file: Path, plasma_file: Path) -> PlasmaState:
        """
        Align MAG and plasma measurements to common time grid
        """
        mag_data = self.load_ace_mag(mag_file)
        plasma_data = self.load_ace_swepam(plasma_file)
        
        # Align times (simple nearest-neighbor for now)
        # In production: use CubicSpline or interpolation
        mag_times = mag_data['time']
        plasma_times = plasma_data['time']
        
        # Find common time range
        t_start = max(mag_times[0], plasma_times[0])
        t_end = min(mag_times[-1], plasma_times[-1])
        
        # Select common indices
        mag_idx = (mag_times >= t_start) & (mag_times <= t_end)
        plasma_idx = (plasma_times >= t_start) & (plasma_times <= t_end)
        
        # Extract and move to GPU
        B_gse_gpu = cp.asarray(mag_data['B_gse'][mag_idx], dtype=cp.float32)
        n_p_gpu = cp.asarray(plasma_data['n_p'][plasma_idx], dtype=cp.float32)
        T_p_gpu = cp.asarray(plasma_data['T_p'][plasma_idx], dtype=cp.float32)
        
        V_bulk_gpu = cp.stack([
            cp.asarray(plasma_data['V_x'][plasma_idx], dtype=cp.float32),
            cp.asarray(plasma_data['V_y'][plasma_idx], dtype=cp.float32),
            cp.asarray(plasma_data['V_z'][plasma_idx], dtype=cp.float32),
        ], axis=1)  # Shape: (N, 3)
        
        state = PlasmaState(
            time=mag_times[mag_idx],
            B_gse=B_gse_gpu,
            n_p=n_p_gpu,
            T_p=T_p_gpu,
            V_bulk=V_bulk_gpu,
            n_he_ratio=None,  # Optional: load if available
        )
        
        return state
    
    def compute_derived_parameters(self, state: PlasmaState) -> PlasmaState:
        """Compute all TIER 4 derived plasma parameters on GPU"""
        
        logger.info("Computing derived plasma parameters on GPU...")
        
        # === BASIC DERIVED QUANTITIES ===
        
        # Magnetic field magnitude and normalized direction
        state.B_mag = cp.sqrt(cp.sum(state.B_gse**2, axis=1))
        state.B_gse_normalized = state.B_gse / state.B_mag[:, cp.newaxis]
        
        # Mass density [g/cm^3]
        # Account for He++ ions (approx: He density ~ 5% of H)
        state.rho = state.n_p * CONSTANTS['m_p'] * 1e-6  # Convert from cm^-3
        
        # === PRESSURE PARAMETERS ===
        
        # Thermal pressure [dyne/cm^2 = erg/cm^3]
        state.P_thermal = state.n_p * CONSTANTS['k_B'] * state.T_p
        
        # Magnetic pressure [dyne/cm^2]
        state.P_magnetic = (state.B_mag**2) / (8 * np.pi)
        
        # Plasma beta (thermal / magnetic pressure)
        state.beta = state.P_thermal / state.P_magnetic
        
        # === WAVE SPEEDS ===
        
        # Alfvén speed [km/s]
        # V_A = B / sqrt(4π ρ) in CGS
        Va_cgs = state.B_mag / cp.sqrt(4 * np.pi * state.rho)  # [cm/s]
        state.Va = Va_cgs * 1e-5  # Convert to [km/s]
        
        # Sound speed [km/s]
        gamma = 5.0 / 3.0  # Adiabatic index for monatomic gas
        Cs_cgs = cp.sqrt(gamma * state.P_thermal / state.rho)  # [cm/s]
        state.Cs = Cs_cgs * 1e-5  # Convert to [km/s]
        
        # === BULK FLOW PARAMETERS ===
        
        # Velocity magnitude [km/s]
        state.V_mag = cp.sqrt(cp.sum(state.V_bulk**2, axis=1))
        
        # Mach number (sonic)
        state.Ma = state.V_mag / state.Cs
        
        # === KINETIC SCALE PARAMETERS ===
        
        # Proton gyroradius [km]
        # r_L = m_p v_perp / (e B) in CGS
        v_perp = state.V_mag  # Simplified: use total velocity
        r_L_cm = (CONSTANTS['m_p'] * v_perp * 1e5) / (CONSTANTS['e'] * state.B_mag * 1e-5)
        state.gyroradius_p = r_L_cm * 1e-5  # Convert to [km]
        
        # Proton gyrofrequency [rad/s]
        # Omega_c = e B / m_p
        state.gyrofreq_p = (CONSTANTS['e'] * state.B_mag * 1e-5) / CONSTANTS['m_p']
        
        # Debye length [km]
        # λ_D = sqrt(k_B T / (4π e^2 n))
        debye_cm = cp.sqrt(
            (CONSTANTS['k_B'] * state.T_p) / (4 * np.pi * CONSTANTS['e']**2 * state.n_p)
        )
        state.debye_length = debye_cm * 1e-5  # Convert to [km]
        
        # === COLLISIONAL PARAMETERS ===
        
        # Proton mean free path [km]
        # λ_mfp ≈ v_th / (σ n)
        v_th = cp.sqrt(CONSTANTS['k_B'] * state.T_p / CONSTANTS['m_p'])  # [cm/s]
        sigma = 1e-15  # Cross-section [cm^2] (order of magnitude)
        mfp_cm = v_th / (sigma * state.n_p)
        state.mfp_p = mfp_cm * 1e-5  # Convert to [km]
        
        # === DIMENSIONLESS PARAMETERS ===
        
        # Magnetic Reynolds number
        # Re_B ≈ (v × L) / (η) → v L B / (m_p v_th)
        L_scale = 1.0  # Characteristic scale [AU] - set appropriately
        state.reynolds_magnetic = (state.V_mag * L_scale * state.B_mag) / state.Va
        
        # Dispersion relation parameter (for wave analysis)
        # κ = k λ_D (wave number × Debye length)
        state.dispersion_parameter = state.debye_length / (state.gyroradius_p + 1e-10)
        
        logger.info("✓ All TIER 4 parameters computed")
        
        return state
    
    def get_summary_statistics(self, state: PlasmaState) -> Dict:
        """Compute summary statistics for a plasma state"""
        
        stats = {
            'B_mag_mean': float(cp.mean(state.B_mag)),
            'B_mag_std': float(cp.std(state.B_mag)),
            'B_mag_min': float(cp.min(state.B_mag)),
            'B_mag_max': float(cp.max(state.B_mag)),
            
            'n_p_mean': float(cp.mean(state.n_p)),
            'n_p_std': float(cp.std(state.n_p)),
            
            'T_p_mean': float(cp.mean(state.T_p)),
            'T_p_std': float(cp.std(state.T_p)),
            
            'V_mag_mean': float(cp.mean(state.V_mag)),
            'V_mag_std': float(cp.std(state.V_mag)),
            
            'beta_mean': float(cp.mean(state.beta)),
            'beta_median': float(cp.median(state.beta)),
            'beta_percentile_95': float(cp.percentile(state.beta, 95)),
            
            'Va_mean': float(cp.mean(state.Va)),
            'Cs_mean': float(cp.mean(state.Cs)),
            'Ma_mean': float(cp.mean(state.Ma)),
            
            'gyroradius_p_mean': float(cp.mean(state.gyroradius_p)),
            'debye_length_mean': float(cp.mean(state.debye_length)),
        }
        
        return stats
    
    def export_to_arrays(self, state: PlasmaState) -> Dict:
        """Convert GPU arrays back to CPU numpy arrays for storage/visualization"""
        
        arrays = {
            'time': state.time,
            'B_gse': cp.asnumpy(state.B_gse),
            'B_mag': cp.asnumpy(state.B_mag),
            'n_p': cp.asnumpy(state.n_p),
            'T_p': cp.asnumpy(state.T_p),
            'V_bulk': cp.asnumpy(state.V_bulk),
            'V_mag': cp.asnumpy(state.V_mag),
            'rho': cp.asnumpy(state.rho),
            'P_thermal': cp.asnumpy(state.P_thermal),
            'P_magnetic': cp.asnumpy(state.P_magnetic),
            'beta': cp.asnumpy(state.beta),
            'Va': cp.asnumpy(state.Va),
            'Cs': cp.asnumpy(state.Cs),
            'Ma': cp.asnumpy(state.Ma),
            'gyroradius_p': cp.asnumpy(state.gyroradius_p),
            'gyrofreq_p': cp.asnumpy(state.gyrofreq_p),
            'debye_length': cp.asnumpy(state.debye_length),
            'mfp_p': cp.asnumpy(state.mfp_p),
            'reynolds_magnetic': cp.asnumpy(state.reynolds_magnetic),
            'dispersion_parameter': cp.asnumpy(state.dispersion_parameter),
        }
        
        return arrays


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
    # Initialize analyzer
    analyzer = GPUPlasmaAnalyzer(device=0)
    
    # Load data for a single comet window
    # Example: Oumuamua baseline (2024-08-09 to 2024-08-23)
    
    mag_file = Path('./comet_plasma_data/ace_mag/2024/ac_h0_mfi_20240809_v07.cdf')
    plasma_file = Path('./comet_plasma_data/ace_swepam/2024/ac_h0_swe_20240809_v07.cdf')
    
    if mag_file.exists() and plasma_file.exists():
        # Align and load data
        state = analyzer.align_datasets(mag_file, plasma_file)
        
        # Compute all TIER 4 parameters
        state = analyzer.compute_derived_parameters(state)
        
        # Get summary statistics
        stats = analyzer.get_summary_statistics(state)
        
        print("\n" + "="*70)
        print("PLASMA DIAGNOSTICS SUMMARY (GPU-Accelerated)")
        print("="*70)
        for key, value in stats.items():
            print(f"  {key:25} = {value:.6e}" if isinstance(value, float) else f"  {key:25} = {value}")
        
        # Export for storage
        arrays = analyzer.export_to_arrays(state)
        print(f"\n✓ Exported {len(arrays)} arrays to CPU memory")
        print(f"  Total size: {sum(v.nbytes for v in arrays.values() if hasattr(v, 'nbytes'))/1e6:.1f} MB")
    
    else:
        print("⚠ CDF files not found. Ensure SPDF data is downloaded first.")
