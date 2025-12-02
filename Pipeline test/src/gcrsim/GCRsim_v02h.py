import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib.resources import files
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


def mean_excitation_energy_HgCdTe(x):
    """
    Calculate the mean excitation energy for Hg_(1−x)Cd_(x)Te using Bragg's sum rule.

    Parameters
    ----------
    x : float
        Fractional composition of Cd in the alloy (0 ≤ x ≤ 1).
        The Hg fraction is automatically taken as (1 − x).

    Returns
    -------
    float
        Effective mean excitation energy of the compound in electronvolts (eV),
        computed via Bragg’s logarithmic rule from the elemental contributions.
    """
    # Mean excitation energies for the elements (in eV), data taken from https://physics.nist.gov/PhysRefData/Star/Text/method.html
    I_Hg = 800.0  # in eV, from NIST: https://pml.nist.gov/cgi-bin/Star/compos.pl?matno=080
    I_Cd = 469.0  # in eV, from NIST: https://pml.nist.gov/cgi-bin/Star/compos.pl?matno=048
    I_Te = 485.0  # in eV, from NIST: https://pml.nist.gov/cgi-bin/Star/compos.pl?matno=052

    # Atomic numbers (number of electrons per atom)
    Z_Hg = 80
    Z_Cd = 48
    Z_Te = 52

    # Electrons contributed by each element in the formula unit Hg_(1-x)Cd_(x)Te
    electrons_Hg = (1 - x) * Z_Hg
    electrons_Cd = x * Z_Cd
    electrons_Te = Z_Te

    # Total number of electrons in the formula unit
    total_electrons = electrons_Hg + electrons_Cd + electrons_Te

    # Weighting factors based on electron contribution
    w_Hg = electrons_Hg / total_electrons
    w_Cd = electrons_Cd / total_electrons
    w_Te = electrons_Te / total_electrons

    # Compute the logarithmic average (Bragg's rule):
    lnI = w_Hg * np.log(I_Hg) + w_Cd * np.log(I_Cd) + w_Te * np.log(I_Te)
    I_compound = np.exp(lnI)

    return I_compound


def radiation_length_HgCdTe(x):
    """
    Compute the radiation length (in g/cm^2) for Hg(1-x)Cd(x)Te.

    Uses the PDG approximate formula for the radiation length of an element:

        X0 = 716.4 * A / (Z*(Z+1)*ln(287/sqrt(Z)))   [g/cm^2]

    and for a compound:

        1/X0_compound = sum_i (w_i / X0_i)

    where w_i = (N_i * A_i) / (sum_j N_j * A_j) are the weight fractions.

    Parameters
    ----------
    x : float
        Molar fraction of Cd (and thus Hg molar fraction is 1-x).

    Returns
    -------
    X0_compound : float
        Radiation length of the compound in g/cm^2.
    """
    # Atomic numbers and atomic masses (g/mol) for each element:
    # Mercury (Hg)
    Z_Hg = 80
    A_Hg = 200.59
    # Cadmium (Cd)
    Z_Cd = 48
    A_Cd = 112.41
    # Tellurium (Te)
    Z_Te = 52
    A_Te = 127.60

    # Helper function: Radiation length for an element (in g/cm^2)
    def X0_element(Z, A):
        return 716.4 * A / (Z * (Z + 1) * np.log(287 / np.sqrt(Z)))

    # Compute radiation lengths for individual elements:
    X0_Hg = X0_element(Z_Hg, A_Hg)
    X0_Cd = X0_element(Z_Cd, A_Cd)
    X0_Te = X0_element(Z_Te, A_Te)

    # Molar amounts: Hg: (1-x), Cd: x, Te: 1.
    # Total molar mass of the compound:
    A_tot = (1 - x) * A_Hg + x * A_Cd + A_Te

    # Weight fractions:
    w_Hg = (1 - x) * A_Hg / A_tot
    w_Cd = x * A_Cd / A_tot
    w_Te = A_Te / A_tot

    # Radiation length of the compound (in g/cm^2):
    X0_compound = 1.0 / (w_Hg / X0_Hg + w_Cd / X0_Cd + w_Te / X0_Te)

    return X0_compound


def density_HgCdTe(x):
    """
    Compute the density (in g/cm^3) of Hg(1-x)Cd(x)Te.

    Assumes:
      - Formula unit: 1 cation (Hg with fraction 1-x or Cd with fraction x) + 1 Te.
      - Zincblende crystal structure (4 formula units per unit cell).
      - Vegard's law for the lattice constant.

    Parameters
    ----------
    x : float
        Molar fraction of Cd (and thus Hg molar fraction is 1-x).

    Returns
    -------
    density : float
        Density of Hg(1-x)Cd(x)Te in g/cm^3.
    """
    # Atomic masses in g/mol
    A_Hg = 200.59  # Mercury
    A_Cd = 112.41  # Cadmium
    A_Te = 127.60  # Tellurium

    # Molar mass of the compound (g/mol)
    M = (1 - x) * A_Hg + x * A_Cd + A_Te

    # Lattice constants (in cm) - 1 Å = 1e-8 cm
    a_HgTe = 6.46e-8  # HgTe lattice constant in cm
    a_CdTe = 6.48e-8  # CdTe lattice constant in cm

    # Vegard's law: linear interpolation of lattice constant
    a = (1 - x) * a_HgTe + x * a_CdTe

    # For zincblende structure: 4 formula units per unit cell
    # Volume per formula unit = a^3 / 4
    volume_per_formula = a**3 / 4

    # Avogadro's number (mol^-1)
    N_A = 6.02214076e23

    # Mass per formula unit in grams
    mass_per_formula = M / N_A

    # Density in g/cm^3
    density = mass_per_formula / volume_per_formula

    return density # g/cm^3


def mean_Z_A_HgCdTe(x):
    """
    Compute the number-averaged mean atomic number (Z) and atomic mass (A)
    for Hg(1-x)Cd(x)Te in a zincblende structure.

    Parameters:
    -----------
    x : float
        Molar fraction of Cd (and hence Hg fraction is 1-x).

    Returns:
    --------
    Z_mean : float
        Number-averaged mean atomic number.
    A_mean : float
        Number-averaged mean atomic mass (in g/mol).
    """
    # Atomic numbers
    Z_Hg = 80
    Z_Cd = 48
    Z_Te = 52

    # Atomic masses in g/mol
    A_Hg = 200.59
    A_Cd = 112.41
    A_Te = 127.60

    # There are (1-x) moles of Hg, x moles of Cd, and 1 mole of Te per formula unit.
    # Total number of atoms per formula unit = (1-x) + x + 1 = 2.
    total_atoms = 2

    Z_mean = ((1 - x) * Z_Hg + x * Z_Cd + Z_Te) / total_atoms
    A_mean = ((1 - x) * A_Hg + x * A_Cd + A_Te) / total_atoms

    return Z_mean, A_mean # Z_mean is unitless, A_mean has units of g/mol


# Compute material properties for Hg0.555Cd0.445Te (x = 0.445)
x = 0.445  # molar fraction of Cd in Hg(1-x)Cd(x)Te
I_value = mean_excitation_energy_HgCdTe(x) # eV
I_value_MeV = I_value * (1e-6) # MeV
X0_gPercmSqd = radiation_length_HgCdTe(x) # g/cm^2
HgCdTe_density = density_HgCdTe(x) # g/cm^3
X0_cm = X0_gPercmSqd / HgCdTe_density # cm
Z_mean, A_mean = mean_Z_A_HgCdTe(x) # Z_mean is unitless, A_mean has units of g/mol

color_list = []
#path = files("gcrsim").joinpath("data/rgb_color_list.txt")
path = "rgb_color_list.txt" #only use when script is local, else uncomment above
with open(path, "r") as file:
    for line in file:
        line = line.strip()  # Remove leading/trailing whitespace
        if not line or line.startswith("#"):
            continue  # Skip blank lines or comment lines
        # Split the line by tab. Adjust the separator if needed.
        parts = line.split("\t")
        if len(parts) < 2:
            continue  # Skip lines that don't have enough parts
        color_name = parts[0].strip()
        hex_code = parts[1].strip()
        color_list.append((color_name, hex_code))

# Reading in sunspot data to compute ISO parameters and rigidity spectrum
# Sunspot data downloaded from https://www.sidc.be/SILSO/datafiles
#csv_path = files("gcrsim").joinpath("data/SN_m_tot_V2.0.csv")
csv_path = "SN_m_tot_V2.0.csv" #only use when script is local, else uncomment above
month_df = pd.read_csv(csv_path, sep=";", engine="python")

# Contents:
# Column 1-2: Gregorian calendar date, 1.Year, 2.Month
# Column 3: Date in fraction of year for the middle of the corresponding month
# Column 4: Monthly mean total sunspot number, W = Ns + 10 * Ng, with Ns the number of spots
# and Ng the number of groups counted over the entire solar disk
# Column 5: Monthly mean standard deviation of the input sunspot numbers from individual stations.
# Column 6: Number of observations used to compute the monthly mean total sunspot number.
# Column 7: Definitive/provisional marker.

month_df.columns = ["year", "month", "date", "mean", "std_dev", "num_obs", "marker"]

frac_amounts = np.linspace((1/12)-(1/24),(12/12)-(1/24),12)
t_plus = 15 # months
delta_w_t = 16 # months

# IF USING SMOOTHED DATA INSTEAD, USE THE FOLLOWING BLOCK:-----
# month_df=month_s_df
# month_df=month_df[:-7]
# -------------------------------------------------------------

# Filter the dataframe to include only dates starting at 1986.707
month_df = month_df[month_df["date"] >= 1986.707].copy()

# Initialize 'solar_cycle' and update according to date ranges:
month_df["solar_cycle"] = 22
month_df.loc[(month_df["date"] >= 1996.624) & (month_df["date"] <= 2008.874), "solar_cycle"] = 23
month_df.loc[(month_df["date"] >= 2008.958) & (month_df["date"] <= 2019.873), "solar_cycle"] = 24
month_df.loc[month_df["date"] >= 2019.958, "solar_cycle"] = 25

# Define the dates where the solar cycle changes
cycle_change_dates = [1996.624, 2008.958, 2019.958]
cycle_labels = ["Cycle 23 starts", "Cycle 24 starts", "Cycle 25 starts"]

# For each solar cycle, find the row with the maximum and minimum 'mean'
cycle_max = month_df.loc[month_df.groupby("solar_cycle")["mean"].idxmax()]
cycle_min = month_df.loc[month_df.groupby("solar_cycle")["mean"].idxmin()]

month_df["cycle_max"] = month_df.groupby("solar_cycle")["mean"].transform("max")
month_df["cycle_min"] = month_df.groupby("solar_cycle")["mean"].transform("min")

# Group the dataframe by 'solar_cycle' and find the index of the row with the maximum 'mean' for each group
cycle_max_df = month_df.loc[month_df.groupby("solar_cycle")["mean"].idxmax()]
# First, extract the sign reversal moments by finding, for each solar cycle,
# the date at which the 'mean' is maximum.
cycle_max_df = month_df.loc[month_df.groupby("solar_cycle")["mean"].idxmax()]

# Create a mapping: solar_cycle -> sign reversal moment (date)
sign_reversal_dict = cycle_max_df.set_index("solar_cycle")["date"].to_dict()


def compute_M(target_date, df, sign_reversal_dict, tol=3e-2):
    """
    Compute the magnetic modulation term M for a given target date based on solar cycle data.

    Given a target date and a DataFrame containing a `'date'` column and solar cycle
    values, this function finds the entry whose `'date'` is within a specified tolerance
    of `target_date`, then computes:
    M = S * (-1)^(solar_cycle - 1) * ((mean - cycle_min) / (cycle_max - cycle_min))^2.7
    where:
    S = 1  if (target_date - sign_reversal_date) ≥ 0
    S = -1 otherwise

    Parameters
    ----------
    target_date : float
        Target date expressed as a fractional year to search for in the dataset.
    df : pandas.DataFrame
        DataFrame containing at least a `'date'` column and solar cycle data columns.
    sign_reversal_dict : dict
        Mapping of each solar cycle number to its sign reversal date.
    tol : float
        Allowed tolerance for matching the target date (in the same units as `target_date`).

    Returns
    -------
    float
        Computed magnetic modulation value `M` for the matching entry.

    Raises
    ------
    ValueError
        If no entry is found within the specified tolerance of `target_date`.
    """

    # Find the row whose 'date' is closest to target_date
    diff = np.abs(df["date"] - target_date)
    if diff.min() > tol:
        raise ValueError(f"No entry found for date {target_date} within tolerance {tol}.")
    idx = diff.idxmin()
    row = df.loc[idx]

    # Extract values from the row
    solar_cycle = row["solar_cycle"]
    mean_val = row["mean"]
    cycle_max_val = row["cycle_max"]
    cycle_min_val = row["cycle_min"]

    # Check for division by zero
    if cycle_max_val == cycle_min_val:
        raise ValueError("cycle_max and cycle_min are equal; cannot compute fraction.")

    fraction = (mean_val - cycle_min_val) / (cycle_max_val - cycle_min_val)

    # Compute the sign factor from solar_cycle
    factor = (-1) ** (int(solar_cycle) - 1)

    # Compute S based on the target_date relative to the sign reversal moment for that cycle
    sign_reversal = sign_reversal_dict[solar_cycle]
    S = 1 if (target_date - sign_reversal) >= 0 else -1 # unitless

    # Compute M using the modified formula
    M_value = S * factor * (fraction**2.7)
    return M_value # unitless


# Now, apply compute_M over the dataframe.
# For each row (using its 'date'), compute the corresponding M_value.
month_df["M_value"] = month_df["date"].apply(lambda d: compute_M(d, month_df, sign_reversal_dict, tol=1e-2))


class CosmicRaySimulation:
    """
    Simulate galactic cosmic ray interactions in a pixelated HgCdTe detector.

    The class holds species catalogs and material properties, and provides methods to
    compute the time-dependent modulation term, sample primary spectra, propagate
    particles through a pixel grid, and generate energy-loss maps / event catalogs.
    """

    # Class-level lists for species (charge and mass)
    Z_list = [-1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
              21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 
              54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 90, 92] # Omitting z=84-89 and z = 91 due to short half-lives
    m_list = [5.109989461e5, 0.9382720813e9, 2*(0.9382720813e9) + 2*(0.9395654133e9),3*(0.9382720813e9) + 4*(0.9395654133e9), 
              4*(0.9382720813e9) + 5*(0.9395654133e9), 5*(0.9382720813e9) + 6*(0.9395654133e9), 6*(0.9382720813e9) + 6*(0.9395654133e9),
              7*(0.9382720813e9) + 7*(0.9395654133e9), 8*(0.9382720813e9) + 8*(0.9395654133e9), 9*(0.9382720813e9) + 10*(0.9395654133e9), 
              10*(0.9382720813e9) + 10*(0.9395654133e9), 11*(0.9382720813e9) + 12*(0.9395654133e9), 12*(0.9382720813e9) + 12*(0.9395654133e9), 
              13*(0.9382720813e9) + 14*(0.9395654133e9), 14*(0.9382720813e9) + 14*(0.9395654133e9), 15*(0.9382720813e9) + 16*(0.9395654133e9), 
              16*(0.9382720813e9) + 16*(0.9395654133e9), 17*(0.9382720813e9) + 18*(0.9395654133e9), 18*(0.9382720813e9) + 22*(0.9395654133e9),
              19*(0.9382720813e9) + 20*(0.9395654133e9), 20*(0.9382720813e9) + 20*(0.9395654133e9), 21*(0.9382720813e9) + 24*(0.9395654133e9),
              22*(0.9382720813e9) + 26*(0.9395654133e9), 23*(0.9382720813e9) + 28*(0.9395654133e9), 24*(0.9382720813e9) + 28*(0.9395654133e9),
              25*(0.9382720813e9) + 30*(0.9395654133e9), 26*(0.9382720813e9) + 30*(0.9395654133e9), 27*(0.9382720813e9) + 32*(0.9395654133e9), 
              28*(0.9382720813e9) + 30*(0.9395654133e9), 29*(0.9382720813e9) + 34*(0.9395654133e9), 30*(0.9382720813e9) + 34*(0.9395654133e9),
              31*(0.9382720813e9) + 38*(0.9395654133e9), 32*(0.9382720813e9) + 42*(0.9395654133e9), 33*(0.9382720813e9) + 42*(0.9395654133e9),
              34*(0.9382720813e9) + 46*(0.9395654133e9), 35*(0.9382720813e9) + 44*(0.9395654133e9), 36*(0.9382720813e9) + 48*(0.9395654133e9),
              37*(0.9382720813e9) + 48*(0.9395654133e9), 38*(0.9382720813e9) + 50*(0.9395654133e9), 39*(0.9382720813e9) + 50*(0.9395654133e9),
              40*(0.9382720813e9) + 50*(0.9395654133e9), 41*(0.9382720813e9) + 52*(0.9395654133e9), 42*(0.9382720813e9) + 56*(0.9395654133e9),
              43*(0.9382720813e9) + 54*(0.9395654133e9), 44*(0.9382720813e9) + 58*(0.9395654133e9), 45*(0.9382720813e9) + 58*(0.9395654133e9), 
              46*(0.9382720813e9) + 60*(0.9395654133e9), 47*(0.9382720813e9) + 60*(0.9395654133e9), 48*(0.9382720813e9) + 66*(0.9395654133e9),
              49*(0.9382720813e9) + 69*(0.9395654133e9), 50*(0.9382720813e9) + 69*(0.9395654133e9), 51*(0.9382720813e9) + 70*(0.9395654133e9),
              52*(0.9382720813e9) + 78*(0.9395654133e9), 53*(0.9382720813e9) + 74*(0.9395654133e9), 54*(0.9382720813e9) + 78*(0.9395654133e9),
              55*(0.9382720813e9) + 78*(0.9395654133e9), 56*(0.9382720813e9) + 82*(0.9395654133e9), 57*(0.9382720813e9) + 82*(0.9395654133e9),
              58*(0.9382720813e9) + 82*(0.9395654133e9), 59*(0.9382720813e9) + 82*(0.9395654133e9), 60*(0.9382720813e9) + 82*(0.9395654133e9),
              61*(0.9382720813e9) + 83*(0.9395654133e9), 62*(0.9382720813e9) + 90*(0.9395654133e9), 63*(0.9382720813e9) + 90*(0.9395654133e9),
              64*(0.9382720813e9) + 94*(0.9395654133e9), 65*(0.9382720813e9) + 94*(0.9395654133e9), 66*(0.9382720813e9) + 98*(0.9395654133e9),
              67*(0.9382720813e9) + 98*(0.9395654133e9), 68*(0.9382720813e9) + 98*(0.9395654133e9), 69*(0.9382720813e9) + 100*(0.9395654133e9),
              70*(0.9382720813e9) + 104*(0.9395654133e9), 71*(0.9382720813e9) + 104*(0.9395654133e9), 72*(0.9382720813e9) + 108*(0.9395654133e9),
              73*(0.9382720813e9) + 108*(0.9395654133e9), 74*(0.9382720813e9) + 112*(0.9395654133e9), 75*(0.9382720813e9) + 112*(0.9395654133e9),
              76*(0.9382720813e9) + 116*(0.9395654133e9), 77*(0.9382720813e9) + 116*(0.9395654133e9), 78*(0.9382720813e9) + 116*(0.9395654133e9),
              79*(0.9382720813e9) + 118*(0.9395654133e9), 80*(0.9382720813e9) + 122*(0.9395654133e9), 81*(0.9382720813e9) + 124*(0.9395654133e9),
              82*(0.9382720813e9) + 126*(0.9395654133e9), 83*(0.9382720813e9) + 126*(0.9395654133e9), 90*(0.9382720813e9) + 142*(0.9395654133e9),
              92*(0.9382720813e9) + 146*(0.9395654133e9) ] # masses in eV/nucleon (except for z<2) 
    A_list = [ 1.0, 1.0, (4.0), (6.9), (9.0), (10.8), (12.0), (14.0), (16.0), (19.0),
             (20.2), (23.0), (24.34), (27.0), (28.1), (31.0), (32.1), (35.4),
             (39.9), (39.1), (40.1), (44.9), (47.9), (50.9), (52.0), (54.9),
             (55.8), (58.9), (58.7), (63.5), (65.4), (69.7), (72.6), (74.9),
             (79.0), (79.9), (83.8), (85.5), (87.6), (88.9), (91.2), (92.9), 
             (95.9), (97.0), (101.0), (102.9), (106.4), (107.9), (112.4), (114.8), 
             (118.7), (121.8), (127.60), (126.9), (131.3), (132.9), (137.3), (138.9), 
             (140.1), (140.9), (144.2), (144.2), (145.0), (150.4), (152.0), (157.3), 
             (158.9), (162.5), (164.9), (167.3), (168.9), (173.0), (175.0), (178.5),
             (180.9), (183.9), (186.2), (190.2), (192.2), (195.1), (197.0), (200.6), 
             (204.4), (207.2), (232.0), (238.0) ] #unitless (mass number) (analogous to num of nucleons/particle)
    C_list = [ 170, 1.85e4, 3.69e3, 19.5, 17.7, 49.2, 103.0, 36.7, 87.4, 3.19, 16.4, 4.43, 19.3, 4.17, 13.4, 1.15, 3.06, 1.30,
             2.33, 1.87, 2.17, 0.74, 2.63, 1.23, 2.12, 1.14, 9.32, 0.10, 0.49,
             (9.32 * 6.8e-4), (9.32 * 8.8e-4), (9.32 * 6.5e-5), (9.32 * 1.4e-4), (9.32 * 8.9e-6), (9.32 * 5.2e-5), (9.32 * 9.7e-6), 
             (9.32 * 2.7e-5), (9.32 * 8.8e-6), (9.32 * 2.9e-5), (9.32 * 6.5e-6), (9.32 * 1.6e-5), (9.32 * 2.9e-6), (9.32 * 8.1e-6), 
             (9.32 * 9.5e-7), (9.32 * 3.1e-6), (9.32 * 1.6e-6), (9.32 * 4.6e-6), (9.32 * 1.5e-6), (9.32 * 4.0e-6), (9.32 * 8.8e-7), 
             (9.32 * 4.7e-6), (9.32 * 9.9e-7), (9.32 * 5.7e-6), (9.32 * 1.1e-6), (9.32 * 2.7e-6), (9.32 * 6.5e-7), (9.32 * 6.7e-7), 
             (9.32 * 6.0e-7), (9.32 * 1.8e-6), (9.32 * 4.3e-7), (9.32 * 1.6e-6), (9.32 * 1.9e-7), (9.32 * 1.8e-6), (9.32 * 3.1e-7), 
             (9.32 * 1.4e-6), (9.32 * 3.5e-7), (9.32 * 1.4e-6), (9.32 * 5.3e-7), (9.32 * 8.8e-7), (9.32 * 1.8e-7), (9.32 * 8.9e-7), 
             (9.32 * 1.3e-7), (9.32 * 8.1e-7), (9.32 * 7.3e-8), (9.32 * 8.1e-7), (9.32 * 2.8e-7), (9.32 * 1.2e-6), (9.32 * 7.9e-7), 
             (9.32 * 1.5e-6), (9.32 * 2.8e-7), (9.32 * 4.9e-7), (9.32 * 1.5e-7), (9.32 * 1.4e-6), (9.32 * 7.3e-8), (9.32 * 8.1e-8), (9.32 * 4.9e-8) ] #normalization, unitless
    alpha_list = [ 1, 2.85, 3.12, 3.41, 4.30, 3.93, 3.18, 3.77, 3.11, 4.05, 3.11, 3.14, 3.65, 3.46, 3.00,
            4.04, 3.30, 4.40, 4.33, 4.49, 2.93, 3.78, 3.79, 3.50, 3.28, 3.29, 3.01, 4.25, 3.52, 
            3.01, 3.01, 3.01, 3.01, 3.01, 3.01, 3.01, 3.01, 3.01, 3.01, 3.01, 3.01,
            3.01, 3.01, 3.01, 3.01, 3.01, 3.01, 3.01, 3.01, 3.01, 3.01, 3.01, 3.01,
            3.01, 3.01, 3.01, 3.01, 3.01, 3.01, 3.01, 3.01, 3.01, 3.01, 3.01, 3.01,
            3.01, 3.01, 3.01, 3.01, 3.01, 3.01, 3.01, 3.01, 3.01, 3.01, 3.01, 3.01,
            3.01, 3.01, 3.01, 3.01, 3.01, 3.01, 3.01, 3.01, 3.01 ] # unitless
    gamma_values_list = [ 2.74, 2.77, 2.82, 3.05, 2.96, 2.76, 2.89, 2.70, 2.82, 2.76, 2.84, 2.70, 2.77, 2.66, 2.89,
            2.71, 3.00, 2.93, 3.05, 2.77, 2.97, 2.99, 2.94, 2.89, 2.74, 2.63, 2.63, 2.63,
            2.63, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63,
            2.63, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63,
            2.63, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63,
            2.63, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63,
            2.63, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63] # unitless (this list is 1 element shorter than the others [85 vs 86] to account for gamma_func() )
    species_names_list = ["e", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", 
        "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", 
        "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co",  "Ni", "Cu", 
        "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", 
        "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", 
        "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", 
        "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", 
        "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", 
        "Hg", "Tl", "Pb", "Bi", "Th", "U"]  
    
    species_names_dict = {0: "e", 1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne", 11: "Na",
        12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca", 21: "Sc", 22: "Ti", 23: "V", 
        24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn", 31: "Ga", 32: "Ge", 33: "As", 34: "Se",
        35: "Br", 36: "Kr", 37: "Rb", 38: "Sr", 39: "Y", 40: "Zr", 41: "Nb", 42: "Mo", 43: "Tc", 44: "Ru", 45: "Rh", 46: "Pd",
        47: "Ag", 48: "Cd", 49: "In", 50: "Sn", 51: "Sb", 52: "Te", 53: "I", 54: "Xe", 55: "Cs", 56: "Ba", 57: "La", 58: "Ce",
        59: "Pr", 60: "Nd", 61: "Pm", 62: "Sm", 63: "Eu", 64: "Gd", 65: "Tb", 66: "Dy", 67: "Ho", 68: "Er", 69: "Tm", 70: "Yb",
        71: "Lu", 72: "Hf", 73: "Ta", 74: "W", 75: "Re", 76: "Os", 77: "Ir", 78: "Pt", 79: "Au", 80: "Hg", 81: "Tl", 82: "Pb",
        83: "Bi", 90: "Th", 92: "U"}
    
    frac_amounts = np.linspace((1/12)-(1/24),(12/12)-(1/24),12)
    t_plus = 15 # months
    delta_w_t = 16 # months

    def __init__(
        self,
        species_index=1,
        grid_size=64,
        cell_size=10, # um
        cell_depth=5, # um
        dt=3.04, # seconds
        step_size=0.1, # um
        material_Z=Z_mean, # unitless
        material_A=A_mean, # g/mol
        I0=I_value_MeV, # MeV
        material_density=HgCdTe_density, # g/cm^3
        X0=X0_cm, # cm
        color_list=color_list,
        date=2018.458, # years
        historic_df=month_df,
        progress_bar=False,
        max_workers=None,
        apply_padding: bool = True,
        pad_pixels: int = 4,
        pad_mode: str = "constant",
        pad_value: int | float = 0,
    ):
        """
        Initialize a cosmic-ray simulation on a pixelated detector.

        Parameters
        ----------
        species_index : int, default=1
            Index into the species lists (charge/mass/spectrum constants).
        grid_size : int, default=64
            Number of pixels per detector side (grid is ``grid_size × grid_size``).
        cell_size : float, default=10
            Pixel pitch (µm).
        cell_depth : float, default=5
            Pixel depth (µm).
        dt : float, default=3.04
            Time step used by the ISO flux model (s).
        step_size : float, default=0.1
            Propagation step length for charged particles (µm).
        material_Z : float, default=Z_mean
            Mean atomic number of the absorber material.
        material_A : float, default=A_mean
            Mean atomic mass (g/mol) of the absorber material.
        I0 : float, default=I_value_MeV
            Mean excitation energy of the absorber (MeV).
        material_density : float, default=HgCdTe_density
            Mass density of the absorber (g cm⁻³).
        X0 : float, default=X0_cm
            Radiation length of the absorber (cm).
        color_list : Sequence[tuple[str, str]], default=color_list
            Display colors for species, as ``(name, hex)`` tuples.
        date : float, default=2018.458
            Simulation date as fractional year; used to set the modulation term.
        historic_df : pandas.DataFrame or None, default=month_df
            Historical dataset used to compute the date-dependent modulation; if ``None``,
            a neutral value is used.
        progress_bar : bool, default=False
            Whether to show progress bars for long operations.
        max_workers : int or None, default=None
            Maximum number of worker threads/processes; if ``None``, a sensible default is used.
        apply_padding : bool, default=True
            Whether to pad arrays before convolution/propagation to reduce edge artifacts.
        pad_pixels : int, default=4
            Number of pixels to pad on each border when ``apply_padding`` is True.
        pad_mode : str, default="constant"
            Padding mode (passed to ``numpy.pad``), e.g., ``"constant"`` or ``"edge"``.
        pad_value : int or float, default=0
            Constant value to use when ``pad_mode="constant"``.

        Notes
        -----
        Units:
            * lengths are in µm for pixel geometry,
            * cm for material constants (e.g., ``X0``),
            * energies are in MeV unless otherwise stated.

        """
        self.Z_particle = self.Z_list[species_index] # unitless
        self.M = self.m_list[species_index] * self.A_list[species_index] * 1e-6  # Convert from eV/nucleon to MeV (eV to MeV for z<2) by accounting for # of nucleons and rescaling 
        self.species_index = species_index # unitless
        self.grid_size = grid_size # unitless
        self.cell_size = cell_size # um
        self.cell_depth = cell_depth # um
        self.step_size = step_size # um
        self.date = date # years
        self.historic_df = historic_df
        # Set M_polar based on the year and historical M data, if available.
        if historic_df is not None:
            self.M_polar = self.get_M_value(self.date, self.historic_df)
        else:
            self.M_polar = 1  # default value if no historical data is provided

        # Material properties (passed in from user)
        self.material_Z = material_Z # unitless
        self.material_A = material_A # g/mol
        self.I0 = I0 # MeV
        self.material_density = material_density # g/cm^3
        self.X0 = X0 # cm

        # Other simulation constants
        self.me = self.m_list[0] * self.A_list[0] * 1e-6  # Convert from eV/nucleon to MeV (eV to MeV for z<2) by accounting for # of nucleons and rescaling
        self.K = 0.307075  # MeV cm^2/mol
        self.c = 2.99792458e10  # Speed of light in cm/s

        # Energy range for primaries (in MeV)
        #self.E_min = 1e1 # MeV (should I have set this harsh boundary?)
        #self.E_max = 1e5 # MeV
        self.start_ISO_energy = 1e3  # eV, only affects month.df #changing from 10 MeV to 1keV to test 
        self.stop_ISO_energy = 1e13  # eV, only affects month.df #changing from 100 GeV to 10TeV to test 

        # ISO model parameters
        self.dt = dt  # seconds
        self.dA = 1e-10 * (self.grid_size) ** 2  # 10 microns in m^2, times number of pixels per microns
        self.dOmega = 2 * np.pi  # sr
        self.R_e = 1 # GV

        # padding parameters
        self.apply_padding = apply_padding
        self.pad_pixels = pad_pixels
        self.pad_mode = pad_mode
        self.pad_value = pad_value

        # Color list for particles
        self.color_list = color_list
        self.progress_bar = progress_bar
        self.max_workers = max_workers or 4  
        self._lock = threading.Lock()

        # Will hold the per-energy-bin ISO table for this species
        # Columns: Start/End/Bin Center/Bin Width/Mean # of particles
        self.num_part_table = None


    @classmethod
    def run_full_sim(
        cls,
        grid_size: int = 4088, #num of pixels
        progress_bar: bool = False,
        apply_padding: bool = True,
        pad_pixels: int = 4, #num of reference pixels on each SCA edge
        pad_mode: str = "constant",
        pad_value: int | float = 0,
        return_sims: bool = False,
        **sim_kwargs,  # keep this last so callers can still pass any other kwargs
    ):
        """
        Run a full sweep of `run_sim()` for every species in `Z_list`.

        Parameters
        ----------
        grid_size : int, default=4088
            Number of pixels per side of the detector grid for each per-species run.
        progress_bar : bool, default=False
            If True, show a progress bar while iterating over species.
        apply_padding : bool, default=True
            Forwarded to the constructor; if True, arrays are padded to mitigate edge effects.
        pad_pixels : int, default=4
            Number of pixels to pad on each border when ``apply_padding`` is True.
        pad_mode : str, default="constant"
            Padding mode (passed to ``numpy.pad``), e.g., ``"constant"`` or ``"edge"``.
        pad_value : int or float, default=0
            Constant value used when ``pad_mode="constant"``.
        **sim_kwargs
            Any additional ``CosmicRaySimulation.__init__`` keyword arguments
            (e.g., ``cell_size``, ``cell_depth``, ``step_size``, ``date``, etc.).

        Returns
        -------
        combined_heatmap : numpy.ndarray
            Placeholder combined heatmap (currently zeros with the shape of a single heatmap).
        heatmap_list : list[numpy.ndarray]
            Per-species heatmaps returned by each `run_sim()` call.
        streaks_list : list[list]
            Per-species lists of streak objects (or metadata) from each `run_sim()` call.
        gcr_counts : list[tuple[str, int]]
            Per-species counts as ``(species_name, count)`` tuples.
        """
        heatmap_list = []
        streaks_list = []
        gcr_counts = []
        sim_list = []

        for idx in tqdm(
            range(len(cls.Z_list)), desc="Running simulation for each species", disable=not progress_bar
        ):
            sim = cls(
                species_index=idx,
                grid_size=grid_size,
                progress_bar=progress_bar,
                apply_padding=apply_padding,
                pad_pixels=pad_pixels,
                pad_mode=pad_mode,
                pad_value=pad_value,
                **sim_kwargs,
            )

            heatmap, streaks, count = sim.run_sim()
            heatmap_list.append(heatmap)
            streaks_list.append(streaks)
            sim_list.append(sim)
            name = cls.species_names_dict.get(idx, f"Z={sim.Z_particle}")
            gcr_counts.append((name, count))

        combined_heatmap = np.zeros_like(heatmap_list[0], dtype=np.int64)  # int64 avoids overflow
        for hm in heatmap_list:
            # in-place: does not allocate a new array
            np.add(combined_heatmap, hm, out=combined_heatmap, casting='unsafe')
            
            
        if return_sims: #this version saves the num_part_table for each species inside sim_list
            return (
                combined_heatmap,
                heatmap_list,
                streaks_list,
                gcr_counts,
                sim_list,        
            )
        else:
            # Backwards-compatible 4-tuple without sim_list
            return (
                combined_heatmap,
                heatmap_list,
                streaks_list,
                gcr_counts,
            )

    @staticmethod
    def encode_pid(species_idx, primary_idx, delta_idx):
        """
        Encode a PID using bit-based packing:
          - 7 bits for species_idx (0–127)
          - 11 bits for primary_idx (0–2047)
          - 14 bits for delta_idx (0–16383)
        Returns a 32-bit integer.
        """
        return (species_idx << (11 + 14)) | (primary_idx << 14) | delta_idx

    @staticmethod
    def decode_pid(encoded, species_names=species_names_list):
        """
        Decode an encoded 32-bit PID integer into a human-readable string.

        By default, species index 0→"e", 1→"H", 2→"He", 3→"Li", … unless
        a custom mapping is provided via ``species_names``.

        Parameters
        ----------
        encoded : int
            The encoded 32-bit PID value.
        species_names : Sequence[str] | None, default=None
            Optional mapping from species index to species symbol. If ``None``,
            a built-in list (``["e", "H", "He", ...]``) is used.

        Returns
        -------
        str
            A string like ``"H-P0045-D00023"`` where:
            * prefix is the species symbol,
            * ``P####`` is the zero-padded primary index,
            * ``D#####`` is the zero-padded delta-ray index.
        """
        # Define bit widths:
        species_bits = 7  # 0-127
        primary_bits = 11  # 0-2047
        delta_bits = 14  # 0-16383

        # Extract bits:
        delta_mask = (1 << delta_bits) - 1  # lower 14 bits mask
        primary_mask = (1 << primary_bits) - 1  # next 11 bits mask
        species_mask = (1 << species_bits) - 1  # top 7 bits mask

        delta_idx = encoded & delta_mask
        primary_idx = (encoded >> delta_bits) & primary_mask
        species_idx = (encoded >> (primary_bits + delta_bits)) & species_mask

        species_name = species_names[species_idx]

        return f"{species_name}-P{primary_idx:04d}-D{delta_idx:05d}"

    @staticmethod
    def encode_pid_string(pid_str,species_names=species_names_list):
        """
        Parse a PID string like ``"H-P0045-D00023"`` and return its 32-bit encoding.

        Parameters
        ----------
        pid_str : str
            PID in the format ``"<Species>-P<primary:04d>-D<delta:05d>"``, e.g.,
            ``"He-P0012-D00007"``. The species part must either be in the built-in
            table (``["e","H","He",...]``) or of the form ``"X<index>"`` for a
            numeric species index.

        Returns
        -------
        int
            The encoded 32-bit PID value.

        Raises
        ------
        ValueError
            If the input string does not match the expected format or indices cannot be parsed.
        """
        # Expected format: "<species>-P<primary_idx:04d>-D<delta_idx:05d>"
        parts = pid_str.split("-")
        if len(parts) != 3:
            raise ValueError("PID string must be in the format 'Species-Pxxxx-Dyyyyy'")

        species_part, primary_part, delta_part = parts

        # Verify that the primary and delta parts start with 'P' and 'D' respectively.
        if not primary_part.startswith("P") or not delta_part.startswith("D"):
            raise ValueError("PID string must have parts in the format 'Pxxxx' and 'Dyyyyy'")

        # try:
        primary_idx = int(primary_part[1:])
        delta_idx = int(delta_part[1:])
        species_idx = species_names.index(species_part)

        return CosmicRaySimulation.encode_pid(species_idx, primary_idx, delta_idx)

    @staticmethod
    def get_parent_pid(encoded_pid):
        """
        Return the parent (primary) PID by clearing the delta-ray portion.

        Parameters
        ----------
        encoded_pid : int
            The encoded 32-bit PID of a delta-ray particle.

        Returns
        -------
        int
            The encoded 32-bit PID of the parent primary (lower 14 bits zeroed).
        """
        # Zero out the lower 14 bits that represent the delta ray index.
        parent_encoded = encoded_pid & ~((1 << 14) - 1)
        # Return the parent's PID in bit format.
        return parent_encoded

    @staticmethod
    def generate_angles(init_en, mass):
        """Generate emission angles and velocity for a given initial energy and mass."""
        vel = np.sqrt((2 * init_en) / mass)
        P = np.random.uniform(0, 1)
        theta = np.arcsin(np.sqrt(P))
        phi = np.random.uniform(0, 2 * np.pi)
        return theta, phi, vel # unitless, unitless, m/s

    @staticmethod
    def gamma(Ekin, mass):
        """
        Compute the Lorentz factor γ = E_total / m for a particle.

        Parameters
        ----------
        Ekin : float
            Kinetic energy of the particle (in the same energy units as `mass`, typically GeV).
        mass : float
            Rest mass energy of the particle (e.g., GeV/nucleon).

        Returns
        -------
        float
            The Lorentz factor γ = (Ekin + mass) / mass.
        """
        return (Ekin + mass) / mass #unitless

    @staticmethod
    def compute_curvature(positions):
        """Compute curvature along a trajectory."""
        positions = np.array(positions)
        n_points = positions.shape[0]
        if n_points < 3:
            return np.array([0])
        kappa_values = np.zeros(n_points - 2)
        for i in range(1, n_points - 1):
            p0, p1, p2 = positions[i - 1], positions[i], positions[i + 1]
            vec1, vec2 = p1 - p0, p2 - p1
            norm_vec1, norm_vec2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
            if norm_vec1 == 0 or norm_vec2 == 0:
                kappa_values[i - 1] = 0
                continue
            t1, t2 = vec1 / norm_vec1, vec2 / norm_vec2
            delta_t = t2 - t1
            ds = (norm_vec1 + norm_vec2) / 2
            kappa_values[i - 1] = 0 if ds == 0 else np.linalg.norm(delta_t) / ds
        return kappa_values # unitless

    @staticmethod
    def transform_angles(theta_p, phi_p, theta_d, phi_d):
        """Transform delta ray emission angles from the particle's frame to the global frame."""
        vp = np.array([np.sin(theta_p) * np.cos(phi_p), np.sin(theta_p) * np.sin(phi_p), np.cos(theta_p)])
        vd = np.array([np.sin(theta_d) * np.cos(phi_d), np.sin(theta_d) * np.sin(phi_d), np.cos(theta_d)])
        axis = np.cross([0, 0, 1], vp)
        if np.linalg.norm(axis) != 0:
            axis = axis / np.linalg.norm(axis)
            angle = np.arccos(np.dot([0, 0, 1], vp))
            K_mat = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            R = np.eye(3) + np.sin(angle) * K_mat + (1 - np.cos(angle)) * np.dot(K_mat, K_mat)
        else:
            R = np.eye(3)
        vd_global = np.dot(R, vd)
        theta_global = np.arccos(vd_global[2])
        phi_global = np.arctan2(vd_global[1], vd_global[0])
        return theta_global, phi_global # unitless, unitless

    @staticmethod
    def load_sim(filename):
        """
        Load simulation outputs from an HDF5 file written by save_sim.
        Returns:
          heatmap      : 2D numpy array
          streaks_list : nested list matching save structure
          gcr_counts   : list of (species_name, count) tuples
        """
        with h5py.File(filename, "r") as f:
            print("Found keys:", list(f.keys()))
            # 1) heatmap
            heatmap = f["heatmap"][()]

            # 2) rebuild streaks_list
            streaks_group = f["streaks"]
            streaks_list = []
            for sp_key in sorted(streaks_group, key=lambda k: int(k.split("_")[1])):
                sp_grp = streaks_group[sp_key]
                species_streaks = []
                for bin_key in sorted(sp_grp, key=lambda k: int(k.split("_")[1])):
                    bin_grp = sp_grp[bin_key]
                    bin_streaks = []
                    for st_key in sorted(bin_grp, key=lambda k: int(k.split("_")[1])):
                        sg = bin_grp[st_key]
                        # Read attrs
                        pid = int(sg.attrs["PID"])
                        num_steps = int(sg.attrs["num_steps"])
                        theta_i = float(sg.attrs["theta_init"])
                        phi_i = float(sg.attrs["phi_init"])
                        theta_f = float(sg.attrs["theta_final"])
                        phi_f = float(sg.attrs["phi_final"])
                        start_pos = tuple(sg.attrs["start_position"])
                        end_pos = tuple(sg.attrs["end_position"])
                        init_en = float(sg.attrs["init_en"])
                        final_en = float(sg.attrs["final_en"])
                        delta_count = int(sg.attrs["delta_count"])
                        is_primary = bool(sg.attrs["is_primary"])
                        # Read datasets
                        positions = [tuple(x) for x in sg["positions"][()]]
                        theta0_vals = sg["theta0_vals"][()].tolist()
                        curr_vels = [tuple(x) for x in sg["curr_vels"][()]]
                        new_vels = [tuple(x) for x in sg["new_vels"][()]]
                        energy_changes = [tuple(x) for x in sg["energy_changes"][()]]

                        streak = (
                            positions,
                            pid,
                            num_steps,
                            theta_i,
                            phi_i,
                            theta_f,
                            phi_f,
                            theta0_vals,
                            curr_vels,
                            new_vels,
                            energy_changes,
                            start_pos,
                            end_pos,
                            init_en,
                            final_en,
                            delta_count,
                            is_primary,
                        )
                        bin_streaks.append(streak)
                    species_streaks.append(bin_streaks)
                streaks_list.append(species_streaks)

            # 3) GCR counts
            species_arr = f["gcr_species"][()].astype(str)
            counts_arr = f["gcr_counts"][()]
            gcr_counts = list(zip(species_arr.tolist(), counts_arr.tolist()))

        print("Data loaded successfully")
        return heatmap, streaks_list, gcr_counts

    def save_sim(self, heatmap, streaks_list, gcr_counts, filename):
        """
        Save simulation outputs to an HDF5 file, including the global heatmap,
        all particle streak data, and species-specific GCR counts.

        The file structure is organized as follows:
            • /heatmap              → 2D array of accumulated pixel counts.
            • /streaks/species_i/   → Group for each species index.
            • /streaks/species_i/bin_j/streak_k → Individual particle streaks.
            • /gcr_species          → Dataset of species names.
            • /gcr_counts           → Dataset of GCR event counts per species.

        Parameters
        ----------
        heatmap : numpy.ndarray
            2D array of accumulated pixel counts from the simulation.
        streaks_list : list
            Nested list containing all streak data for each species and energy bin.
            Each streak entry is a tuple containing:
                (positions, pid, num_steps, theta_i, phi_i, theta_f, phi_f,
                theta0_vals, curr_vels, new_vels, energy_changes,
                start_pos, end_pos, init_en, final_en, delta_count, is_primary).
        gcr_counts : list of tuple
            List of (species_name, count) tuples representing the number of
            GCR events detected for each particle species.
        filename : str
            Path to the output HDF5 file.

        Returns
        -------
        None
            The function saves data to disk and prints a confirmation message.

        Notes
        -----
        - Each streak is stored as a separate HDF5 group with compressed datasets.
        - Uses gzip compression (level 4) for efficiency.
        - Overwrites existing files of the same name.
        """
        # Prepare GCR counts arrays
        species_names_array, counts = zip(*gcr_counts)
        species_arr = np.array(
            [str(s) for s in species_names_array], dtype=h5py.string_dtype(encoding="utf-8")
        )
        counts_arr = np.array(counts, dtype=np.int64)

        with h5py.File(filename, "w") as f:
            # 1) heatmap
            f.create_dataset("heatmap", data=heatmap, compression="gzip", compression_opts=4)

            # 2) streaks hierarchy
            g_streaks = f.create_group("streaks")
            for sp_idx, species_streaks in enumerate(streaks_list):
                gp = g_streaks.create_group(f"species_{sp_idx}")
                for bin_idx, bin_streaks in enumerate(species_streaks):
                    gb = gp.create_group(f"bin_{bin_idx}")
                    for st_idx, streak in enumerate(bin_streaks):
                        (
                            positions,
                            pid,
                            num_steps,
                            theta_i,
                            phi_i,
                            theta_f,
                            phi_f,
                            theta0_vals,
                            curr_vels,
                            new_vels,
                            energy_changes,
                            start_pos,
                            end_pos,
                            init_en,
                            final_en,
                            delta_count,
                            is_primary,
                        ) = streak

                        gs = gb.create_group(f"streak_{st_idx}")
                        # Attributes
                        gs.attrs["PID"] = int(pid)
                        gs.attrs["num_steps"] = int(num_steps)
                        gs.attrs["theta_init"] = float(theta_i)
                        gs.attrs["phi_init"] = float(phi_i)
                        gs.attrs["theta_final"] = float(theta_f)
                        gs.attrs["phi_final"] = float(phi_f)
                        gs.attrs["start_position"] = tuple(map(float, start_pos))
                        gs.attrs["end_position"] = tuple(map(float, end_pos))
                        gs.attrs["init_en"] = float(init_en)
                        gs.attrs["final_en"] = float(final_en)
                        gs.attrs["delta_count"] = int(delta_count)
                        gs.attrs["is_primary"] = bool(is_primary)
                        # Datasets
                        gs.create_dataset(
                            "positions", data=np.array(positions), compression="gzip", compression_opts=4
                        )
                        gs.create_dataset(
                            "theta0_vals", data=np.array(theta0_vals), compression="gzip", compression_opts=4
                        )
                        gs.create_dataset(
                            "curr_vels", data=np.array(curr_vels), compression="gzip", compression_opts=4
                        )
                        gs.create_dataset(
                            "new_vels", data=np.array(new_vels), compression="gzip", compression_opts=4
                        )
                        gs.create_dataset(
                            "energy_changes",
                            data=np.array(energy_changes),
                            compression="gzip",
                            compression_opts=4,
                        )

            # 3) GCR counts
            f.create_dataset("gcr_species", data=species_arr)
            f.create_dataset("gcr_counts", data=counts_arr)

        print(f"Saved heatmap, streaks, and GCR counts to '{filename}'")

    def Tmax_primary(self, Ekin):
        """
        Maximum transferable energy to an electron for a primary (heavy) particle.

        Parameters
        ----------
        Ekin : float
            Kinetic energy of the primary particle (MeV).

        Returns
        -------
        float
            Maximum energy transfer `T_max` (MeV) from the primary to an electron.
        """
        beta_val = self.beta(Ekin, self.M) # send as {MeV, MeV}, received unitless
        gamma_val = self.gamma(Ekin, self.M) # send as {MeV, MeV}, received unitless
        return (2 * self.me * beta_val**2 * gamma_val**2) / (
            1 + 2 * gamma_val * self.me / self.M + (self.me / self.M) ** 2
        ) #returned in MeV 

    def dEdx_primary(self, Ekin):
        """
        Stopping power (−dE/dx) for a primary (heavy) particle in the detector material.

        Uses a Bethe–Bloch–like expression with material properties stored on the instance.

        Parameters
        ----------
        Ekin : float
            Kinetic energy of the primary particle (MeV).

        Returns
        -------
        float
            Stopping power (MeV/cm) in the current material at the given energy.
        """
        beta_val = self.beta(Ekin, self.M) # send as {MeV, MeV}, received unitless
        gamma_val = self.gamma(Ekin, self.M) # send as {MeV, MeV}, received unitless
        tmax = self.Tmax_primary(Ekin) # MeV
        prefactor = (self.K * self.material_Z * self.Z_particle**2) / (self.material_A * beta_val**2) # MeV*cm^2*mol^-1 *(mol/g) ((and density has units of g/cm^3))
        argument = (2 * self.me * self.c**2 * beta_val**2 * gamma_val**2 * tmax) / (self.I0**2) # unitless
        return prefactor * (0.5 * np.log(argument) - beta_val**2) * self.material_density # MeV/cm

    def dEdx_electron(self, E):
        """
        Stopping power (−dE/dx) for an electron in the detector material.

        Parameters
        ----------
        E : float
            Electron kinetic energy (MeV).

        Returns
        -------
        float
            Stopping power (MeV/cm) for an electron of energy `E`.
        """
        beta_val = np.sqrt(1 - (self.me / (E + self.me)) ** 2) #unitless
        gamma_val = (E + self.me) / self.me #unitless
        W_max = E  # MeV
        return (
            (self.K * self.material_Z)
            / (self.material_A * beta_val**2)
            * (0.5 * np.log(2 * self.me * beta_val**2 * gamma_val**2 * W_max / self.I0**2) - beta_val**2)
            * self.material_density  # and here we don't have the c^2 factor, but should we??
        ) # MeV/cm

    def rigidity(self, energy, A, Z, m):
        """
        Particle rigidity R = pc / (|Z|e) GV expressed as a function of kinetic energy.

        Parameters
        ----------
        energy : float
            Kinetic energy (GeV)/nucleon.
        A : float
            Mass number (unitless) (the number of nucleons).
        Z : int
            Charge number (can be negative for electrons).
        m : float
            Rest mass (GeV)/nucleon.

        Returns
        -------
        float
            Rigidity (GV). A small floor (1e-20) is applied for numerical stability.
        """
        R = (A / abs(Z)) * (np.sqrt(energy * (energy + 2 * m))) # removed the (1e-9) from before because now {energy, m} come in as {GeV/nucleon, GeV/nucleon}
        return max(R, 1e-20) # GV

    def get_M_value(self, input_date, df):
        """
        Look up (or extrapolate) the modulation parameter M for a given date.

        If `input_date` is within the range of `df['date']`, returns the closest row's
        `M_value`. Otherwise, uses the 22-year offset heuristic (solar cycle) to select
        a proxy date and returns that row's `M_value`.

        Parameters
        ----------
        input_date : float
            Date as fractional year (e.g., 2018.5).
        df : pandas.DataFrame
            Table with columns ``'date'`` and ``'M_value'``.

        Returns
        -------
        float
            Modulation parameter M for the requested date.
        """
        max_date = df["date"].max()
        if input_date <= max_date:
            diff = (df["date"] - input_date).abs()
            closest_idx = diff.idxmin()
            return df.loc[closest_idx, "M_value"] # unitless
        else:
            predicted_date = input_date - 22
            diff = (df["date"] - predicted_date).abs()
            closest_idx = diff.idxmin()
            return df.loc[closest_idx, "M_value"] # unitless

    def t_minus(self, R):
        """
        Compute the lag timescale :math:`t_-` used in the transport model.

        Parameters
        ----------
        R : float
            Rigidity (GV).

        Returns
        -------
        float
            :math:`t_-` (months), modeled as ``7.5 * R**(-0.45)``.
        """
        t_minus = 7.5 * R**-0.45
        return t_minus # months

    def compute_R0(self, date, R):
        """
        Compute the modulation scale :math:`R_0` for a given date and rigidity.

        This uses the historical dataframe on the instance to find the current cycle
        context, applies a phase lag that depends on :math:`t_+` and :math:`t_- (R)`,
        and evaluates the mean activity at an adjusted date to derive :math:`R_0`.

        Parameters
        ----------
        date : float
            Target date as fractional year.
        R : float
            Rigidity (GV).

        Returns
        -------
        float
            Modulation scale :math:`R_0` (GV) computed from the adjusted mean.
        """
        # --- Find the current row corresponding to target_date ---
        df = self.historic_df
        diff_current = np.abs(df["date"] - date)
        idx_current = diff_current.idxmin()
        current_row = df.loc[idx_current]

        solar_cycle = current_row["solar_cycle"]
        cycle_min = current_row["cycle_min"]
        cycle_max = current_row["cycle_max"]

        # --- Find the row corresponding to (target_date - offset) for the 'mean' value ---
        target_date = date - self.delta_w_t
        diff_old = np.abs(df["date"] - target_date)
        idx_old = diff_old.idxmin()
        old_row = df.loc[idx_old]
        old_mean = old_row["mean"]

        fraction = (old_mean - cycle_min) / cycle_max
        tau = (-1) ** (solar_cycle) * (fraction**0.2)
        dt = 0.5 * (self.t_plus + self.t_minus(R)) + 0.5 * (self.t_plus - self.t_minus(R)) * tau # months
        dt = dt / 12  # Changes dt from months to years to match date in years

        adjusted_date = date - dt # years (fraction thereof)

        # --- Find the row closest to the adjusted_date ---
        diff_adj = (df["date"] - adjusted_date).abs()
        sorted_pos_adj = np.argsort(diff_adj.values)
        closest_pos_adj = sorted_pos_adj[0]
        closest_idx_adj = df.index[closest_pos_adj]
        row_adj = df.loc[closest_idx_adj]
        mean_val = row_adj["mean"]  # Retrieve the 'mean' value from the row corresponding to adjusted_date

        # Compute and return R_0 using the given formula
        R_0 = (mean_val**1.45) * 3e-4 + 0.37 # GV
        return R_0 # GV

    def gamma_func(self, R, i):
        """
        Species-dependent spectral index :math:`\\gamma(R)`.

        Parameters
        ----------
        R : float
            Rigidity (GV).
        i : int
            Species index. If 0 (electrons), uses a rigidity-dependent form;
            otherwise looks up a fixed value from ``gamma_values_list`` (index i-1).

        Returns
        -------
        float
            Spectral index :math:`\\gamma`.
        """
        if i == 0:
            return 3.0 - 1.4 * np.exp(-R / self.R_e) # unitless
        else:
            return self.gamma_values_list[i - 1] # unitless

    def Delta(self, Z, beta, R, R0):
        """
        Modulation correction term :math:`\\Delta` used in the rigidity spectrum.

        Parameters
        ----------
        Z : int
            Charge number of the species (can be negative).
        beta : float
            Relative velocity.
        R : float
            Rigidity (GV).
        R0 : float
            Modulation scale (GV).

        Returns
        -------
        float
            The correction term :math:`\\Delta`.
        """
        D = 5.5 + 1.13 * (Z / abs(Z)) * self.M_polar * ((beta * R) / R0) * np.exp(-(beta * R) / R0)
        return D # unitless

    def log_rigidity_spectrum(self, alpha, beta, g, C, R, D, R0):
        """
        Log of the differential rigidity spectrum :math:`\\ln\\phi(R)`.

        Parameters
        ----------
        alpha : float
            Low-energy slope coefficient (typically species dependent).
        beta : float
            Relative velocity (unitless).
        g : float
            High-energy spectral index (see :func:`gamma_func`).
        C : float
            Normalization constant.
        R : float
            Rigidity (GV).
        D : float
            Modulation correction term (see :func:`Delta`).
        R0 : float
            Modulation scale (GV).

        Returns
        -------
        float
            :math:`\\ln\\phi(R)` at the supplied rigidity.
        """
        R = max(R, 1e-20)
        ln_phi = np.log(C) + alpha * np.log(beta) - g * np.log(R) + D * np.log(R / (R + R0))
        return ln_phi # units inside log are (s*sr*m^2*GeV)^-1 (usually don't want units inside arg but exponentiating right away)

    def delta_rigidity(self, E, delta_E, A, Z, m):
        """
        Convert an energy step :math:`\\Delta E` into a rigidity step :math:`\\Delta R`.

        Parameters
        ----------
        E : float
            Kinetic energy (GeV/nucleon).
        delta_E : float
            Energy increment (GeV/nucleon). 
        A : float
            Mass number (# particles in nuclei).
        Z : int
            Charge number (can be negative).
        m : float
            Rest mass (GeV/nucleon)

        Returns
        -------
        float
            Rigidity increment :math:`\\Delta R` (GV).
        """
        numerator = (A / abs(Z)) * (E + m) * delta_E
        denominator = np.sqrt(E * (E + 2 * m))
        delta_R = (numerator / denominator) #removed 1e-9 factor as now {E,delta_E,m} come in as GeV/nucleon
        return delta_R # GV 

    @staticmethod
    def relative_velocity(R,m,A,Z):
        """
        Dimensionless speed, β, from rigidity R , mass m (in GeV/nucleon), mass number A, and charge number Z.

        Parameters
        ----------
        R : float
            Rigidity in (GV).
        m : float
            Rest mass (GeV)/nucleon.
        A : float
            nucleon (mass) number (unitless) (number of nucleons).
        Z : float
            charge number 

        Returns
        -------
        float
            :math:`\\beta = v/c` (clipped to a small positive minimum for stability).
        """
        denom = np.sqrt(R**2 + ((A*m)/np.abs(Z))**2 )
        return np.maximum(R/denom,1e-40) # unitless
        #beta = (1 / (energy + m)) * (np.sqrt(energy * (energy + 2 * m)))
        #return max(beta, 1e-20) # unitless

    @staticmethod
    def beta(Ekin, mass):
        """
        Compute the dimensionless relativistic velocity β = v/c for a particle from kinetic energy, Ekin, and mass.

        Parameters
        ----------
        Ekin : float
            Kinetic energy of the particle in MeV.
        mass : float
            Rest mass energy of the particle in MeV.

        Returns
        -------
        float
            The dimensionless velocity β = p / E_total, where
            p = sqrt(Ekin(Ekin+2*mass)) and E_total = Ekin + mass.
        """
        total_energy = Ekin + mass # MeV
        p = np.sqrt(Ekin*(Ekin+2*mass)) #MeV
        return p / total_energy #unitless

#NEW DELTA RAY POPULATION CODE BELOW

    def _Eproj_min_from_electron_E(self, Te_MeV, Mproj_MeV):
        """
        Minimal projectile kinetic energy needed to produce an electron
        with kinetic energy Te_MeV, using derivation:
            E_z,min = M * [ (E/(2 m_e)) / ( sqrt(1 + E/(2 m_e)) + 1 ) ]
        Works with scalars or numpy arrays.
        """
        x = np.asarray(Te_MeV, dtype=float) / (2.0 * self.me)      # Te/(2 m_e)
        return Mproj_MeV * ( x / (np.sqrt(1.0 + x) + 1.0) ) # MeV/nucleon (MeV for z<2)


    def _primary_flux_per_species(self, species_idx, bin_edges_eV): # I THINK THIS IS WHERE MY UNITS START GOING WRONG
        """F_Z(E) per eV (same ISO machinery as run_sim, but flux density only?)."""
        E_mid = 0.5 * (bin_edges_eV[:-1] + bin_edges_eV[1:]) # eV/nucleon
        dE    = np.diff(bin_edges_eV) # eV/nucleon

        A  = self.A_list[species_idx] #unitless (num of nucleons)
        Zp = self.Z_list[species_idx] #unitless
        m  = self.m_list[species_idx]  # eV/nucleon
        phi = np.zeros_like(E_mid, dtype=float)
        delta_R = np.zeros_like(E_mid, dtype=float)

        for i, Eev in enumerate(E_mid):
            R   = self.rigidity(Eev*1e-9, A, Zp, m*1e-9) # GV
            R0  = self.compute_R0(self.date, R) # GV
            bet = self.relative_velocity(R,m*1e-9,A,Zp) # s:{GV,GeV/nucleon, # nucleons, charge #}, r: unitless
            g   = self.gamma_func(R, species_idx) # unitless
            D   = self.Delta(Zp, bet, R, R0) # unitless
            ln_phi = self.log_rigidity_spectrum(self.alpha_list[species_idx], bet, g,
                                                self.C_list[species_idx], R, D, R0)
            val = np.exp(ln_phi)
            phi[i] = 0.0 if (not np.isfinite(val) or val <= 0) else val # (s*sr*m^2*GV)^-1
            delta_R[i] = self.delta_rigidity(Eev*1e-9, dE[i]*1e-9, A, Zp, m*1e-9) # GV
            
        flux_z = delta_R*phi # (s*sr*m^2)^-1
        return E_mid, dE, flux_z  # centers (eV/nucleon), widths (eV/nucleon), flux_z in (s*sr*m^2)^-1 


    def _Wmax_primary(self, Ekin_MeV, M_MeV):
        """W_max (MeV) for a primary with kinetic energy Ekin_MeV and rest mass M_MeV."""
        beta_val = self.beta(Ekin_MeV, self.M) # send as {MeV, MeV}, received unitless
        gamma_val = self.gamma(Ekin_MeV, self.M) # send as {MeV, MeV}, received unitless
        return (2.0 * self.me * beta_val**2 * gamma_val**2) / (
            1.0 + 2.0 * gamma_val * self.me / M_MeV + (self.me / M_MeV)**2 ) # MeV


    def _lnLambda(self, E_e_MeV):
        """
        ln Λ(E) for electrons, using the *same argument* as the Bethe–Bloch log in dEdx_electron:
            argument = 2 * m_e * beta_e^2 * gamma_e^2 * W_max / I^2
        with W_max = E 
        """
        # electron beta/gamma at kinetic energy E_e_MeV
        beta_e = np.sqrt(1.0 - (self.me / (E_e_MeV + self.me))**2)
        gamma_e = (E_e_MeV + self.me) / self.me
        argument = (2.0 * self.me * (beta_e**2) * (gamma_e**2) * max(E_e_MeV, 1e-30)) / (self.I0**2) #unitless
        argument = max(argument, 1e-300)
        return np.log(argument) 


    def compute_secondary_electron_flux(
        self,
        kin_energy_bins_eV=None,
        extend_low_electron_E=True,
        E_e_min_eV=1.0e3
    ):
        """
        Implements the formula:

        F_e(E) = (1 / ln Λ(E)) * sum_{z=-1}^{92} z^2 * β_e^2 ∑_{bins} [ F_z(E'_z) * K(E, E'_z) * Θ(Wmax(E'_z) - E) * ΔE'_z ]

        with
        K(E, E'_z) = 1/2 (1/E - 1/Wmax) - (β_z(E'_z)^2 / Wmax) * ln(Wmax/E)

        The integral over dE'_z is evaluated as a Riemann sum over your energy bins,
        and the Heaviside Θ implements the lower bound E_{z,min}(E).

        Returns
        -------
        ebin_edges_eV, ebin_centers_eV, F_e_per_eV
        """
        # Use/extend ISO binning
        if kin_energy_bins_eV is None:
            kin_energy_bins_eV = np.logspace(np.log10(self.start_ISO_energy),
                                            np.log10(self.stop_ISO_energy), 101) # eV/nucleon (eV for z<2)

        e_edges = kin_energy_bins_eV.copy() # eV/nucleon (eV for z<2)
        if extend_low_electron_E and E_e_min_eV < e_edges[0]:
            extra_bins = np.logspace(np.log10(E_e_min_eV),np.log10(self.start_ISO_energy),101)
            extra_bins = extra_bins[:-1]
            e_edges = np.concatenate((extra_bins,kin_energy_bins_eV)) # eV/nucleon (or just eV since now were considering energy of the electrons?)

        E_e_mid_eV = 0.5 * (e_edges[:-1] + e_edges[1:]) # eV/nucleon
        dE_e_eV    = np.diff(e_edges) # eV/nucleon
        E_e_mid_MeV = E_e_mid_eV * 1e-6  # MeV/nucleon

        # Accumulator for F_e(E) per eV
        F_e = np.zeros_like(E_e_mid_eV, dtype=float)

        # Loop over projectile species (z from your Z_list)
        for sidx, zcharge in enumerate(self.Z_list):
            # Primary flux density for this species (per eV)
            Ep_mid_eV, dEp_eV, flux_z = self._primary_flux_per_species(sidx, e_edges) # Ep_mid_eV and dEp_eV in eV/nucleon, units of flux_z are (s*sr*m^2)^-1
            # Kinematics for this species
            M_MeV = self.m_list[sidx] * self.A_list[sidx] * 1e-6 # MeV 
            Ep_mid_MeV = Ep_mid_eV * self.A_list[sidx] * 1e-6 # MeV

            # β_z(E'_z) and Wmax(E'_z)
            beta_p = np.array([self.beta(Ep, M_MeV) for Ep in Ep_mid_MeV]) #unitless, Ep and M_MeV sent as MeV
            Wmax   = np.array([self._Wmax_primary(Ep, M_MeV) for Ep in Ep_mid_MeV])  # MeV

            # Precompute logs safely
            # We'll vectorize over electron energies E and sum over E'_z bins that satisfy Wmax >= E
            for iE, Te in enumerate(E_e_mid_MeV): # MeV/nucleon  (MeV for z<2)
                if Te <= 0.0:
                    continue

                # Heaviside Θ(E′ - E_{z,min}(E))
                Ezmin = self._Eproj_min_from_electron_E(Te, M_MeV)/self.A_list[sidx] # MeV/nucleon (MeV for z<2) * (num of nucleon) = MeV (we divided?)
                mask  = Ep_mid_MeV >= Ezmin
                if not np.any(mask):
                    continue

                Wm   = Wmax[mask] # MeV
                bet  = beta_p[mask] # unitless
                FZ_i = flux_z[mask] # (s*sr*m^2)^-1 
                dEp  = dEp_eV[mask]  # eV

                # Compute beta of the delta electrons at this E (Te)
                beta_e = self.beta(Te, self.me) #needs to be a call to beta, send as {MeV, MeV} since Te for electrons is MeV
                #beta_e = self.relative_velocity(Te*1e-3, self.me*1e-3)

                # kernel with β_e² z² factor
                ratio = np.clip(Wm / Te, 1.0, None)
                K = beta_e**2 * (zcharge**2) * (
                    0.5 * (1.0/Te - 1.0/Wm) - ( (bet**2) / Wm ) * np.log(ratio)
                ) # MeV^-1

                contrib = np.sum(FZ_i * K )  # (s*sr*m^2*MeV)^-1
                lnLambda = self._lnLambda(Te)
                if lnLambda > 0.0:
                    F_e[iE] += contrib / lnLambda #(s*sr*m^2*MeV)^-1 
                if zcharge == 26 and iE%4 == 0:
                    print(f'Ez_min (MeV)={Ezmin},mask={mask},Wm (MeV) = {Wm}, bet = {bet}, FZ_i (s*st*m^2)^-1= {FZ_i}, beta_e={beta_e},K (MeV^-1)={K},contribution(s*st*m^2*MeV)^-1 ={contrib},F_e (s*st*m^2*MeV)^-1 ={F_e}')
        
    
        # Make sure we never return a negative δ-electron flux
        F_e = np.asarray(F_e, dtype=float)
        F_e[F_e < 0] = 0.0

        return e_edges, E_e_mid_eV, F_e*1e-6  # units = {eV/nucleon,eV/nucleon,(s*st*m^2*eV)^-1} changed energy units from MeV to eV

#END NEW DELTA RAY CODE


    def propagate_delta_ray(self, heatmap, x, y, z, theta, phi, init_en, PID, streaks):
        """
        Propagate a secondary (delta ray) particle.
        Records the trajectory on the heatmap and appends a streak record.
        """
        s = self.step_size # um
        x0 = x * self.cell_size # um
        y0 = y * self.cell_size # um
        z0 = z * self.cell_depth # um
        current_energy = init_en # MeV
        positions = [] # um
        theta0_values = [] # unitless
        current_vels = [] # unitless
        new_vels = [] # m/s
        energy_changes = [] # MeV
        theta_init, phi_init = theta, phi # unitless
        s_cm = s * 1e-4  # Convert step size to cm
        X0 = self.X0  # Radiation length in cm

        while current_energy > 0:
            delta_x = s * np.sin(theta) * np.cos(phi) # um
            delta_y = s * np.sin(theta) * np.sin(phi) # um
            delta_z = s * np.cos(theta) # um
            x0 += delta_x # um
            y0 += delta_y # um
            z0 += delta_z # um

            if not (
                0 <= x0 <= self.cell_size * self.grid_size
                and 0 <= y0 <= self.cell_size * self.grid_size
                and 0 <= z0 <= self.cell_depth
            ):
                break

            grid_x = int(x0 / self.cell_size)
            grid_y = int(y0 / self.cell_size)
            if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                heatmap[grid_y, grid_x] += 1 # num of propagation events/um
                positions.append((x0, y0, z0)) # um
            else:
                break

            dE_dx = self.dEdx_electron(current_energy) # MeV/cm
            dE = dE_dx * s_cm # MeV

            # Stop simulation if energy loss is negative; code added by Zac
            if dE < 0:
                dE = current_energy # MeV
                current_energy = 0  # force stop
                break

            if dE > current_energy:
                dE = current_energy # MeV
                current_energy = 0
                break

            beta_val1 = np.sqrt(1 - (self.me / (current_energy + self.me)) ** 2) # unitless
            # p = beta_val1 * (current_energy + self.me) / self.c
            theta0 = (
                (13.6 / (beta_val1 * current_energy)) * np.sqrt(s_cm / X0) * (1 + 0.038 * np.log(s_cm / X0))
            )
            theta0_values.append(theta0) # unitless
            delta_theta = np.random.normal(0, theta0, size=2)  # generate 2D Gaussian on both transverse axes, unitless
            R = np.array(
                [
                    [-np.cos(theta) * np.cos(phi), -np.sin(phi)],
                    [-np.cos(theta) * np.sin(phi), np.cos(phi)],
                    [np.sin(theta), 0.0],
                ]
            )  # rotation matrix: 1st column is "North" direction, 2nd column is "East", unitless
            dvx, dvy, dvz = R @ delta_theta  # get deflection angles in the inertial frame
            vx = np.sin(theta) * np.cos(phi) # unitless
            vy = np.sin(theta) * np.sin(phi) # unitless
            vz = np.cos(theta) # unitless
            vx_new = vx + dvx # unitless
            vy_new = vy + dvy # unitless
            vz_new = vz + dvz # unitless

            norm = np.sqrt(vx_new**2 + vy_new**2 + vz_new**2) # unitless
            vx_new /= norm # unitless
            vy_new /= norm # unitless
            vz_new /= norm # unitless
            theta = np.arccos(vz_new) # unitless
            phi = np.arctan2(vy_new, vx_new) # unitless
            current_vels.append((vx, vy, vz)) # unitless
            new_vels.append((vx_new, vy_new, vz_new)) # unitless
            energy_changes.append((dE, 0)) # MeV

        if positions:
            # pdb.set_trace()
            streaks.append(
                (
                    positions, #um
                    PID, # unitless
                    len(positions), # unitless
                    theta_init, # unitless
                    phi_init, # unitless
                    theta, # unitless
                    phi, # unitless
                    theta0_values, # unitless
                    current_vels, # unitless?
                    new_vels, # unitless?
                    energy_changes, # MeV
                    positions[0], # um
                    positions[-1], #um
                    init_en, # MeV
                    current_energy, # MeV
                    0,
                    False,
                )
            )

    def propagate_GCR(self, heatmap, x, y, theta, phi, init_en, PID, streaks):
        """
        Propagate a primary cosmic ray (GCR). This routine simulates energy loss,
        potential delta ray production, and multiple scattering. It records the primary's
        trajectory on the heatmap and appends a streak record.
        """
        s = self.step_size # um
        x0 = x * self.cell_size # um
        y0 = y * self.cell_size # um
        z0 = 0 # um
        current_energy = init_en # MeV
        positions = [] # um
        theta0_values = [] # unitless
        current_vels = [] # unitless
        new_vels = [] # unitless
        energy_changes = [] # MeV
        theta_init, phi_init = theta, phi # unitless
        s_cm = s * 1e-4  # cm
        delta_ray_counter = 1 # unitless
        primary_idx = (PID >> 14) & ((1 << 11) - 1) # unitless

        # create executor for delta rays
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []

            while current_energy > 0:
                delta_x = s * np.sin(theta) * np.cos(phi) # um
                delta_y = s * np.sin(theta) * np.sin(phi) # um
                delta_z = s * np.cos(theta) # um
                x0 += delta_x # um
                y0 += delta_y # um
                z0 += delta_z # um

                if not (
                    0 <= x0 <= self.cell_size * self.grid_size
                    and 0 <= y0 <= self.cell_size * self.grid_size
                    and 0 <= z0 <= self.cell_depth
                ):
                    break

                grid_x = int(x0 / self.cell_size)
                grid_y = int(y0 / self.cell_size)
                if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                    heatmap[grid_y, grid_x] += 1
                    positions.append((x0, y0, z0)) # um
                else:
                    break

                # Energy loss for primary particle
                dE_dx = self.dEdx_primary(current_energy) # MeV/cm
                dE = dE_dx * s_cm # MeV

                # Stop simulation if energy loss is negative
                if dE < 0:
                    dE = current_energy # MeV
                    current_energy = 0
                    break

                if dE > current_energy:
                    dE = current_energy # MeV
                    current_energy = 0
                    break

                T_delta = 0.0
                # --- Delta ray production ---
                #T_min = 1e-5  # Change from 1 keV in MeV to 10eV in MeV, and then changed back when this ate up too much time (also think its unallowed per PDG?)
                #can also try setting it to I0? seems supported by literature
                T_min = self.I0/5 # I0 ~ 582 eV in MeV => setting T_min to ~ 116 eV in MeV
                
                T_max_val = self.Tmax_primary(current_energy) # MeV
                if T_max_val > current_energy:
                    T_max_val = current_energy

                if T_max_val <= T_min:
                    delta_N = 0  # Avoid issues if Tmax is invalid
                else:
                    num_points = 1000
                    T_vals = np.logspace(np.log10(T_min), np.log10(T_max_val), num_points)
                    dT_vals = np.diff(T_vals)
                    T_centers = (T_vals[:-1] + T_vals[1:]) / 2
                    K = self.K  # 0.307075 MeV*cm^2/g
                    Z = self.material_Z # unitless
                    A = self.material_A # g/mol
                    z = self.Z_particle # unitless
                    beta = self.beta(current_energy, self.M) # unitless
                    rho = self.material_density  # g/cm^3
                    s_cm = self.step_size * 1e-4  # cm
                    E_tot = current_energy + self.M  # total energy (MeV)
                    g_T = 1 - (beta**2 * T_centers / T_max_val) + (T_centers**2) / (2 * E_tot**2) #unitless?
                    g_T = np.maximum(g_T, 0)
                    integrand = np.where(g_T > 0, g_T / T_centers**2, 0)
                    integral_value = np.sum(integrand * dT_vals)

                    delta_N = (K / 2) * (Z / A) * (z**2 / beta**2) * integral_value * rho * s_cm

                # --- delta-ray event logic ---
                if delta_N > 0:
                    if delta_N < 1:
                        # Bernoulli trial: produce 1 delta ray with probability delta_N
                        n_delta = 1 if np.random.uniform(0, 1) < delta_N else 0
                    else:
                        # Poisson-draw number of delta rays when mean is >= 1
                        n_delta = np.random.poisson(delta_N)
                else:
                    n_delta = 0

                for _ in range(n_delta):
                    accepted = False
                    while not accepted:
                        x_inv = np.random.uniform(1 / T_max_val, 1 / T_min)
                        T_candidate = 1 / x_inv
                        accepted = True
                    T_delta = T_candidate # MeV
                    current_energy -= T_delta # MeV
                    if current_energy <= 0: # MeV
                        current_energy = 0
                        break
                    theta_delta = np.arccos(np.sqrt(T_delta / T_max_val))
                    phi_delta = 2 * np.pi * np.random.uniform(0, 1)
                    theta_global, phi_global = self.transform_angles(theta, phi, theta_delta, phi_delta)
                    delta_ray_PID = CosmicRaySimulation.encode_pid(
                        self.species_index, primary_idx, delta_ray_counter
                    )
                    delta_ray_counter += 1
                    futures.append(
                        executor.submit(
                            self._propagate_delta_ray_threadsafe,
                            heatmap,
                            x0 / self.cell_size, # unitless
                            y0 / self.cell_size, # unitless
                            z0 / self.cell_depth, # unitless
                            theta_global, # unitless
                            phi_global, # unitless
                            T_delta, # MeV
                            delta_ray_PID,
                            streaks,
                        )
                    )

                # Multiple scattering for primary
                mp = self.M
                beta_val2 = np.sqrt(1 - (mp / (current_energy + mp)) ** 2)
                p = beta_val2 * (current_energy + mp) / self.c
                theta0 = (
                    (13.6 / (beta_val2 * p * self.c))
                    * np.sqrt(s_cm / self.X0)
                    * (1 + 0.038 * np.log(s_cm / self.X0))
                )
                theta0_values.append(theta0)
                delta_theta = np.random.normal(0, theta0)
                delta_phi = np.random.uniform(0, 2 * np.pi)
                vx = np.sin(theta) * np.cos(phi)
                vy = np.sin(theta) * np.sin(phi)
                vz = np.cos(theta)
                vx_new = vx + delta_theta * np.cos(delta_phi)
                vy_new = vy + delta_theta * np.sin(delta_phi)
                vz_new = vz
                norm = np.sqrt(vx_new**2 + vy_new**2 + vz_new**2)
                vx_new /= norm
                vy_new /= norm
                vz_new /= norm
                theta = np.arccos(vz_new)
                phi = np.arctan2(vy_new, vx_new)
                current_vels.append((vx, vy, vz))
                new_vels.append((vx_new, vy_new, vz_new))
                energy_changes.append((dE, T_delta))

            # wait for all delta-rays to finish
            for _ in as_completed(futures):
                pass
        if positions:
            streaks.append(
                (
                    positions,
                    PID,
                    len(positions),
                    theta_init,
                    phi_init,
                    theta,
                    phi,
                    theta0_values,
                    current_vels,
                    new_vels,
                    energy_changes,
                    positions[0],
                    positions[-1],
                    init_en,
                    current_energy,
                    delta_ray_counter - 1,
                    True,
                )
            )

    def get_particle_color(self, PID):
        """
        Given an encoded PID, extract the primary GCR portion and return the corresponding
        hex color from self.color_list. Assumes:
          - PID is encoded as:
              7 bits: species index
             11 bits: primary index (starting at 1)
             14 bits: delta ray index (0 for primary)
          - self.color_list is a list of tuples (name, hex_code).
        """
        # Extract the primary index: shift out the delta ray bits (14 bits)
        #  and then mask with 11 bits (for primary indices).
        species_idx = (PID >> (11 + 14)) & ((1 << 7) - 1)
        return self.color_list[species_idx][1]

    def run_sim(self, species_index=None):
        """
        Run the simulation for a given number of primary events.
        For each event, a Poisson draw determines the number of primary cosmic rays.
        Each primary is propagated (with secondary delta rays generated along the way).
        Returns:
          heatmap: 2D numpy array of pixel counts.
          streaks: list of tuples recording position and energy loss details for each particle (by PID).
        """
        idx = self.species_index if species_index is None else species_index

        num_pixels = self.grid_size
        heatmap = np.zeros((num_pixels, num_pixels), dtype=int)

        kin_energy_bins = np.logspace(np.log10(self.start_ISO_energy), np.log10(self.stop_ISO_energy), 101) #ISO energies in eV/nucleon
        kin_energies = (kin_energy_bins[:-1] + kin_energy_bins[1:]) / 2 # eV/nucleon
        delta_energies = np.diff(kin_energy_bins) # eV/nucleon

        # Calculate the expected number of particles per energy bin.
        product_values = []
        for iE in range(len(kin_energies)):
            E = kin_energies[iE] # eV/nucleon
            delta_E = delta_energies[iE] # eV/nucleon
            R = self.rigidity(E*1e-9, self.A_list[idx], self.Z_list[idx], self.m_list[idx]*1e-9) # s: {GeV/nucleon, # nucleons, charge #, GeV/nucleon}; r: GV
            R0 = self.compute_R0(self.date, R) # s: GV; r: GV
            beta = self.relative_velocity(R, self.m_list[idx]*1e-9, self.A_list[idx], self.Z_list[idx]) # s: {GV, GeV/nucleon, # nucleons, charge number}; r: {unitless}
            g_val = self.gamma_func(R, idx) # unitless
            D = self.Delta(self.Z_list[idx], beta, R, R0) # unitless
            ln_phi = self.log_rigidity_spectrum(self.alpha_list[idx], beta, g_val, self.C_list[idx], R, D, R0)
            phi_val = np.exp(ln_phi) # (s*st*m^2*GV)^-1 = dN/(dR dA dt dΩ)
            if not np.isfinite(phi_val) or phi_val <= 0:
                phi_val = 0.0

            delta_R = self.delta_rigidity(E*1e-9, delta_E*1e-9, self.A_list[idx], self.Z_list[idx], self.m_list[idx]*1e-9) # GV ~ (dR/dE)*ΔE
            product = phi_val * delta_R * self.dOmega * self.dt * self.dA # unitless = (dN/dR*dA*dt*dΩ)*ΔR * dΩ * dt *dA
            product_values.append(product) # unitless (num of particles)

        product_values = np.array(product_values)
        product_values[product_values <= 0] = np.nan

        # Build a DataFrame for the energy bins.
        year_df_bins = pd.DataFrame(
            {
                "Start Energy (eV/nuc)": kin_energy_bins[:-1],
                "End Energy (eV/nuc)": kin_energy_bins[1:],
                "Bin Center Energy (eV/nuc)": kin_energies, #should I be marking these as eV/nucleon as well?? (it is eV/nucleon, change later)
                "Bin Width (eV/nuc)": delta_energies, #should I be marking these as eV/nucleon as well??
                "Mean # of particles": product_values,
            }
        )
        num_part_table = year_df_bins
        
#NEW DELTA RAY POPULATION CODE
        if idx == 0:  # electron channel (Z = -1)
            # Compute δ-electron secondary flux
            e_edges, e_centers, F_e = self.compute_secondary_electron_flux(
                kin_energy_bins_eV=kin_energy_bins,
                extend_low_electron_E=True,
                E_e_min_eV=1e3  #1keV in eV
            ) # units={eV/nucleon,eV/nucleon,(s*sr*m^2*eV)^-1} but here nucleon # = 1 because its electrons?
            #debug prints
            print_objects = [e_edges,e_centers,F_e]
            for i in range(len(print_objects)):
                print(print_objects[i])
                
            # Convert flux density [per eV] → expected counts (ΔE × Ω × Δt × A)
            dE_e = np.diff(e_edges) # eV/nucleon
            extra_means = np.nan_to_num(F_e * dE_e * self.dOmega * self.dt * self.dA) #  (num of particles) /nucleon #Do I need to multiply through A list?
            
            #debug prints
            print(f"Extra means per species:")
            print(extra_means)
            # Baseline electron mean particle counts (from primaries)
            base = np.nan_to_num(num_part_table['Mean # of particles'].to_numpy(copy=True))

            # Align grids — interpolate δ-electron flux onto the same energy centers
            E_base = num_part_table['Bin Center Energy (eV/nuc)'].values # eV/nucleon
            E_extra = e_centers
            if E_extra.shape[0] != E_base.shape[0]:
                extra_means = np.interp(E_base, E_extra, extra_means, left=0, right=0)

            # Combine the primary and secondary populations
            combined_means = base + extra_means
            num_part_table['Mean # of particles'] = combined_means

#END NEW DELTA RAY CODE

        # Cache a copy of the final per-energy-bin table on the instance
        self.num_part_table = num_part_table.copy()
        
        
        primary_gcr_count = 0
        species_streaks = []
        primary_counter = 1  # Global primary counter for unique primary_idx

        for j in tqdm(range(len(num_part_table)), desc="Processing energy bins", disable=self.progress_bar):
            lambda_value = num_part_table["Mean # of particles"].iat[j]
            if lambda_value <= 0 or not np.isfinite(lambda_value):
                continue
            poisson_samples = np.random.poisson(lambda_value, 1)
            count = int(poisson_samples.sum())
            primary_gcr_count += count
            if count == 0:
                continue

            E_min = num_part_table["Start Energy (eV/nuc)"].iat[j] # eV/nucleon
            E_max = num_part_table["End Energy (eV/nuc)"].iat[j] # eV/nucleon
            streaks = []
            for _ in range(count):
                x = np.random.randint(0, num_pixels)
                y = np.random.randint(0, num_pixels)
                init_en = np.random.uniform(E_min, E_max)
                theta, phi, vel = self.generate_angles(init_en, self.m_list[idx])
                encoded_PID = CosmicRaySimulation.encode_pid(idx, primary_counter, 0)
                primary_counter += 1
                self.propagate_GCR(heatmap, x, y, theta, phi, init_en * 1e-6, encoded_PID, streaks) # {x,y,theta,phi} unitless, energy in MeV/nucleon
            species_streaks.append(streaks)

        if self.apply_padding:
            if self.pad_mode == "constant":
                heatmap = np.pad(
                    heatmap, pad_width=self.pad_pixels, mode=self.pad_mode, constant_values=self.pad_value
                )
            else:
                heatmap = np.pad(heatmap, pad_width=self.pad_pixels, mode=self.pad_mode)
            pad_um = self.pad_pixels * self.cell_size

            # Shift stored positions so they align with the padded heatmap
            for streak_bin in species_streaks:
                for i, streak in enumerate(streak_bin):
                    positions = streak[0]
                    for j in range(len(positions)):
                        x, y, *rest = positions[j]
                        positions[j] = (x + pad_um, y + pad_um, *rest)

                    streak_list = list(streak)
                    start_pos = streak_list[11]
                    streak_list[11] = (start_pos[0] + pad_um, start_pos[1] + pad_um, *start_pos[2:])
                    end_pos = streak_list[12]
                    streak_list[12] = (end_pos[0] + pad_um, end_pos[1] + pad_um, *end_pos[2:])
                    streak_bin[i] = tuple(streak_list)
        # else: no padding, no coordinate shift

        return heatmap, species_streaks, primary_gcr_count

    def plot_energy_spectrum(
        self,
        ax=None,
        show: bool = True,
        label: str | None = None,
        loglog: bool = True,
        xmin: float | None = None,
        xmax: float | None = None,
        ymin: float | None = None,
        ymax: float | None = None,
        alpha: float | None = None,
    ):
        """
        Plot the mean number of particles vs kinetic energy for this species,
        using the cached `num_part_table`.

        Requires that `run_sim()` has been called on this instance so that
        `self.num_part_table` is populated.
        """
        if self.num_part_table is None:
            raise RuntimeError(
                "num_part_table is not set. Run sim.run_sim() on this instance first."
            )

        df = self.num_part_table

        E_centers = df["Bin Center Energy (eV/nuc)"].to_numpy()
        E_start   = df["Start Energy (eV/nuc)"].to_numpy()
        E_end     = df["End Energy (eV/nuc)"].to_numpy()
        weights   = df["Mean # of particles"].to_numpy()

        # Reconstruct bin edges from start/end
        bin_edges = np.concatenate([E_start, E_end[-1:]])

        if label is None:
            Z = self.Z_list[self.species_index]
            m = self.m_list[self.species_index]
            label = f"Z={Z}, m={m:.2e} eV"

        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            created_fig = True

        if alpha is not None:
                    ax.hist(
            E_centers,
            bins=bin_edges,
            weights=weights,
            histtype="bar",
            label=label,
            alpha = alpha
        )
        else:
                    ax.hist(
            E_centers,
            bins=bin_edges,
            weights=weights,
            histtype="bar",
            label=label,
        )
        ax.set_xlabel("Kinetic Energy (eV)")
        ax.set_ylabel("Mean Number of Particles")

        if loglog:
            ax.set_xscale("log")
            ax.set_yscale("log")

        if xmin is not None or xmax is not None:
            # Get existing limits
            lo, hi = ax.get_xlim()
            if xmin is not None:
                lo = xmin
            if xmax is not None:
                hi = xmax
            ax.set_xlim(lo, hi)
            
        if ymin is not None or ymax is not None:
            # Get existing limits
            lo, hi = ax.get_ylim()
            if ymin is not None:
                lo = ymin
            if ymax is not None:
                hi = ymax
            ax.set_ylim(lo, hi)        

        ax.grid(True, which="both", ls="--")
        ax.legend()

        if created_fig and show:
            plt.show()

        return ax

    def total_expected_primaries(self) -> float:
        """
        Total expected number of particles for this species,
        i.e. sum over `Mean # of particles` across all energy bins.

        Requires that `run_sim()` has been called.
        """
        if self.num_part_table is None:
            raise RuntimeError(
                "num_part_table is not set. Run sim.run_sim() on this instance first."
            )

        values = self.num_part_table["Mean # of particles"].to_numpy()
        return float(np.nansum(values))


    def _propagate_delta_ray_threadsafe(self, heatmap, x, y, z, theta, phi, init_en, PID, streaks): # {x,y,theta,phi} unitless, energy in MeV
        """Wrapper to call propagate_delta_ray under a lock."""
        with self._lock:
            self.propagate_delta_ray(heatmap, x, y, z, theta, phi, init_en, PID, streaks)

    def build_energy_loss_csv(self, streaks_list, csv_filename):
        """
        For each unique PID in streaks_list, use self.get_positions_by_pid to pull out
        the full trajectory positions and the list of energy change tuples, then
        write one row per step to a CSV with columns:
          PID, step, x, y, z, dE, delta_energy

        Parameters
        ----------
        streaks_list : list
            The full list of all streaks from a simulation run
            (can be nested [species][primaries][streaks]).
        csv_filename : str
            Output filename for the CSV.
        """
        records = []
        # Find all unique PIDs present in the simulation
        unique_pids = {streak[1] for group in streaks_list for sublist in group for streak in sublist}

        #  For each PID, extract trajectory and energy changes
        for pid in unique_pids:
            positions_lists, _, energy_change_lists = self.get_positions_by_pid(streaks_list, pid)
            for positions, e_changes in zip(positions_lists, energy_change_lists):
                for step_idx, ((x, y, z), (dE, delta)) in enumerate(zip(positions, e_changes)):
                    records.append(
                        {
                            "PID": pid,
                            "step": step_idx,
                            "x": x,
                            "y": y,
                            "z": z,
                            "dE": dE,
                            "delta_energy": delta,
                        }
                    )
        #  Build DataFrame and write to CSV
        df = pd.DataFrame.from_records(records, columns=["PID", "step", "x", "y", "z", "dE", "delta_energy"])
        df.to_csv(csv_filename, index=False)
        print(f"Saved {len(df)} energy‐loss records to '{csv_filename}'")

    def get_positions_by_pid(self, streaks_list, target_pid):
        """
        For a given PID, collect all (x, y, z) positions and energy-change tuples
        from every matching streak in streaks_list.

        Returns
        -------
        positions_list: list of list of (x, y, z)
            Each entry is a trajectory (list of positions) for one matching streak.
        target_pid: int
            The PID queried (returned for convenience).
        en_changes_list: list of list of (dE, delta)
            Each entry is the list of energy changes for one matching streak.
        """
        positions_list = []
        en_changes_list = []
        for streak_group in streaks_list:
            for sublist in streak_group:
                for streak in sublist:
                    if streak[1] == target_pid:
                        positions_list.append(streak[0])
                        en_changes_list.append(streak[-7])
        return positions_list, target_pid, en_changes_list
