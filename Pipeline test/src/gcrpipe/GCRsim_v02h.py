import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from itertools import chain
from tqdm import tqdm
from importlib.resources import files
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

def mean_excitation_energy_HgCdTe(x):
    """
    Calculate the mean excitation energy for Hg_(1-x)Cd_(x)Te using Bragg's sum rule.
    
    Parameters:
        x (float): Fraction of Cd (0 <= x <= 1). The Hg fraction is (1-x).
    
    Returns:
        float: The effective mean excitation energy (in eV) for the compound.
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
        return 716.4 * A / (Z * (Z + 1) * np.log(287/np.sqrt(Z)))
    
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
    A_Hg = 200.59   # Mercury
    A_Cd = 112.41   # Cadmium
    A_Te = 127.60   # Tellurium
    
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
    
    return density

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

    return Z_mean, A_mean

# Compute material properties for Hg0.555Cd0.445Te (x = 0.445)
x = 0.445 # molar fraction of Cd in Hg(1-x)Cd(x)Te
I_value = mean_excitation_energy_HgCdTe(x)
I_value_MeV = I_value*(1e-6)
X0_gPercmSqd = radiation_length_HgCdTe(x)
HgCdTe_density = density_HgCdTe(x)
X0_cm=X0_gPercmSqd/HgCdTe_density
Z_mean, A_mean = mean_Z_A_HgCdTe(x)

color_list = []
path = files("gcrpipe").joinpath("data/rgb_color_list.txt")
with open(path, 'r') as file:
    for line in file:
        line = line.strip()  # Remove leading/trailing whitespace
        if not line or line.startswith('#'):
            continue  # Skip blank lines or comment lines
        # Split the line by tab. Adjust the separator if needed.
        parts = line.split('\t')
        if len(parts) < 2:
            continue  # Skip lines that don't have enough parts
        color_name = parts[0].strip()
        hex_code = parts[1].strip()
        color_list.append((color_name, hex_code))

# Reading in sunspot data to compute ISO parameters and rigidity spectrum
# Sunspot data downloaded from https://www.sidc.be/SILSO/datafiles
csv_path = files("gcrpipe").joinpath("data/SN_m_tot_V2.0.csv")
month_df = pd.read_csv(csv_path, sep=";", engine="python")

#Contents:
  #Column 1-2: Gregorian calendar date, 1.Year, 2.Month
  #Column 3: Date in fraction of year for the middle of the corresponding month
  #Column 4: Monthly mean total sunspot number, W = Ns + 10 * Ng, with Ns the number of spots
   # and Ng the number of groups counted over the entire solar disk
  #Column 5: Monthly mean standard deviation of the input sunspot numbers from individual stations.
  #Column 6: Number of observations used to compute the monthly mean total sunspot number.
  #Column 7: Definitive/provisional marker.

month_df.columns = ['year', 'month', 'date' ,'mean', 'std_dev','num_obs','marker']

frac_amounts = [0.042, 0.123, 0.204, 0.288, 0.371, 0.455, 0.538, 0.623, 0.707, 0.790, 0.874, 0.958]
t_plus = 1 + (frac_amounts[2] + frac_amounts[3])*(1/2)
delta_w_t = 1 + (frac_amounts[3] + frac_amounts[4])*(1/2)

# IF USING SMOOTHED DATA INSTEAD, USE THE FOLLOWING BLOCK:-----
#month_df=month_s_df
#month_df=month_df[:-7]
# -------------------------------------------------------------

# Filter the dataframe to include only dates starting at 1986.707
month_df = month_df[month_df['date'] >= 1986.707].copy()

# Initialize 'solar_cycle' and update according to date ranges:
month_df['solar_cycle'] = 22
month_df.loc[(month_df['date'] >= 1996.624) & (month_df['date'] <= 2008.874), 'solar_cycle'] = 23
month_df.loc[(month_df['date'] >= 2008.958) & (month_df['date'] <= 2019.873), 'solar_cycle'] = 24
month_df.loc[month_df['date'] >= 2019.958, 'solar_cycle'] = 25

# Define the dates where the solar cycle changes
cycle_change_dates = [1996.624, 2008.958, 2019.958]
cycle_labels = ['Cycle 23 starts', 'Cycle 24 starts', 'Cycle 25 starts']

# For each solar cycle, find the row with the maximum and minimum 'mean'
cycle_max = month_df.loc[month_df.groupby("solar_cycle")["mean"].idxmax()]
cycle_min = month_df.loc[month_df.groupby("solar_cycle")["mean"].idxmin()]

month_df['cycle_max'] = month_df.groupby('solar_cycle')['mean'].transform('max')
month_df['cycle_min'] = month_df.groupby('solar_cycle')['mean'].transform('min')

# Group the dataframe by 'solar_cycle' and find the index of the row with the maximum 'mean' for each group
cycle_max_df = month_df.loc[month_df.groupby("solar_cycle")["mean"].idxmax()]
# First, extract the sign reversal moments by finding, for each solar cycle, 
# the date at which the 'mean' is maximum.
cycle_max_df = month_df.loc[month_df.groupby("solar_cycle")["mean"].idxmax()]

# Create a mapping: solar_cycle -> sign reversal moment (date)
sign_reversal_dict = cycle_max_df.set_index('solar_cycle')['date'].to_dict()

def compute_M(target_date, df, sign_reversal_dict, tol=3e-2):
    """
    Given a target date and a dataframe (with a 'date' column and solar cycle columns),
    this function finds the entry whose 'date' is within a tolerance of target_date,
    and then computes:
    
    M = S * (-1)^(solar_cycle - 1) * ((mean - cycle_min) / (cycle_max - cycle_min))^2.7
    
    where S = 1 if (target_date - sign_reversal_date) >= 0,
          S = -1 otherwise.
    
    Parameters:
      target_date (float): The date (fraction of year) to search for.
      df (pd.DataFrame): DataFrame containing the data.
      sign_reversal_dict (dict): Dictionary mapping solar_cycle to its sign reversal date.
      tol (float): Tolerance for matching the date.
    
    Returns:
      float: The computed M value for the matching entry.
    
    Raises:
      ValueError: If no entry is found within the tolerance.
    """
    # Find the row whose 'date' is closest to target_date
    diff = np.abs(df['date'] - target_date)
    if diff.min() > tol:
        raise ValueError(f"No entry found for date {target_date} within tolerance {tol}.")
    idx = diff.idxmin()
    row = df.loc[idx]
    
    # Extract values from the row
    solar_cycle = row['solar_cycle']
    mean_val = row['mean']
    cycle_max_val = row['cycle_max']
    cycle_min_val = row['cycle_min']
    
    # Check for division by zero
    if cycle_max_val == cycle_min_val:
        raise ValueError("cycle_max and cycle_min are equal; cannot compute fraction.")
    
    fraction = (mean_val - cycle_min_val) / (cycle_max_val - cycle_min_val)
    
    # Compute the sign factor from solar_cycle
    factor = (-1)**(int(solar_cycle) - 1)
    
    # Compute S based on the target_date relative to the sign reversal moment for that cycle
    sign_reversal = sign_reversal_dict[solar_cycle]
    S = 1 if (target_date - sign_reversal) >= 0 else -1
    
    # Compute M using the modified formula
    M_value = S * factor * (fraction**2.7)
    return M_value

# Now, apply compute_M over the dataframe. 
# For each row (using its 'date'), compute the corresponding M_value.
month_df['M_value'] = month_df['date'].apply(lambda d: compute_M(d, month_df, sign_reversal_dict, tol=1e-2))

class CosmicRaySimulation:
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
              92*(0.9382720813e9) + 146*(0.9395654133e9) ] # masses in eV
    A_list = [ 1.0, 1.0, (4.0 / 4), (6.9 / 7), (9.0 / 9), (10.8 / 11), (12.0 / 12), (14.0 / 14), (16.0 / 16), (19.0 / 19),
             (20.2 / 20), (23.0 / 23), (24.3 / 24), (27.0 / 27), (28.1 / 28), (31.0 / 31), (32.1 / 32), (35.4 / 35),
             (39.9 / 40), (39.1 / 39), (40.1 / 40), (44.9 / 45), (47.9 / 48), (50.9 / 51), (52.0 / 52), (54.9 / 55),
             (55.8 / 56), (58.9 / 59), (58.7 / 58), (63.5 / 63), (65.4 / 64), (69.7 / 69), (72.6 / 74), (74.9 / 75),
             (79.0 / 80), (79.9 / 79), (83.8 / 83), (85.5 / 85), (87.6 / 88), (88.9 / 89), (91.2 / 90), (92.9 / 93), 
             (95.9 / 98), (97.0 / 97), (101.0 / 102), (102.9 / 103), (106.4 / 106), (107.9 / 107), (112.4 / 114), (114.8 / 118), 
             (118.7 / 119), (121.8 / 121), (127.6 / 130), (126.9 / 127), (131.3 / 132), (132.9 / 133), (137.3 / 138), (138.9 / 139), 
             (140.1 / 140), (140.9 / 141), (144.2 / 142), (144.2 / 144), (145.0 / 152), (150.4 / 153), (152.0 / 158), (157.3 / 159), 
             (158.9 / 164), (162.5 / 165), (164.9 / 166), (167.3 / 169), (168.9 / 174), (173.0 / 175), (175.0 / 180), (178.5 / 181),
             (180.9 / 186), (183.9 / 187), (186.2 / 192), (190.2 / 193), (192.2 / 194), (195.1 / 197), (197.0 / 202), (200.6 / 205), 
             (204.4 / 208), (207.2 / 209), (232.0 / 232), (238.0 / 238) ] 
    C_list = [ 170, 1.85e4, 3.69e3, 19.5, 17.7, 49.2, 103.0, 36.7, 87.4, 3.19, 16.4, 4.43, 19.3, 4.17, 13.4, 1.15, 3.06, 1.30,
             2.33, 1.87, 2.17, 0.74, 2.63, 1.23, 2.12, 1.14, 9.32, 0.10, 0.49,
             (9.32 * 6.8e-4), (9.32 * 8.8e-4), (9.32 * 6.5e-5), (9.32 * 1.4e-4), (9.32 * 8.9e-6), (9.32 * 5.2e-5), (9.32 * 9.7e-6), 
             (9.32 * 2.7e-5), (9.32 * 8.8e-6), (9.32 * 2.9e-5), (9.32 * 6.5e-6), (9.32 * 1.6e-5), (9.32 * 2.9e-6), (9.32 * 8.1e-6), 
             (9.32 * 9.5e-7), (9.32 * 3.1e-6), (9.32 * 1.6e-6), (9.32 * 4.6e-6), (9.32 * 1.5e-6), (9.32 * 4.0e-6), (9.32 * 8.8e-7), 
             (9.32 * 4.7e-6), (9.32 * 9.9e-7), (9.32 * 5.7e-6), (9.32 * 1.1e-6), (9.32 * 2.7e-6), (9.32 * 6.5e-7), (9.32 * 6.7e-7), 
             (9.32 * 6.0e-7), (9.32 * 1.8e-6), (9.32 * 4.3e-7), (9.32 * 1.6e-6), (9.32 * 1.9e-7), (9.32 * 1.8e-6), (9.32 * 3.1e-7), 
             (9.32 * 1.4e-6), (9.32 * 3.5e-7), (9.32 * 1.4e-6), (9.32 * 5.3e-7), (9.32 * 8.8e-7), (9.32 * 1.8e-7), (9.32 * 8.9e-7), 
             (9.32 * 1.3e-7), (9.32 * 8.1e-7), (9.32 * 7.3e-8), (9.32 * 8.1e-7), (9.32 * 2.8e-7), (9.32 * 1.2e-6), (9.32 * 7.9e-7), 
             (9.32 * 1.5e-6), (9.32 * 2.8e-7), (9.32 * 4.9e-7), (9.32 * 1.5e-7), (9.32 * 1.4e-6), (9.32 * 7.3e-8), (9.32 * 8.1e-8), (9.32 * 4.9e-8) ]
    alpha_list = [ 1, 2.85, 3.12, 3.41, 4.30, 3.93, 3.18, 3.77, 3.11, 4.05, 3.11, 3.14, 3.65, 3.46, 3.00,
            4.04, 3.30, 4.40, 4.33, 4.49, 2.93, 3.78, 3.79, 3.50, 3.28, 3.29, 3.01, 4.25, 3.52, 
            3.01, 3.01, 3.01, 3.01, 3.01, 3.01, 3.01, 3.01, 3.01, 3.01, 3.01, 3.01,
            3.01, 3.01, 3.01, 3.01, 3.01, 3.01, 3.01, 3.01, 3.01, 3.01, 3.01, 3.01,
            3.01, 3.01, 3.01, 3.01, 3.01, 3.01, 3.01, 3.01, 3.01, 3.01, 3.01, 3.01,
            3.01, 3.01, 3.01, 3.01, 3.01, 3.01, 3.01, 3.01, 3.01, 3.01, 3.01, 3.01,
            3.01, 3.01, 3.01, 3.01, 3.01, 3.01, 3.01, 3.01, 3.01 ]
    gamma_values_list = [ 2.74, 2.77, 2.82, 3.05, 2.96, 2.76, 2.89, 2.70, 2.82, 2.76, 2.84, 2.70, 2.77, 2.66, 2.89,
            2.71, 3.00, 2.93, 3.05, 2.77, 2.97, 2.99, 2.94, 2.89, 2.74, 2.63, 2.63, 2.63,
            2.63, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63,
            2.63, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63,
            2.63, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63,
            2.63, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63,
            2.63, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63]
    frac_amounts = [0.042, 0.123, 0.204, 0.288, 0.371, 0.455, 0.538, 0.623, 0.707, 0.790, 0.874, 0.958]
    t_plus = 1 + (frac_amounts[2] + frac_amounts[3])*(1/2)
    delta_w_t = 1 + (frac_amounts[3] + frac_amounts[4])*(1/2)

    def __init__(self, species_index = 1, grid_size=64, cell_size=10, cell_depth=5, dt = 3.04, 
                 step_size=0.1, material_Z=Z_mean, material_A=A_mean, I0=I_value_MeV, material_density=HgCdTe_density, X0=X0_cm,
                 color_list=color_list, date=2018+frac_amounts[6], historic_df=month_df, progress_bar = False, max_workers = None,
                 apply_padding: bool = True, pad_pixels: int = 4, pad_mode: str = "constant", pad_value: int | float = 0):
        """
        Parameters:
          species_index = index of species to use for various species dependent lists
          grid_size: number of pixels per side of the detector grid.
          cell_size: side length of each pixel (in micrometers).
          cell_depth: pixel depth (in micrometers).
          step_size: step size (in micrometers) for particle propagation.
          material_Z, material_A, I0, material_density, X0: material properties.
          color_list: list of tuples (name, hex_code) for particle coloring.
        """
        self.Z_particle = self.Z_list[species_index]
        self.M = self.m_list[species_index] * 1e-6  # Convert from eV to MeV
        self.species_index = species_index
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.cell_depth = cell_depth
        self.step_size = step_size
        self.date = date
        self.historic_df = historic_df
        # Set M_polar based on the year and historical M data, if available.
        if historic_df is not None:
            self.M_polar = self.get_M_value(self.date, self.historic_df)
        else:
            self.M_polar = 1  # default value if no historical data is provided
            
        # Material properties (passed in from user)
        self.material_Z = material_Z
        self.material_A = material_A
        self.I0 = I0
        self.material_density = material_density
        self.X0 = X0

        # Other simulation constants
        self.me = self.m_list[0] * 1e-6  # Electron mass in MeV/c^2
        self.K = 0.307075  # MeV cm^2/mol
        self.c = 2.99792458e10  # Speed of light in cm/s

        # Energy range for primaries (in MeV)
        self.E_min = 1e1
        self.E_max = 1e5
        self.start_ISO_energy = 1e7 #these are in eV, only affects month.df
        self.stop_ISO_energy = 1e11 #these are in eV, only affects month.df
        
        # ISO model parameters
        self.dt = dt  # seconds
        self.dA = 1e-10  * (self.grid_size) ** 2 # 10 microns in m^2, times number of pixels per microns
        self.dOmega = 2*np.pi  # sr        
        self.R_e = 1
        
        #padding parameters
        self.apply_padding = apply_padding
        self.pad_pixels = pad_pixels
        self.pad_mode = pad_mode
        self.pad_value = pad_value
        
        # Color list for particles
        self.color_list = color_list 
        self.progress_bar = progress_bar
        self.max_workers = max_workers or 4     # or whatever default you prefer
        self._lock = threading.Lock()  
    
    @classmethod
    def run_full_sim(
        cls,
        grid_size: int = 4088,
        progress_bar: bool = False,
        # --- NEW: plumb padding options through this API ---
        apply_padding: bool = True,
        pad_pixels: int = 4,
        pad_mode: str = "constant",
        pad_value: int | float = 0,
        # keep this last so callers can still pass any other CosmicRaySimulation kwargs
        **sim_kwargs,
    ):
        """
        Run run_sim() for every species in Z_list.

        Parameters
        ----------
        grid_size : int
        progress_bar : bool
        apply_padding : bool
            If True, pad the heatmap and shift coordinates (passed to constructor).
        pad_pixels : int
        pad_mode : str
        pad_value : int | float
        **sim_kwargs :
            Any other CosmicRaySimulation.__init__ kwargs (e.g., cell_size, dt, etc.)

        Returns
        -------
        combined_heatmap, heatmap_list, streaks_list, gcr_counts
        """
        heatmap_list = []
        streaks_list = []
        gcr_counts   = []

        for idx in tqdm(range(len(cls.Z_list)),
                        desc="Running simulation for each species",
                        disable=not progress_bar):
            sim = cls(
                species_index=idx,
                grid_size=grid_size,
                progress_bar=progress_bar,
                # --- forward padding args to the constructor ---
                apply_padding=apply_padding,
                pad_pixels=pad_pixels,
                pad_mode=pad_mode,
                pad_value=pad_value,
                # --- plus anything else the caller provided ---
                **sim_kwargs,
            )

            heatmap, streaks, count = sim.run_sim()
            heatmap_list.append(heatmap)
            streaks_list.append(streaks)
            name = cls.species_names.get(idx, f"Z={sim.Z_particle}")
            gcr_counts.append((name, count))
            
        combined_heatmap = np.zeros(heatmap.shape)
        #combined_heatmap = np.sum(heatmap_list, axis=0)
        return combined_heatmap, heatmap_list, streaks_list, gcr_counts #must fix combined_heatmap eventually

    
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
    def decode_pid(encoded, species_names=None):
        """
        Decode the encoded PID integer into a human-readable string.
        By default, species index 0 is "e", 1 is "H", 2 is "He", 3 is "Li", etc.
        
        Parameters:
          encoded : int
            The encoded PID.
          species_names : list of str, optional
            A list mapping species indices to names. If not provided, a default list is used.
        
        Returns:
          A formatted string like "H-P0045-D00023" where:
            - "H" is the species name,
            - "P0045" is the primary index padded to 4 digits,
            - "D00023" is the delta ray index padded to 5 digits.
        """
        # Define bit widths:
        species_bits = 7   # 0-127
        primary_bits = 11  # 0-2047
        delta_bits = 14    # 0-16383

        # Extract bits:
        delta_mask = (1 << delta_bits) - 1         # lower 14 bits mask
        primary_mask = (1 << primary_bits) - 1       # next 11 bits mask
        species_mask = (1 << species_bits) - 1       # top 7 bits mask

        delta_idx = encoded & delta_mask
        primary_idx = (encoded >> delta_bits) & primary_mask
        species_idx = (encoded >> (primary_bits + delta_bits)) & species_mask

        # Default species names list.
        if species_names is None:
            species_names = ["e", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", 
                             "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", 
                             "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co",  "Ni", "Cu", 
                             "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", 
                             "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", 
                             "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", 
                             "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", 
                             "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", 
                             "Hg", "Tl", "Pb", "Bi", "Th", "U"]
        try:
            species_name = species_names[species_idx]
        except IndexError:
            species_name = f"X{species_idx}"
        # Format primary and delta indices with leading zeros.
        return f"{species_name}-P{primary_idx:04d}-D{delta_idx:05d}"
        
    @staticmethod
    def encode_pid_string(pid_str):
        """
        Convert a PID string in the format "Species-Pxxxx-Dyyyyy" (e.g., "H-P0045-D00023")
        into its encoded 32-bit integer representation using bit-based packing.
    
        Parameters:
          pid_str : str
            The formatted PID string.        
        Returns:
          An integer representing the encoded PID.
        """
        # Expected format: "<species>-P<primary_idx:04d>-D<delta_idx:05d>"
        parts = pid_str.split('-')
        if len(parts) != 3:
            raise ValueError("PID string must be in the format 'Species-Pxxxx-Dyyyyy'")
        
        species_part, primary_part, delta_part = parts
        
        # Verify that the primary and delta parts start with 'P' and 'D' respectively.
        if not primary_part.startswith("P") or not delta_part.startswith("D"):
            raise ValueError("PID string must have parts in the format 'Pxxxx' and 'Dyyyyy'")
        
        try:
            primary_idx = int(primary_part[1:])
            delta_idx = int(delta_part[1:])
        except Exception as e:
            raise ValueError("Error parsing primary or delta indices: " + str(e))
        
        species_names = ["e", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", 
                         "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", 
                         "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co",  "Ni", "Cu", 
                         "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", 
                         "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", 
                         "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", 
                         "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", 
                         "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", 
                         "Hg", "Tl", "Pb", "Bi", "Th", "U"]       
        try:
            species_idx = species_names.index(species_part)
        except ValueError:
            # Optionally, if your species part can be something like "X#" for unknowns,
            # you might try to parse the number after 'X' here.
            if species_part.startswith("X"):
                try:
                    species_idx = int(species_part[1:])
                except Exception as e:
                    raise ValueError("Invalid species format in PID string: " + str(e))
            else:
                raise ValueError(f"Species '{species_part}' not found in species_names list.")
        
        # Now encode using your existing encode_pid method.
        return CosmicRaySimulation.encode_pid(species_idx, primary_idx, delta_idx)
        
    @staticmethod
    def get_parent_pid(encoded_pid):
        """
        Given an encoded PID of a delta ray particle, returns the PID of its parent GCR
        by zeroing out the delta ray portion (the lower 14 bits) of the PID.
        
        Parameters:
          encoded_pid : int
             The encoded PID (32-bit integer) of the delta ray particle.
          species_names : list of str, optional
             A list mapping species indices to names. If not provided, a default list is used.
        
        Returns:
           A 32-bit integer representing the parent's PID.
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
        return theta, phi, vel

    @staticmethod
    def beta(Ekin, mass):
        total_energy = Ekin + mass
        p = np.sqrt(total_energy**2 - mass**2)
        return p / total_energy

    @staticmethod
    def gamma(Ekin, mass):
        return (Ekin + mass) / mass

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
        return kappa_values

    @staticmethod
    def transform_angles(theta_p, phi_p, theta_d, phi_d):
        """Transform delta ray emission angles from the particle's frame to the global frame."""
        vp = np.array([
            np.sin(theta_p) * np.cos(phi_p),
            np.sin(theta_p) * np.sin(phi_p),
            np.cos(theta_p)
        ])
        vd = np.array([
            np.sin(theta_d) * np.cos(phi_d),
            np.sin(theta_d) * np.sin(phi_d),
            np.cos(theta_d)
        ])
        axis = np.cross([0, 0, 1], vp)
        if np.linalg.norm(axis) != 0:
            axis = axis / np.linalg.norm(axis)
            angle = np.arccos(np.dot([0, 0, 1], vp))
            K_mat = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            R = np.eye(3) + np.sin(angle) * K_mat + (1 - np.cos(angle)) * np.dot(K_mat, K_mat)
        else:
            R = np.eye(3)
        vd_global = np.dot(R, vd)
        theta_global = np.arccos(vd_global[2])
        phi_global = np.arctan2(vd_global[1], vd_global[0])
        return theta_global, phi_global
    
    @staticmethod
    def load_sim(filename):
        """
        Load simulation outputs from an HDF5 file written by save_sim.
        Returns:
          heatmap      : 2D numpy array
          streaks_list : nested list matching save structure
          gcr_counts   : list of (species_name, count) tuples
        """
        with h5py.File(filename, 'r') as f:
            print("Found keys:", list(f.keys()))
            # 1) heatmap
            heatmap = f['heatmap'][()]

            # 2) rebuild streaks_list
            streaks_group = f['streaks']
            streaks_list = []
            for sp_key in sorted(streaks_group, key=lambda k: int(k.split('_')[1])):
                sp_grp = streaks_group[sp_key]
                species_streaks = []
                for bin_key in sorted(sp_grp, key=lambda k: int(k.split('_')[1])):
                    bin_grp = sp_grp[bin_key]
                    bin_streaks = []
                    for st_key in sorted(bin_grp, key=lambda k: int(k.split('_')[1])):
                        sg = bin_grp[st_key]
                        # Read attrs
                        pid          = int(sg.attrs['PID'])
                        num_steps    = int(sg.attrs['num_steps'])
                        theta_i      = float(sg.attrs['theta_init'])
                        phi_i        = float(sg.attrs['phi_init'])
                        theta_f      = float(sg.attrs['theta_final'])
                        phi_f        = float(sg.attrs['phi_final'])
                        start_pos    = tuple(sg.attrs['start_position'])
                        end_pos      = tuple(sg.attrs['end_position'])
                        init_en      = float(sg.attrs['init_en'])
                        final_en     = float(sg.attrs['final_en'])
                        delta_count  = int(sg.attrs['delta_count'])
                        is_primary   = bool(sg.attrs['is_primary'])
                        # Read datasets
                        positions      = [tuple(x) for x in sg['positions'][()]]
                        theta0_vals    = sg['theta0_vals'][()].tolist()
                        curr_vels      = [tuple(x) for x in sg['curr_vels'][()]]
                        new_vels       = [tuple(x) for x in sg['new_vels'][()]]
                        energy_changes = [tuple(x) for x in sg['energy_changes'][()]]

                        streak = (positions, pid, num_steps,
                                  theta_i, phi_i, theta_f, phi_f,
                                  theta0_vals, curr_vels, new_vels,
                                  energy_changes,
                                  start_pos, end_pos,
                                  init_en, final_en,
                                  delta_count, is_primary)
                        bin_streaks.append(streak)
                    species_streaks.append(bin_streaks)
                streaks_list.append(species_streaks)

            # 3) GCR counts
            species_arr = f['gcr_species'][()].astype(str)
            counts_arr  = f['gcr_counts'][()]
            gcr_counts  = list(zip(species_arr.tolist(), counts_arr.tolist()))

        print('Data loaded successfully')
        return heatmap, streaks_list, gcr_counts
    
    def save_sim(self, heatmap, streaks_list, gcr_counts, filename):
        """
        Save simulation outputs to HDF5:
          • heatmap      → /heatmap
          • streaks_list → /streaks/species_i/bin_j/streak_k
          • gcr_counts   → /gcr_species and /gcr_counts

        Parameters:
          heatmap      : 2D numpy array of pixel counts
          streaks_list : list of species_streaks lists
          gcr_counts   : list of (species_name, count) tuples
          filename     : str, output HDF5 path
        """
        # Prepare GCR counts arrays
        species_names_list, counts = zip(*gcr_counts)
        species_arr = np.array([str(s) for s in species_names_list],
                               dtype=h5py.string_dtype(encoding='utf-8'))
        counts_arr = np.array(counts, dtype=np.int64)

        with h5py.File(filename, 'w') as f:
            # 1) heatmap
            f.create_dataset('heatmap', data=heatmap,
                             compression='gzip', compression_opts=4)

            # 2) streaks hierarchy
            g_streaks = f.create_group('streaks')
            for sp_idx, species_streaks in enumerate(streaks_list):
                gp = g_streaks.create_group(f'species_{sp_idx}')
                for bin_idx, bin_streaks in enumerate(species_streaks):
                    gb = gp.create_group(f'bin_{bin_idx}')
                    for st_idx, streak in enumerate(bin_streaks):
                        (positions, pid, num_steps,
                         theta_i, phi_i, theta_f, phi_f,
                         theta0_vals, curr_vels, new_vels,
                         energy_changes,
                         start_pos, end_pos,
                         init_en, final_en,
                         delta_count, is_primary) = streak

                        gs = gb.create_group(f'streak_{st_idx}')
                        # Attributes
                        gs.attrs['PID']           = int(pid)
                        gs.attrs['num_steps']     = int(num_steps)
                        gs.attrs['theta_init']    = float(theta_i)
                        gs.attrs['phi_init']      = float(phi_i)
                        gs.attrs['theta_final']   = float(theta_f)
                        gs.attrs['phi_final']     = float(phi_f)
                        gs.attrs['start_position']= tuple(map(float, start_pos))
                        gs.attrs['end_position']  = tuple(map(float, end_pos))
                        gs.attrs['init_en']       = float(init_en)
                        gs.attrs['final_en']      = float(final_en)
                        gs.attrs['delta_count']   = int(delta_count)
                        gs.attrs['is_primary']    = bool(is_primary)
                        # Datasets
                        gs.create_dataset('positions',       data=np.array(positions),
                                          compression='gzip', compression_opts=4)
                        gs.create_dataset('theta0_vals',     data=np.array(theta0_vals),
                                          compression='gzip', compression_opts=4)
                        gs.create_dataset('curr_vels',       data=np.array(curr_vels),
                                          compression='gzip', compression_opts=4)
                        gs.create_dataset('new_vels',        data=np.array(new_vels),
                                          compression='gzip', compression_opts=4)
                        gs.create_dataset('energy_changes',  data=np.array(energy_changes),
                                          compression='gzip', compression_opts=4)

            # 3) GCR counts
            f.create_dataset('gcr_species', data=species_arr)
            f.create_dataset('gcr_counts',  data=counts_arr)

        print(f"Saved heatmap, streaks, and GCR counts to '{filename}'")

    def Tmax_primary(self, Ekin):
        beta_val = self.beta(Ekin, self.M)
        gamma_val = self.gamma(Ekin, self.M)
        return (2 * self.me * beta_val**2 * gamma_val**2) / (
            1 + 2 * gamma_val * self.me / self.M + (self.me / self.M)**2)

    def dEdx_primary(self, Ekin):
        beta_val = self.beta(Ekin, self.M)
        gamma_val = self.gamma(Ekin, self.M)
        tmax = self.Tmax_primary(Ekin)
        prefactor = (self.K * self.material_Z * self.Z_particle**2) / (self.material_A * beta_val**2)
        argument = (2 * self.me * self.c**2 * beta_val**2 * gamma_val**2 * tmax) / (self.I0**2)
        return prefactor * ( 0.5 * np.log(argument) - beta_val**2) * self.material_density

    def dEdx_electron(self, E):
        beta_val = np.sqrt(1 - (self.me / (E + self.me))**2)
        gamma_val = (E + self.me) / self.me
        W_max = E  # Maximum energy transfer assumed equal to total kinetic energy
        return (self.K * self.material_Z) / (self.material_A * beta_val**2) * (
            0.5 * np.log(2 * self.me * beta_val**2 * gamma_val**2 * W_max / self.I0**2) - beta_val**2) * self.material_density

    def rigidity(self, energy, A, Z, m):
        R = (A/abs(Z)) * (np.sqrt(energy*(energy+2*m)))*1e-9
        return max(R, 1e-20)
        
    def get_M_value(self, input_date, df):
        max_date = df['date'].max()
        if input_date <= max_date:
            diff = (df['date'] - input_date).abs()
            closest_idx = diff.idxmin()
            return df.loc[closest_idx, 'M_value']
        else:
            predicted_date = input_date - 22
            diff = (df['date'] - predicted_date).abs()
            closest_idx = diff.idxmin()
            return df.loc[closest_idx, 'M_value']
            
    def t_minus(self, R):
        t_minus = 7.5 * R**-0.45
        return t_minus
    
    def compute_R0(self, date, R):
        # --- Find the current row corresponding to target_date ---
        df = self.historic_df
        diff_current = np.abs(df['date'] - date)
        idx_current = diff_current.idxmin()
        current_row = df.loc[idx_current]
    
        solar_cycle = current_row['solar_cycle']
        cycle_min = current_row['cycle_min']
        cycle_max = current_row['cycle_max']
        
        # --- Find the row corresponding to (target_date - offset) for the 'mean' value ---
        target_date = date - self.delta_w_t
        diff_old = np.abs(df['date'] - target_date)
        idx_old = diff_old.idxmin()
        old_row = df.loc[idx_old]
        old_mean = old_row['mean']
    
        fraction = (old_mean - cycle_min) / cycle_max
        tau = (-1)**(solar_cycle) * (fraction**0.2)        
        dt = 0.5*(self.t_plus + self.t_minus(R)) + 0.5*(self.t_plus - self.t_minus(R)) * tau
        dt = dt/12 # Changes dt from months to years to match date in years
        
        adjusted_date = date - dt
    
        # --- Find the row closest to the adjusted_date ---
        diff_adj = (df['date'] - adjusted_date).abs()
        sorted_pos_adj = np.argsort(diff_adj.values)
        closest_pos_adj = sorted_pos_adj[0]
        closest_idx_adj = df.index[closest_pos_adj]
        row_adj = df.loc[closest_idx_adj]
        mean_val = row_adj['mean'] # Retrieve the 'mean' value from the row corresponding to adjusted_date
    
        # Compute and return R_0 using the given formula
        R_0 = (mean_val ** 1.45) * 3e-4 + 0.37
        return R_0

    def gamma_func(self, R, i):
        if i == 0:
            return 3.0 - 1.4*np.exp(-R/self.R_e)
        else:
            return self.gamma_values_list[i-1]

    def Delta(self, Z, beta, R, R0):
        D = 5.5 + 1.3*(Z/abs(Z))*self.M_polar*((beta*R)/R0)*np.exp(-(beta*R)/R0)
        return D

    def log_rigidity_spectrum(self, alpha, beta, g, C, R, D, R0):
        R = max(R,1e-20)
        ln_phi = np.log(C) + alpha*np.log(beta) - g*np.log(R) + D*np.log(R/(R+R0))
        return ln_phi

    def delta_rigidity(self, E, delta_E, A, Z, m):
        numerator = (A / abs(Z)) * (E + m) * delta_E
        denominator = np.sqrt(E*(E+2*m))
        delta_R = (numerator / denominator)*1e-9
        return delta_R

    def relative_velocity(self, energy, m):
        beta = (1/(energy+m)) * (np.sqrt(energy*(energy+2*m)))
        return max(beta, 1e-20)

    def propagate_delta_rays(self, heatmap, x, y, z, theta, phi, init_en, PID, streaks):
        """
        Propagate a secondary (delta ray) particle.
        Records the trajectory on the heatmap and appends a streak record.
        """
        s = self.step_size
        x0 = x * self.cell_size
        y0 = y * self.cell_size
        z0 = z * self.cell_depth
        current_energy = init_en
        positions = []
        theta0_values = []
        current_vels = []
        new_vels = []
        energy_changes = []
        theta_init, phi_init = theta, phi
        s_cm = s * 1e-4  # Convert step size to cm
        X0 = self.X0  # Radiation length

        while current_energy > 0:
            delta_x = s * np.sin(theta) * np.cos(phi)
            delta_y = s * np.sin(theta) * np.sin(phi)
            delta_z = s * np.cos(theta)
            x0 += delta_x; y0 += delta_y; z0 += delta_z

            if not (0 <= x0 <= self.cell_size * self.grid_size and
                    0 <= y0 <= self.cell_size * self.grid_size and
                    0 <= z0 <= self.cell_depth):
                break

            grid_x = int(x0 / self.cell_size)
            grid_y = int(y0 / self.cell_size)
            if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                heatmap[grid_y, grid_x] += 1
                positions.append((x0, y0, z0))
            else:
                break

            dE_dx = self.dEdx_electron(current_energy)
            dE = dE_dx * s_cm
            
            # Stop simulation if energy loss is negative; code added by Zac
            if dE < 0:
                dE = current_energy  
                current_energy = 0 # force stop
                break
                
            if dE > current_energy:
                dE = current_energy
                current_energy = 0
                break   

            beta_val1 = np.sqrt(1 - (self.me / (current_energy + self.me))**2)
            p = beta_val1 * (current_energy + self.me) / self.c
            theta0 = (13.6 / (beta_val1 * current_energy)) * np.sqrt(s_cm / X0) * (1 + 0.038 * np.log(s_cm / X0))
            theta0_values.append(theta0)
            delta_theta = np.random.normal(0, theta0, size=2) # generate 2D Gaussian on both transverse axes
            R = np.array([[-np.cos(theta)*np.cos(phi), -np.sin(phi)], [-np.cos(theta)*np.sin(phi), np.cos(phi)], [np.sin(theta),0.]]) # rotation matrix: 1st column is "North" direction, 2nd column is "East"
            dvx,dvy,dvz=R@delta_theta # get deflection angles in the inertial frame
            vx = np.sin(theta) * np.cos(phi)
            vy = np.sin(theta) * np.sin(phi)
            vz = np.cos(theta)
            vx_new = vx + dvx
            vy_new = vy + dvy
            vz_new = vz + dvz

            norm = np.sqrt(vx_new**2 + vy_new**2 + vz_new**2)
            vx_new /= norm; vy_new /= norm; vz_new /= norm
            theta = np.arccos(vz_new)
            phi = np.arctan2(vy_new, vx_new)
            current_vels.append((vx, vy, vz))
            new_vels.append((vx_new,vy_new,vz_new))
            energy_changes.append((dE, 0))
        
        if positions:
            #pdb.set_trace()
            streaks.append( ( positions, PID, len(positions), theta_init, phi_init, theta, phi, theta0_values, current_vels,
                             new_vels, energy_changes, positions[0], positions[-1], init_en, current_energy, 0, False ) )

    def propagate_GCR(self, heatmap, x, y, theta, phi, init_en, PID, streaks):
        """
        Propagate a primary cosmic ray (GCR). This routine simulates energy loss,
        potential delta ray production, and multiple scattering. It records the primary's
        trajectory on the heatmap and appends a streak record.
        """
        s = self.step_size
        x0 = x * self.cell_size
        y0 = y * self.cell_size
        z0 = 0
        current_energy = init_en
        positions = []
        theta0_values = []
        current_vels = []
        new_vels = []
        energy_changes = []
        theta_init, phi_init = theta, phi
        s_cm = s * 1e-4 # cm
        delta_ray_counter = 1
        primary_idx = (PID >> 14) & ((1 << 11) - 1)

        # create executor for delta rays
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []

            while current_energy > 0:
                delta_x = s * np.sin(theta) * np.cos(phi)
                delta_y = s * np.sin(theta) * np.sin(phi)
                delta_z = s * np.cos(theta)
                x0 += delta_x; y0 += delta_y; z0 += delta_z

                if not (0 <= x0 <= self.cell_size * self.grid_size and
                        0 <= y0 <= self.cell_size * self.grid_size and
                        0 <= z0 <= self.cell_depth):
                    break

                grid_x = int(x0 / self.cell_size)
                grid_y = int(y0 / self.cell_size)
                if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                    heatmap[grid_y, grid_x] += 1
                    positions.append((x0, y0, z0))
                else:
                    break

                # Energy loss for primary particle
                dE_dx = self.dEdx_primary(current_energy)
                dE = dE_dx * s_cm

                # Stop simulation if energy loss is negative
                if dE < 0:
                    dE = current_energy
                    current_energy = 0
                    break

                if dE > current_energy:
                    dE = current_energy
                    current_energy = 0
                    break

                T_delta = 0.0
                # --- Delta ray production ---
                T_min = 0.001  # 1 keV in MeV
                T_max_val = self.Tmax_primary(current_energy)
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
                    Z = self.material_Z
                    A = self.material_A
                    z = self.Z_particle
                    beta = self.beta(current_energy, self.M)
                    rho = self.material_density  # g/cm^3
                    s_cm = self.step_size * 1e-4  # cm
                    E_tot = current_energy + self.M  # total energy (MeV)
                    g_T = 1 - (beta**2 * T_centers / T_max_val) + (T_centers**2) / (2 * E_tot**2)
                    g_T = np.maximum(g_T, 0)
                    integrand = np.where(g_T > 0, g_T / T_centers**2, 0)
                    integral_value = np.sum(integrand * dT_vals)

                    delta_N = (K/2) * (Z/A) * (z**2 / beta**2) * integral_value * rho * s_cm

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
                        x_inv = np.random.uniform(1/T_max_val, 1/T_min)
                        T_candidate = 1 / x_inv
                        accepted = True  
                    T_delta = T_candidate
                    current_energy -= T_delta
                    if current_energy <= 0:
                        current_energy = 0
                        break
                    theta_delta = np.arccos(np.sqrt(T_delta / T_max_val))
                    phi_delta = 2 * np.pi * np.random.uniform(0, 1)
                    theta_global, phi_global = self.transform_angles(theta, phi, theta_delta, phi_delta)
                    delta_ray_PID = CosmicRaySimulation.encode_pid(self.species_index, primary_idx, delta_ray_counter)
                    delta_ray_counter += 1
                    futures.append(executor.submit(self._propagate_delta_ray_threadsafe, heatmap,
                            x0/self.cell_size, y0/self.cell_size, z0/self.cell_depth,
                            theta_global, phi_global, T_delta, delta_ray_PID, streaks))

                # Multiple scattering for primary
                mp = self.M
                beta_val2 = np.sqrt(1 - (mp / (current_energy + mp))**2)
                p = beta_val2 * (current_energy + mp) / self.c
                theta0 = (13.6 / (beta_val2 * p * self.c)) * np.sqrt(s_cm / self.X0) * (1 + 0.038 * np.log(s_cm / self.X0))
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
                vx_new /= norm; vy_new /= norm; vz_new /= norm
                theta = np.arccos(vz_new)
                phi = np.arctan2(vy_new, vx_new)
                current_vels.append((vx, vy, vz))
                new_vels.append((vx_new,vy_new,vz_new))
                energy_changes.append((dE, T_delta))

            # wait for all delta-rays to finish
            for _ in as_completed(futures):
                pass
        if positions:
            streaks.append( ( positions, PID, len(positions), theta_init, phi_init, theta, phi, theta0_values,
                            current_vels, new_vels, energy_changes, positions[0], positions[-1], init_en,
                            current_energy, delta_ray_counter - 1, True ) )

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
        # Extract the primary index: shift out the delta ray bits (14 bits) and then mask with 11 bits (for primary indices).
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
        if species_index is not None:
            idx = species_index
        else:
            idx = self.species_index

        num_pixels = self.grid_size
        heatmap = np.zeros((num_pixels, num_pixels), dtype=int)

        kin_energy_bins = np.logspace(np.log10(self.start_ISO_energy), np.log10(self.stop_ISO_energy), 101)
        kin_energies = (kin_energy_bins[:-1] + kin_energy_bins[1:]) / 2
        delta_energies = np.diff(kin_energy_bins)
        
        # Calculate the expected number of particles per energy bin.
        product_values = []
        for iE in range(len(kin_energies)):
            E = kin_energies[iE]
            delta_E = delta_energies[iE]
            R = self.rigidity(E, self.A_list[idx], self.Z_list[idx], self.m_list[idx])
            R0 = self.compute_R0(self.date, R)
            beta = self.relative_velocity(E, self.m_list[idx])
            g_val = self.gamma_func(R, idx)
            D = self.Delta(self.Z_list[idx], beta, R, R0)
            ln_phi = self.log_rigidity_spectrum(self.alpha_list[idx], beta, g_val, self.C_list[idx], R, D, R0)
            phi_val = np.exp(ln_phi)
            if not np.isfinite(phi_val) or phi_val <= 0:
                phi_val = 0.0

            delta_R = self.delta_rigidity(E, delta_E, self.A_list[idx], self.Z_list[idx], self.m_list[idx])
            product = phi_val * delta_R * self.dOmega * self.dt * self.dA
            product_values.append(product)
            
        product_values = np.array(product_values)
        product_values[product_values <= 0] = np.nan
        
        # Build a DataFrame for the energy bins.
        year_df_bins = pd.DataFrame({
            'Start Energy (eV)': kin_energy_bins[:-1],
            'End Energy (eV)': kin_energy_bins[1:],
            'Bin Center Energy (eV)': kin_energies,
            'Bin Width (eV)': delta_energies,
            'Mean # of particles': product_values
        })
        # Since we're running a single species simulation, we don't need a list.
        num_part_table = year_df_bins
        primary_gcr_count = 0
        species_streaks = []
        primary_counter = 1  # Global primary counter for unique primary_idx
    
        for j in tqdm(range(len(num_part_table)), desc="Processing energy bins", disable=self.progress_bar):
            lambda_value = num_part_table['Mean # of particles'].iat[j]
            if lambda_value <= 0 or not np.isfinite(lambda_value):
                continue
            poisson_samples = np.random.poisson(lambda_value, 1)
            count = int(poisson_samples.sum())
            primary_gcr_count += count
            if count == 0:
                continue

            E_min = num_part_table['Start Energy (eV)'].iat[j]
            E_max = num_part_table['End Energy (eV)'].iat[j]
            streaks = []
            for _ in range(count):
                x = np.random.randint(0, num_pixels)
                y = np.random.randint(0, num_pixels)
                init_en = np.random.uniform(E_min, E_max)
                theta, phi, vel = self.generate_angles(init_en, self.m_list[idx])
                encoded_PID = CosmicRaySimulation.encode_pid(idx, primary_counter, 0)
                primary_counter += 1
                self.propagate_GCR(heatmap, x, y, theta, phi, init_en*1e-6, encoded_PID, streaks)
            species_streaks.append(streaks)
            
        if self.apply_padding:
            if self.pad_mode == "constant":
                heatmap = np.pad(
                    heatmap,
                    pad_width=self.pad_pixels,
                    mode=self.pad_mode,
                    constant_values=self.pad_value
                )
            else:
                heatmap = np.pad(
                    heatmap,
                    pad_width=self.pad_pixels,
                    mode=self.pad_mode
                )
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
        
    def _propagate_delta_ray_threadsafe(self, heatmap, x, y, z, theta, phi, init_en, PID, streaks):
        """Wrapper to call propagate_delta_rays under a lock."""
        with self._lock:
            self.propagate_delta_rays(heatmap, x, y, z, theta, phi, init_en, PID, streaks)


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
        unique_pids = {
            streak[1]
            for group in streaks_list
            for sublist in group
            for streak in sublist
        }

        #  For each PID, extract trajectory and energy changes
        for pid in unique_pids:
            positions_lists, _, energy_change_lists = self.get_positions_by_pid(streaks_list, pid)
            for positions, e_changes in zip(positions_lists, energy_change_lists):
                for step_idx, ((x, y, z), (dE, delta)) in enumerate(zip(positions, e_changes)):
                    records.append({
                        'PID': pid,
                        'step': step_idx,
                        'x': x,
                        'y': y,
                        'z': z,
                        'dE': dE,
                        'delta_energy': delta
                    })
        #  Build DataFrame and write to CSV
        df = pd.DataFrame.from_records(records,
                                       columns=['PID', 'step', 'x', 'y', 'z', 'dE', 'delta_energy'])
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

    # Define a mapping from species index to species name.
    species_names = {0: "e", 1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne", 11: "Na",
        12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca", 21: "Sc", 22: "Ti", 23: "V", 
        24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn", 31: "Ga", 32: "Ge", 33: "As", 34: "Se",
        35: "Br", 36: "Kr", 37: "Rb", 38: "Sr", 39: "Y", 40: "Zr", 41: "Nb", 42: "Mo", 43: "Tc", 44: "Ru", 45: "Rh", 46: "Pd",
        47: "Ag", 48: "Cd", 49: "In", 50: "Sn", 51: "Sb", 52: "Te", 53: "I", 54: "Xe", 55: "Cs", 56: "Ba", 57: "La", 58: "Ce",
        59: "Pr", 60: "Nd", 61: "Pm", 62: "Sm", 63: "Eu", 64: "Gd", 65: "Tb", 66: "Dy", 67: "Ho", 68: "Er", 69: "Tm", 70: "Yb",
        71: "Lu", 72: "Hf", 73: "Ta", 74: "W", 75: "Re", 76: "Os", 77: "Ir", 78: "Pt", 79: "Au", 80: "Hg", 81: "Tl", 82: "Pb",
        83: "Bi", 90: "Th", 92: "U"}