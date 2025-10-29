#drop your code here

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from importlib.resources import files
import pathlib

#Recommend packaging the solar parameters into a class, e.g.:
class SolarEnvironment():   
    def __init__(self):
        # load SN_m_tot_V2.0.csv into month_df as an attribute of the class, even if the script is called from another directory
        fn = pathlib.Path("SN_m_tot_V2.0.csv").with_name('SN_m_tot_V2.0.csv')
        self.month_df = pd.read_csv(fn, sep='\;', engine = 'python')

        # Reading in sunspot data to compute ISO parameters and rigidity spectrum
        # Sunspot data downloaded from https://www.sidc.be/SILSO/datafiles
        # csv_path = files("gcrsim").joinpath("data/SN_m_tot_V2.0.csv")
        self.month_df = pd.read_csv("SN_m_tot_V2.0.csv", sep=";", engine="python")
        
        #Contents:
        #Column 1-2: Gregorian calendar date, 1.Year, 2.Month
        #Column 3: Date in fraction of year for the middle of the corresponding month
        #Column 4: Monthly mean total sunspot number, W = Ns + 10 * Ng, with Ns the number of spots
        # and Ng the number of groups counted over the entire solar disk
        #Column 5: Monthly mean standard deviation of the input sunspot numbers from individual stations.
        #Column 6: Number of observations used to compute the monthly mean total sunspot number.
        #Column 7: Definitive/provisional marker.
        
        self.month_df.columns = ['year', 'month', 'date' ,'mean', 'std_dev','num_obs','marker']
        
        frac_amounts = [0.042, 0.123, 0.204, 0.288, 0.371, 0.455, 0.538, 0.623, 0.707, 0.790, 0.874, 0.958]
        t_plus = 1 + (frac_amounts[2] + frac_amounts[3])*(1/2)
        delta_w_t = 1 + (frac_amounts[3] + frac_amounts[4])*(1/2)
        
        # IF USING SMOOTHED DATA INSTEAD, USE THE FOLLOWING BLOCK:-----
        #month_df=month_s_df
        #month_df=month_df[:-7]
        # -------------------------------------------------------------
        
        # Filter the dataframe to include only dates starting at 1986.707
        self.month_df = self.month_df[self.month_df['date'] >= 1986.707].copy()
        
        # Initialize 'solar_cycle' and update according to date ranges:
        self.month_df['solar_cycle'] = 22
        self.month_df.loc[(self.month_df['date'] >= 1996.624) & (self.month_df['date'] <= 2008.874), 'solar_cycle'] = 23
        self.month_df.loc[(self.month_df['date'] >= 2008.958) & (self.month_df['date'] <= 2019.873), 'solar_cycle'] = 24
        self.month_df.loc[self.month_df['date'] >= 2019.958, 'solar_cycle'] = 25
        
        # Define the dates where the solar cycle changes
        cycle_change_dates = [1996.624, 2008.958, 2019.958]
        cycle_labels = ['Cycle 23 starts', 'Cycle 24 starts', 'Cycle 25 starts']
        
        # For each solar cycle, find the row with the maximum and minimum 'mean'
        cycle_max = self.month_df.loc[self.month_df.groupby("solar_cycle")["mean"].idxmax()]
        cycle_min = self.month_df.loc[self.month_df.groupby("solar_cycle")["mean"].idxmin()]
        
        self.month_df['cycle_max'] = self.month_df.groupby('solar_cycle')['mean'].transform('max')
        self.month_df['cycle_min'] = self.month_df.groupby('solar_cycle')['mean'].transform('min')
        
        # Group the dataframe by 'solar_cycle' and find the index of the row with the maximum 'mean' for each group
        self.cycle_max_df = self.month_df.loc[self.month_df.groupby("solar_cycle")["mean"].idxmax()]
        # First, extract the sign reversal moments by finding, for each solar cycle, 
        # the date at which the 'mean' is maximum.
        
        # Create a mapping: solar_cycle -> sign reversal moment (date)
        sign_reversal_dict = self.cycle_max_df.set_index('solar_cycle')['date'].to_dict()
        # Create a mapping: solar_cycle -> sign reversal moment (date)
        self.sign_reversal_dict = self.cycle_max_df.set_index('solar_cycle')['date'].to_dict()
        
        # -- this can be in the __init__ function
        # Now, apply compute_M over the dataframe. 
        # For each row (using its 'date'), compute the corresponding M_value.
        
        self.month_df['M_value'] = self.month_df['date'].apply(lambda d: self.compute_M(d, self.month_df, self.sign_reversal_dict, tol=1e-2))

    def compute_M(self,target_date, df, sign_reversal_dict, tol=3e-2):
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
