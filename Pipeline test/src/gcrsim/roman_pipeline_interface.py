#gcrsim lvl1 pipeline interface test
import numpy as np
from .GCRsim_v02h import CosmicRaySimulation
from .electron_spread2 import process_electrons_to_DN_by_blob
from datetime import datetime

def generate_singleframe_cr(rng, nat_pix:int = 4088, date:float = 2026.790, dt:float = 3.04,
                            apply_padding: bool = False, settings_dict = None):
    rng = rng #not sure what our plan was for this rng again?
    
    #create sim object to run gcrs through the detector
    sim = CosmicRaySimulation(grid_size=nat_pix, date=date)
    _,_, trajectory_data, _ = sim.run_full_sim(grid_size=nat_pix, dt=dt, progress_bar=True, apply_padding = apply_padding)
    
    #extract the energy deposition and energy transfer data into a csv file
    current_date = datetime.now()
    computer_friendly_date = current_date.strftime("%Y%m%d%H%M")
    file_name = computer_friendly_date+'_energy_loss.csv'
    output_path = computer_friendly_date+'_outputArray.npy'
    
    sim.build_energy_loss_csv(trajectory_data, file_name)
    
    #send to electron_spread2.py for pixelation (requires having energy deposition csv)
    out_array = process_electrons_to_DN_by_blob(
                    csvfile=file_name,
                    n_pixels = nat_pix,
                    output_array_path=output_path,
                    apply_gain = False)
    
    #assuming no gain in electron_spread2(apply_gain = False)
    #at this point, out_array is in electrons per pixel and size (4088,4088)
    
    return out_array

def main():
    rng = np.random.default_rng()
    out_array_img = generate_singleframe_cr(rng)
    print(f"Simulation complete. Array shape: {out_array_img.shape}")
    return 0
