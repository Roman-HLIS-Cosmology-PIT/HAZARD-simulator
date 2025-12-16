#gcrsim lvl1 pipeline interface test
import time
import numpy as np
from GCRsim_v02h import CosmicRaySimulation
from electron_spread2 import process_electrons_to_DN_by_blob2
from datetime import datetime, timezone
from fastforwardRNG import FastForwardRNG as ffRNG

def generate_singleframe_cr(seed:int = 1234, nat_pix:int = 4088, date:float = 2026.790, dt:float = 3.04,
                            apply_padding: bool = False, settings_dict = None):
    rng = ffRNG(seed) #now that we passed it in, what do we do with it again?
    
    #create sim object to run gcrs through the detector
    sim = CosmicRaySimulation(grid_size=nat_pix, date=date)
    _,_, trajectory_data, _ = sim.run_full_sim(grid_size=nat_pix, dt=dt, progress_bar=True, apply_padding = apply_padding)
    
    #extract the energy deposition and energy transfer data into a csv file
    #current_date = datetime.now()
    #computer_friendly_date = current_date.strftime("%Y%m%d%H%M")
    #file_name = computer_friendly_date+'_energy_loss.csv'
    #output_path = computer_friendly_date+'_outputArray.npy'
    
    #sim.build_energy_loss_csv(trajectory_data, file_name)
    
    #send to electron_spread2.py for pixelation (requires having energy deposition csv)
    out_array = process_electrons_to_DN_by_blob2(
                    rng_ff=rng,
                    csvfile=None,
                    streaks=trajectory_data,
                    n_pixels = nat_pix,
                    apply_gain = False)
    
    #assuming no gain in electron_spread2(apply_gain = False)
    #at this point, out_array is in electrons per pixel and size (4088,4088)
    return out_array

def main():
    start_time = time.time()
    utc_time = datetime.fromtimestamp(start_time, tz=timezone.utc)
    print(f"Starting GCRsim at {utc_time}")
    out_array_img = generate_singleframe_cr(seed=2026)
    end_time = time.time()
    print(f"Cosmic ray simulation complete. Array shape: {out_array_img.shape}")
    print(f"Time to complete = {end_time - start_time} seconds")

    #if you want to save the numpy array for inspection
    np.save(datetime.now().strftime("%Y%m%d%H%M")+'testOuputFrame.npy',out_array_img)
    print('Frame saved.')

    
# need to build RNG fast forward (FF) capability to be able to step to any particular RN in the series
#also want to figure out how to call single frames of CRs using a call to the RNG

if __name__ == "__main__":
    main()