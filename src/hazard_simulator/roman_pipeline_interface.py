# gcrsim lvl1 pipeline interface test
import time
from datetime import UTC, datetime

import numpy as np

from .electron_spread2 import process_electrons_to_DN_by_blob2
from .ffrng import FastForwardRNG as ffRNG
from .gcrsim import CosmicRaySimulation


def generate_singleframe_cr(
    seed: int = 1234,
    nat_pix: int = 4088,
    date: float = 2026.790,
    dt: float = 3.04,
    apply_padding: bool = False,
    settings_dict=None,
):
    """
    Generate a single simulated cosmic-ray exposure and return a pixelated electron map.

    This function:
      1. Initializes a deterministic fast-forward RNG.
      2. Runs a full multi-species GCR simulation for a single exposure.
      3. Extracts the resulting particle trajectories (energy depositions).
      4. Passes the events to the charge-diffusion/pixelation pipeline.
      5. Returns the final detector image in electrons per pixel.

    Parameters
    ----------
    seed : int, default=1234
        Seed for the fast-forward random number generator used in charge diffusion.
        Controls deterministic reproducibility of stochastic sampling.
        Units: N/A.
    nat_pix : int, default=4088
        Detector size in pixels per side (assumes a square array).
        Units: pixels.
    date : float, default=2026.790
        Observation date expressed as a fractional year.
        Used by the GCR flux and solar-modulation model.
        Units: years (fractional year).
    dt : float, default=3.04
        Exposure time for the simulated frame.
        Units: seconds.
    apply_padding : bool, default=False
        If True, applies edge padding during the simulation to mitigate boundary effects.
    settings_dict : dict or None, optional
        Placeholder for additional simulation configuration parameters.
        (Currently unused.)

    Returns
    -------
    out_array : numpy.ndarray
        2D array of simulated detector response in electrons per pixel.
        Shape: ``(nat_pix, nat_pix)``.
        Units: electrons.

    Notes
    -----
    - Gain is not applied in this wrapper (``apply_gain=False`` is passed to the
      charge-diffusion routine), so the output is in electrons rather than DN.
    - The returned frame represents a single simulated exposure suitable for
      downstream analysis or visualization.
    """
    rng = ffRNG(seed)  # now that we passed it in, what do we do with it again?

    # create sim object to run gcrs through the detector
    sim = CosmicRaySimulation(grid_size=nat_pix, date=date)
    _, _, trajectory_data, _ = sim.run_full_sim(
        grid_size=nat_pix, dt=dt, progress_bar=True, apply_padding=apply_padding
    )

    # extract the energy deposition and energy transfer data into a csv file
    # current_date = datetime.now()
    # computer_friendly_date = current_date.strftime("%Y%m%d%H%M")
    # file_name = computer_friendly_date+'_energy_loss.csv'
    # output_path = computer_friendly_date+'_outputArray.npy'

    # sim.build_energy_loss_csv(trajectory_data, file_name)

    # send to electron_spread2.py for pixelation (requires having energy deposition csv)
    out_array = process_electrons_to_DN_by_blob2(
        rng_ff=rng, csvfile=None, streaks=trajectory_data, n_pixels=nat_pix, apply_gain=False
    )

    # assuming no gain in electron_spread2(apply_gain = False)
    # at this point, out_array is in electrons per pixel and size (4088,4088)
    return out_array


def main():
    """
    Run a single-frame GCR simulation from the command line and save the result.

    This function:
      1. Records and prints the simulation start time.
      2. Calls ``generate_singleframe_cr`` to produce one simulated detector frame.
      3. Reports the array shape and total runtime.
      4. Saves the output array to disk as a NumPy ``.npy`` file with a timestamped name.

    Returns
    -------
    None

    Side Effects
    ------------
    - Writes a ``.npy`` file containing the simulated frame to the current directory.
    - Prints timing and status information to standard output.

    Notes
    -----
    - Intended as a simple executable entry point for testing and benchmarking the
      GCR simulation + charge-diffusion pipeline.
    - The output file is saved in electrons per pixel (no gain applied).
    """

    start_time = time.time()
    utc_time = datetime.fromtimestamp(start_time, tz=UTC)
    print(f"Starting GCRsim at {utc_time}")
    out_array_img = generate_singleframe_cr(seed=2026)
    end_time = time.time()
    print(f"Cosmic ray simulation complete. Array shape: {out_array_img.shape}")
    print(f"Time to complete = {end_time - start_time} seconds")

    # if you want to save the numpy array for inspection
    np.save(datetime.now().strftime("%Y%m%d%H%M") + "testOuputFrame.npy", out_array_img)
    print("Frame saved.")


# need to build RNG fast forward (FF) capability to be able to step to any particular RN in the series
# also want to figure out how to call single frames of CRs using a call to the RNG

if __name__ == "__main__":
    main()
