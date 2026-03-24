import numpy as np
import pandas as pd
from astropy.io import fits
from pathlib import Path


def load_data():
    try:
        here = Path(__file__).parent
    except NameError:
        # __file__ doesn't exist in notebooks or interactive sessions
        here = Path.cwd()
    fits_path = here / '../Data/Real/20190919_95k_1p1m0p1_fe55_20663_003_diff.fits'  # experimental data file, difference of frames in a fits data cube
    
    with fits.open(fits_path) as hdulist:
        # hdulist is a list of HDU (Header/Data Unit) objects
        primary_hdu = hdulist[0]
        data = primary_hdu.data      # NumPy array of your image/spectrum/whatever
        header = primary_hdu.header  # FITS header metadata

    gain_array      = np.loadtxt('../Select_20663X_summary.txt')[:, 5].reshape((32, 32))
    supercell_size  = 128   # pixels per supercell

    print(f"Data shape: {data.shape}")
    print("Header keys:", list(header.keys())[:10])
    return(data,gain_array,supercell_size)

def main():
    data_cube, gain_array, supercell_size = load_data()


if __name__ == "__main__":
    main()