|badge1| |badge2|

.. |badge1| image:: https://codecov.io/github/Roman-HLIS-Cosmology-PIT/HAZARD-simulator/graph/badge.svg

.. |badge2| image:: https://github.com/Roman-HLIS-Cosmology-PIT/HAZARD-simulator/actions/workflows/smoke-test.yml/badge.svg

###################################################################

# HAZARD Simulator

The High Atomic Z Astrophysical Radiation Dynamics (**HAZARD**) simulator is a framework for forecasting Galactic cosmic rays fluxes and simulating their passage through the photodiode volume of an NIR detector.

The package is currently configured to simulate interactions with a Hg<sub>(1-x)</sub>Cd<sub>x</sub>Te semiconductor, where the molar ratio x is set to 0.445. 

HAZARD models the incident Galactic cosmic ray (GCR) population of the first 92 elemental species, as well as electrons, according to International Standard ISO 15390:2004(E), Space environment (natural and artificial) - Galactic cosmic ray model. An additional flux of ejected low-energy electrons resulting from the GCRs passing through the rest of the spacecraft before entering the Cold Sensing Module (CSM) and impacting the Wide Field Instrument (WFI) of the Nancy Grace Roman Space Telescope (NGRST).

HAZARD takes these charge particle populations and uses Monte Carlo and the Bethe-Bloch equations to track how they lose energy and spawn secondary delta-ray particles as they travel through a single Sensor Chip Assembly (SCA) in the WFI. HAZARD calculates the corresponding charge generation and diffusion due to all of the particle interactions during a given timeframe (with the default exposure time, `dt`, set to 3.04 seconds) and outputs arrays of pixel values in units of either electrons or digital-numbers (DN), to be used as cosmic ray masks in the Roman science pipeline.

#Overview
The following scripts make up the core ingredients of HAZARD:

`hazard_simulator.gcrsim`
Generates the initial GCR populations across all energy bins (spanning 10 MeV to 100 GeV for the ISO model and 1 keV to 10 MeV (**double check**) for the low-energy electron correction) and runs them through the Monte Carlo with steps of 0.1 microns inside a volume of 4.088 cm by 4.088 cm by 5 microns to generate the linear energy deposition (LET) data needed by `electron_spread`.

`hazard_simulator.electron_spread`
Converts the data it obtains from `gcrsim`, which contains the energy particles lost to the semiconductor lattice during their passing, into a discrete number of charges according to approximations of the Fano factor and the mean electron-hole creation energy made by fitting data from a 55Fe X-ray energy response test conducted in 2017 by the Detector Characterization Lab (DCL) at Goddard. These charge clouds are then distributed spatially according to a diffusion model taken from Macbeth, et al (2026 pre-release), specifically the sum-of-three Gaussian approximation to reduce `electron_spread`'s computational overhead. These final output image is sent as a numpy array with values given in electrons or DNs, as long as 32x32 superpixel gain map corresponding to that SCA is also passed along and `apply_gain` is set to `True`.

`hazard_simulator.ffrng`
A standard random number generator packaged with additional functionality to help with deterministic reproducibility and multi-threading. Must be fed to both `gcrsim` and `electron_spread`.


## Installation

HAZARD currently requires Python version 3.12 or higher.

To clone this repo and install the package into a Python enviromnent, you can use:

.. code-block:: bash

```
git clone https://github.com/Roman-HLIS-Cosmology-PIT/HAZARD-simulator.git
cd HAZARD-simulator

python -m pip install --upgrade pip
python -m pip install -e .
```

A full pipy release is in the works!

## Example use

.. code-block:: python
    
    import hazard_simulator.gcrsim as sim
    import hazard_simulator.electronspread as es
    import hazard_simulator.ffrng as ffrng

    simulator = sim.CosmicRaySimulation # set up simulator object

    my_rng = ffrng.FastForwardRNG() # can also pass a seed number
    data = simulator.run_full_sim(my_rng) # third element in data is trajectory/LET info

    #output_array below is in electrons, can set apply_gain to True and
    # send a gain_txt file (32x32 supercells) in order to get the array in DN
    output_array = es.process_electrons_to_DN(rng_ff=my_rng,streaks=data[2],apply_gain=False)


Questions or Issues?
Please open an issue or submit a pull request for bug fixes, enhancements, or documentation improvements.
