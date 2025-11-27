#GCR Simulation Framework — Main Scripts Overview (WORK IN PROGRESS)

This repository implements a modular simulation pipeline for studying galactic cosmic ray (GCR) interactions with space-based detectors, specifically tailored for HgCdTe infrared arrays (e.g., as used in the Nancy Grace Roman Space Telescope). The framework enables detailed Monte Carlo simulations of particle events and advanced visualization of charge diffusion and detector response.

Three primary scripts form the backbone of the simulation and analysis workflow:

1. GCRsim_v02i.py

   Purpose:
    Core simulation engine for generating GCR events and their interactions within the detector medium.
    
    Key Features:
    
    Monte Carlo Particle Simulation: Realistic GCR particle generation using customizable input parameters (flux, energy spectrum, incident angle, etc.).
    
    Energy Deposition Modeling: Tracks energy loss and charge creation for each event using stochastic models.
    
    Delta Ray Production: Implements Poisson statistics for simulating secondary electron (“delta ray”) creation.
    
    Data Export: Outputs event data as CSV or HDF5 for further processing.
    
    Typical Usage:
    
    <pre> 
    from GCRsim_v02i import CosmicRaySimulation
    
    # Set up a simulation instance and run
    sim = CosmicRaySimulation(config_file="config.yml")
    sim.run()
    sim.save("output_filename.h5")</pre>
    Inputs:
    
    Detector/material properties (as config file or script arguments)
    
    Simulation settings (particle rate, simulation time, etc.)
    
    Outputs:
    
    Tabular files (CSV/HDF5) with event-level energy deposition, position, and particle ID. This is run for a single species only, as specified by the species index passed in the config.yml file.
    
3. electron_spread2.py

   Purpose:
    Simulates charge diffusion and conversion of deposited energy into electron count and digital number (DN) maps.
    
    Key Features:
    
    Charge Diffusion Modeling: Convolves initial energy depositions with spatially varying Gaussian kernels to mimic realistic electron/hole cloud spreading.
    
    Electron Conversion: Converts energy loss to charge using physical constants (Fano factor, ionization energy, etc.).
    
    Downsampling: Aggregates high-resolution simulation results into low-resolution DN maps matching detector readout.
    
    Batch Processing: Efficiently processes large CSV/HDF5 event datasets in chunks.
    
    Typical Usage:
    
    <pre> 
    from electron_spread import process_electrons_to_DN
    
    process_electrons_to_DN(
        csvfile="gcr_events.csv",
        gain_txt="gain_table.txt",
        det_pixels_lo=4096,
        kernel_size_hi=50,
        sigma=0.314,
        output_DN_path="DN_map.npy"
    )</pre> 
    Inputs:
    
    Event list (CSV/HDF5) from GCRsim_v02f.py
    
    Detector gain/response table
    
    Outputs:
    
    High-res and downsampled DN maps (NumPy arrays or image files)
    
5. GCR_GUI.py

   Purpose:
    Graphical user interface for interactive exploration and visualization of simulated GCR events and DN maps.
    
    Key Features:
    
    Interactive Heatmap Viewer: Inspect downsampled and high-resolution DN maps with zoom, crosshair, and event selection.
    
    Overlay & Annotation: Visualize event positions, detector boundaries, and gridlines for spatial context.
    
    Event Inspection: Double-click to display detailed event info and high-res charge diffusion patterns.
    
    Export & Save Options: Save current visualizations or simulation state for further analysis.
    
    Typical Usage:
    <pre>
    shell
    python GCR_GUI.py</pre>
    or
    
    <pre>
    import GCR_GUI
    GCR_GUI.run()</pre>
    Inputs:
    
    Output files from GCRsim_v02f.py and electron_spread.py (event lists, DN maps, etc.)
    
    Outputs:
    
    Interactive visualizations
    
    Exported plots or event data (optional)

Repository Structure
<pre>
HAZARD-simulator/
├── Sample Outputs/          # Example simulated output data
├── GCRsim_v02i.py           # Main GCR simulation engine
├── electron_spread2.py       # Charge diffusion & DN map processor
├── GCR_GUI.py               # Tkinter-based GUI for visualization
├── requirements.txt         # List of Python dependencies
├── README.md                # (You are here!)
└── [other scripts/modules]  # (e.g., utility modules, tests)</pre>

Quickstart
Install dependencies:

<pre>
pip install -r requirements.txt</pre>

Run a full simulation pipeline:

<pre>
# 1. Simulate GCR events
python GCRsim_v02i.py --config config.yml --output gcr_events.h5

# 2. Process events to create DN maps
python electron_spread2.py --input gcr_events.h5 --output DN_map.npy

# 3. Visualize results with GUI
python GCR_GUI.py</pre>

Note: The GUI requires tkinter. On some Linux systems you may need to run sudo apt-get install python3-tk.

Questions or Issues?
Please open an issue or submit a pull request for bug fixes, enhancements, or documentation improvements.
