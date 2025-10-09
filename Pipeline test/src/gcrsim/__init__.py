"""
gcrsim — Galactic Cosmic Ray Simulation and Pixelization Framework

This package provides a modular simulation and analysis pipeline
for modeling galactic cosmic ray (GCR) interactions in 
infrared detector arrays (e.g., Teledyne H4RG for Roman Space Telescope).

Core features:
- Monte Carlo cosmic ray event simulation
- Energy deposition tracking and CSV output
- Charge diffusion and pixelization mapping
- Optional GUI interfaces for visualization
"""

__version__ = "0.2.0"
__author__ = "Anthony Harbo Torres et al."
__license__ = "MIT"

# Optionally expose key entry points at top level
from .roman_pipeline_interface import generate_singleframe_cr
from .GCRsim_v02h import CosmicRaySimulation
from .electron_spread2 import process_electrons_to_DN_by_blob

__all__ = [
    "generate_singleframe_cr",
    "CosmicRaySimulation",
    "process_electrons_to_DN_by_blob",
]
