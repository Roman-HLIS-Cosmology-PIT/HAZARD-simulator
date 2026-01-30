"""Tests for package data IO"""

import numpy as np
from hazard_simulator import gcrsim


def test_save_load_sim_roundtrip(tmp_path):
    """Test for saving and loading in hdf5 format"""
    CRS = gcrsim.CosmicRaySimulation

    # Create a tiny sim instance with historic_df=None to avoid date-dependent modulation logic
    sim = CRS(species_index=1, grid_size=4, historic_df=None, progress_bar=False)

    heatmap = np.arange(16, dtype=np.int64).reshape(4, 4)

    # Minimal streak structure: [species][bin][streak]
    positions = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    pid = CRS.encode_pid(1, 0, 0)
    streak = (
        positions,  # positions
        pid,  # pid
        2,  # num_steps
        0.1,
        0.2,  # theta_i, phi_i
        0.3,
        0.4,  # theta_f, phi_f
        [0.0, 0.0],  # theta0_vals
        [(0.0, 0.0, 1.0)],  # curr_vels
        [(0.0, 0.0, 1.0)],  # new_vels
        [(0.0, -0.1)],  # energy_changes
        (0.0, 0.0, 0.0),  # start_pos
        (1.0, 0.0, 0.0),  # end_pos
        10.0,  # init_en
        9.9,  # final_en
        0,  # delta_count
        True,  # is_primary
    )
    streaks_list = [[[streak]]]
    gcr_counts = [("H", 1)]

    out = tmp_path / "test_sim.h5"
    sim.save_sim(heatmap, streaks_list, gcr_counts, str(out))

    heatmap2, streaks2, counts2 = CRS.load_sim(str(out))

    assert np.array_equal(heatmap2, heatmap)
    assert counts2 == gcr_counts

    # Spot-check one field in the round-tripped streak
    st0 = streaks2[0][0][0]
    assert st0[1] == pid  # pid
    assert st0[2] == 2  # num_steps
    assert tuple(st0[11]) == (0.0, 0.0, 0.0)  # start_pos

