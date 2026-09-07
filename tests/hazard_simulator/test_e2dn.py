import numpy as np
import sep
from hazard_simulator.electron_spread2 import (
    process_electrons_to_DN_by_blob,
    process_electrons_to_DN_by_blob2,
)
from hazard_simulator.ffrng import FastForwardRNG as ffRNG
from hazard_simulator.gcrsim import CosmicRaySimulation


def test_compare_e2dn(tmp_path):
    """Simple test to compare the two e2dn versions."""

    rng = ffRNG(2026)  # now that we passed it in, what do we do with it again?

    # create sim object to run gcrs through the detector
    sim = CosmicRaySimulation(grid_size=4088, date=2026.0, rng=rng)
    _, _, trajectory_data, _ = sim.run_full_sim(
        rng, grid_size=4088, dt=10.0, progress_bar=False, apply_padding=False
    )

    # 4 ways of getting the output array
    out_array1 = process_electrons_to_DN_by_blob(
        csvfile=None, streaks=trajectory_data, n_pixels=4088, apply_gain=False, rng=rng
    ).astype(np.float32, copy=False)
    out_array2 = process_electrons_to_DN_by_blob2(
        rng_ff=rng, csvfile=None, streaks=trajectory_data, n_pixels=4088, apply_gain=False
    ).astype(np.float32, copy=False)
    out_array3 = process_electrons_to_DN_by_blob(
        csvfile=None, streaks=trajectory_data, n_pixels=4088, apply_gain=False, rng=rng, one_explicit=True
    ).astype(np.float32, copy=False)
    out_array4 = process_electrons_to_DN_by_blob2(
        rng_ff=rng, csvfile=None, streaks=trajectory_data, n_pixels=4088, apply_gain=False, one_explicit=True
    ).astype(np.float32, copy=False)

    obj1 = sep.extract(out_array1, 50.0, minarea=2)
    obj2 = sep.extract(out_array2, 50.0, minarea=2)
    obj3 = sep.extract(out_array3, 50.0, minarea=2)
    obj4 = sep.extract(out_array4, 50.0, minarea=2)

    # verify the first 3 hits are similar
    for i in range(3):
        print(i)
        assert np.hypot(obj1["x"][i] - obj2["x"][i], obj1["y"][i] - obj2["y"][i]) < 0.5
        assert -0.2 < np.log(obj1["flux"][i] / obj2["flux"][i]) < 0.2
        assert np.hypot(obj1["x"][i] - obj3["x"][i], obj1["y"][i] - obj3["y"][i]) < 0.5
        assert -0.2 < np.log(obj1["flux"][i] / obj3["flux"][i]) < 0.2
        assert np.hypot(obj1["x"][i] - obj4["x"][i], obj1["y"][i] - obj4["y"][i]) < 0.5
        assert -0.2 < np.log(obj1["flux"][i] / obj4["flux"][i]) < 0.2

    assert -5 <= len(obj1) - len(obj2) <= 5
    assert -5 <= len(obj1) - len(obj3) <= 5
    assert -5 <= len(obj1) - len(obj4) <= 5
    del obj3, obj4  # cleanup

    # Gain tests
    gain_txt = str(tmp_path) + "/gain.txt"
    g = np.linspace(1.4, 1.6, 1024).reshape((32, 32))
    _g = np.zeros((1024, 8))
    _g[:, 5] = g.ravel()
    np.savetxt(gain_txt, _g)

    out_array3g = process_electrons_to_DN_by_blob(
        csvfile=None,
        streaks=trajectory_data,
        n_pixels=4088,
        apply_gain=True,
        rng=rng,
        one_explicit=True,
        gain_txt=gain_txt,
        output_array_path=str(tmp_path) + "/e3p.npy",
    ).astype(np.float32, copy=False)
    out_array4g = process_electrons_to_DN_by_blob2(
        rng_ff=rng,
        csvfile=None,
        streaks=trajectory_data,
        n_pixels=4088,
        apply_gain=True,
        one_explicit=True,
        gain_txt=gain_txt,
        output_array_path=str(tmp_path) + "/e4p.npy",
    ).astype(np.float32, copy=False)

    # check agreement
    assert np.allclose(out_array3g, np.load(str(tmp_path) + "/e3p.npy"))
    assert np.allclose(out_array4g, np.load(str(tmp_path) + "/e4p.npy"))

    # now look at the gains
    obj3 = sep.extract(out_array3g * g[0, 16], 50.0, minarea=2)
    obj4 = sep.extract(out_array4g * g[0, 16], 50.0, minarea=2)

    for i in range(3):
        print(i)
        assert np.hypot(obj1["x"][i] - obj3["x"][i], obj1["y"][i] - obj3["y"][i]) < 0.5
        assert -0.2 < np.log(obj1["flux"][i] / obj3["flux"][i]) < 0.2
        assert np.hypot(obj1["x"][i] - obj4["x"][i], obj1["y"][i] - obj4["y"][i]) < 0.5
        assert -0.2 < np.log(obj1["flux"][i] / obj4["flux"][i]) < 0.2
