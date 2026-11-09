import pandas as pd

# Change this import if your analysis script has a different filename.
from cr_event_analysis import add_is_sim_flag


def test_flags_detection_inside_simulated_bounding_box():
    """A detection inside the injected footprint and in the same frame is simulated."""
    detections_df = pd.DataFrame({
        "frame": [3, 3, 4],
        "y": [105, 150, 105],
        "x": [205, 250, 205],
    })

    sim_truth_df = pd.DataFrame({
        "frame": [3],
        "y0": [100],
        "y1": [110],
        "x0": [200],
        "x1": [210],
    })

    result = add_is_sim_flag(
        detections_df,
        sim_truth_df,
        padding=0,
    )

    assert result["is_sim"].tolist() == [True, False, False]


def test_requires_matching_frame():
    """Matching coordinates in a different frame must not be flagged."""
    detections_df = pd.DataFrame({
        "frame": [2, 7],
        "y": [25, 25],
        "x": [35, 35],
    })

    sim_truth_df = pd.DataFrame({
        "frame": [2],
        "y0": [20],
        "y1": [30],
        "x0": [30],
        "x1": [40],
    })

    result = add_is_sim_flag(
        detections_df,
        sim_truth_df,
        padding=0,
    )

    assert result["is_sim"].tolist() == [True, False]


def test_padding_expands_matching_region():
    """A detection just outside the footprint can match when padding is used."""
    detections_df = pd.DataFrame({
        "frame": [5, 5, 5],
        "y": [49, 47, 61],
        "x": [75, 75, 75],
    })

    sim_truth_df = pd.DataFrame({
        "frame": [5],
        "y0": [50],
        "y1": [60],
        "x0": [70],
        "x1": [80],
    })

    result = add_is_sim_flag(
        detections_df,
        sim_truth_df,
        padding=2,
    )

    # y=49 is within the two-pixel padding.
    # y=47 is too far above the box.
    # y=61 is within the two-pixel padding below the box.
    assert result["is_sim"].tolist() == [True, False, True]


def test_none_or_empty_truth_table_flags_everything_false():
    """Runs without injected simulation truth should still receive the column."""
    detections_df = pd.DataFrame({
        "frame": [0, 1],
        "y": [10, 20],
        "x": [30, 40],
    })

    result_none = add_is_sim_flag(
        detections_df,
        sim_truth_df=None,
    )

    empty_truth_df = pd.DataFrame(
        columns=["frame", "y0", "y1", "x0", "x1"]
    )
    result_empty = add_is_sim_flag(
        detections_df,
        sim_truth_df=empty_truth_df,
    )

    assert result_none["is_sim"].tolist() == [False, False]
    assert result_empty["is_sim"].tolist() == [False, False]
    assert result_none["is_sim"].dtype == bool
    assert result_empty["is_sim"].dtype == bool


def test_original_detection_dataframe_is_not_modified():
    """The helper should return a copy rather than changing its input."""
    detections_df = pd.DataFrame({
        "frame": [1],
        "y": [10],
        "x": [20],
    })

    original_df = detections_df.copy(deep=True)

    sim_truth_df = pd.DataFrame({
        "frame": [1],
        "y0": [5],
        "y1": [15],
        "x0": [15],
        "x1": [25],
    })

    result = add_is_sim_flag(
        detections_df,
        sim_truth_df,
        padding=0,
    )

    pd.testing.assert_frame_equal(detections_df, original_df)
    assert "is_sim" not in detections_df.columns
    assert result["is_sim"].tolist() == [True]


def test_empty_detection_dataframe_gets_boolean_column():
    """An empty input should return an empty dataframe with the new column."""
    detections_df = pd.DataFrame(
        columns=["frame", "y", "x"]
    )

    sim_truth_df = pd.DataFrame({
        "frame": [0],
        "y0": [0],
        "y1": [10],
        "x0": [0],
        "x1": [10],
    })

    result = add_is_sim_flag(
        detections_df,
        sim_truth_df,
    )

    assert result.empty
    assert "is_sim" in result.columns
    assert result["is_sim"].dtype == bool
