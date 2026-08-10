from types import SimpleNamespace

import numpy as np
import pandas as pd
from scripts.benchmarks.run_tatn_cmu import _feature_range, _to_shen_cycles


def test_cmu_tatn_uses_source_feature_range_without_target_fitting():
    frame = pd.DataFrame(
        {
            "current": np.zeros(90),
            "voltage": np.full(90, 3.0),
            "temperature": np.full(90, 30.0),
            "soc": np.linspace(1.0, 0.0, 90),
        }
    )
    record = SimpleNamespace(
        data=frame,
        metadata={"mission_cycle_id": "mission_1", "source_path": "cell.parquet"},
    )

    cycle = _to_shen_cycles(
        [record],
        np.array([-1.0, 2.0, 20.0]),
        np.array([1.0, 4.0, 25.0]),
        require_labels=True,
    )[0]

    np.testing.assert_allclose(cycle.features[0], [0.5, 0.5, 2.0])
    np.testing.assert_allclose(cycle.soc, frame["soc"])


def test_target_feature_range_is_fitted_from_adaptation_records_only():
    adaptation = SimpleNamespace(
        data=pd.DataFrame(
            {
                "current": [1.0, 3.0],
                "voltage": [2.0, 4.0],
                "temperature": [20.0, 30.0],
            }
        )
    )

    feature_min, feature_max = _feature_range([adaptation])

    np.testing.assert_allclose(feature_min, [1.0, 2.0, 20.0])
    np.testing.assert_allclose(feature_max, [3.0, 4.0, 30.0])
