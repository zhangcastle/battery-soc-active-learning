from battery_benchmark.reproductions.vidal_2019.case2 import Case2Config, run_case2
from tests.reproductions.vidal_2019.vidal_case1_test_utils import make_case1_checkpoint, make_case1_run, make_v2_split


def test_case2_scaler_depends_on_target_train_not_source_or_test(tmp_path):
    checkpoint_a = make_case1_checkpoint(tmp_path / "source_a.pt", "lg_hg2", formal=True)
    checkpoint_b = make_case1_checkpoint(tmp_path / "source_b.pt", "panasonic_18650pf", formal=True)
    case1_root = tmp_path / "case1"
    make_case1_run(case1_root, "samsung_30t", formal=True)
    roots = []
    for name, test_n in (("a", 9), ("b", 21)):
        root = tmp_path / f"data_{name}"
        make_v2_split(root, "samsung_30t", "full_train", [(1, 31, "PASS", "drive_cycle", "Mixed1")], timestamp_step=0.1)
        make_v2_split(root, "samsung_30t", "test", [(1, test_n, "PASS", "drive_cycle", "UDDS")])
        roots.append(root)
    cfg_a = Case2Config("lg_hg2", "samsung_30t", epochs=1, batch_size=2, seq_len=4, stride=4, max_train_windows=1, source_checkpoint=checkpoint_a)
    cfg_b = Case2Config("panasonic_18650pf", "samsung_30t", epochs=1, batch_size=2, seq_len=4, stride=4, max_train_windows=1, source_checkpoint=checkpoint_b)
    result_a = run_case2(cfg_a, roots[0], tmp_path / "out_a", tmp_path, "cpu", case1_output_root=case1_root)
    result_b = run_case2(cfg_b, roots[1], tmp_path / "out_b", tmp_path, "cpu", case1_output_root=case1_root)
    assert result_a["scaler"]["feature_min"] == result_b["scaler"]["feature_min"]
    assert result_a["scaler"]["feature_max"] == result_b["scaler"]["feature_max"]
    assert result_a["scaler"]["target_scaler_refit"] is True
    assert result_a["scaler"]["source_scaler_used"] is False
