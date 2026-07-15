import pytest

from battery_benchmark.experiments.vidal_case1 import Case1Config, resolve_run_dir


def test_smoke_and_formal_paths_are_separate(tmp_path):
    smoke = resolve_run_dir(tmp_path, Case1Config("lg_hg2"), tmp_path)
    formal_cfg = Case1Config("lg_hg2", run_mode="formal", epochs=515, max_train_windows=None, development_only=False)
    formal = resolve_run_dir(tmp_path, formal_cfg, tmp_path)
    assert "_smoke" in smoke.parts
    assert "_smoke" not in formal.parts
    formal.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        resolve_run_dir(tmp_path, formal_cfg, tmp_path)
