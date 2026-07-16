import pytest

from battery_benchmark.experiments.vidal_case2 import Case2Config, resolve_case2_run_dir


def test_case2_smoke_and_formal_paths_are_directional(tmp_path):
    smoke = resolve_case2_run_dir(
        tmp_path,
        Case2Config("lg_hg2", "samsung_30t", source_checkpoint=tmp_path / "x.pt"),
        tmp_path,
    )
    formal = resolve_case2_run_dir(
        tmp_path,
        Case2Config(
            "lg_hg2", "samsung_30t", run_mode="formal", epochs=515,
            max_train_windows=None, development_only=False, source_checkpoint=tmp_path / "x.pt",
        ),
        tmp_path,
    )
    assert "_smoke" in smoke.parts
    assert "_smoke" not in formal.parts
    assert "source_lg_hg2__target_samsung_30t" in str(smoke)
    formal.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        resolve_case2_run_dir(
            tmp_path,
            Case2Config(
                "lg_hg2", "samsung_30t", run_mode="formal", epochs=515,
                max_train_windows=None, development_only=False, source_checkpoint=tmp_path / "x.pt",
            ),
            tmp_path,
        )
