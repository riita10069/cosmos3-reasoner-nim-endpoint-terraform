from pathlib import Path

from benchmark_tool.road_paper_cli import INPUT_ARGUMENTS


def test_paper_cli_declares_all_locked_inputs() -> None:
    assert len(INPUT_ARGUMENTS) == 21
    assert len(set(INPUT_ARGUMENTS.values())) == 21
    assert all(Path(name).suffix == ".json" for name in INPUT_ARGUMENTS.values())
