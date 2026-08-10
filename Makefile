.PHONY: paper test verify

PYTHON ?= python3

paper:
	PYTHONDONTWRITEBYTECODE=1 PATH=/opt/homebrew/bin:$$PATH PYTHONPATH=benchmark \
		$(PYTHON) -m benchmark_tool.road_paper_cli --paper-dir paper

test:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=benchmark \
		$(PYTHON) -m pytest -q -p no:cacheprovider \
		benchmark/tests/test_road_report.py \
		benchmark/tests/test_road_paper_cli.py

verify: test
	$(PYTHON) scripts/verify-public-release.py .
