TARGETS = devenv env format lint test typecheck unittest
DEVPKGS = $(shell cat devpkgs)
ENVNAME = pygraf

.PHONY: $(TARGETS)


devenv: env
	mamba install -y -n $(ENVNAME) $(DEVPKGS)

env:
	mamba env create -y -f environment.yml

format:
	@./format

lint:
	ruff check .

test: lint typecheck unittest

typecheck:
	mypy --install-types --non-interactive .

unittest:
	pytest --cov -k "not hrrr_maps" -n 4 .

memtest:
	pytest --memray -k "not hrrr" .

