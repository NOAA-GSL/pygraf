TARGETS = format lint test typecheck unittest

.PHONY: $(TARGETS)


format:
	@./format

lint:
	ruff check .

test: lint typecheck unittest

typecheck:
	mypy --install-types --non-interactive .

unittest:
	pytest --cov -k "not hrrr" -n 4 .

