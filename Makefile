TARGETS = format lint test typecheck unittest

.PHONY: $(TARGETS)


format:
	@./format

lint:
	recipe/run_test.sh lint

test:
	recipe/run_test.sh

typecheck:
	recipe/run_test.sh typecheck

unittest:
	recipe/run_test.sh unittest

