COMMIT_HASH := $(shell eval git rev-parse HEAD)

test:
	python -m pytest tests
