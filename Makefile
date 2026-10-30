.PHONY: test compile

test:
	PYTHONPATH=src python -m unittest discover -s tests -v

compile:
	python -m compileall -q src tests
