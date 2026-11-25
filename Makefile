.PHONY: lint test test-regression

lint:
	python3 -m flake8 kmvmesh tests

test:
	python3 -m pytest tests

# The 3D cases need KMVMESH_SALT3D_DATA (see tests/regression/test_regression_3d.py).
test-regression:
	python3 -m pytest tests/regression -rs
