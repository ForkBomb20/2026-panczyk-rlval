import pytest

# only run these tests if the user has gym installed (and is therefore trying to run RL frameworks in experiments)
pytest.importorskip("gymnasium")
