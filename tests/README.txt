On pushes and pull requests, github will run pytest. As a result all commits on github.com will have a green checkmark or red x depending on if the commit passed all tests. You can browse the "actions" tab of the AC repo on github.com to see the detailed output of the automated tests.

As a developer, you should manually run the tests before all commits.
Run "pytest" from the repo's base directory AdaptiveComputing (the place where the pytest directory is located). Only the output of failed tests will be displayed.
Run "pytest -s" to see output of tests that pass too.

All tests must be defined in file with file names that begin with "test_*.py"
The name of all test functions must begin with "test_"

Test Organization:
- tests/                        # Main test directory
  - evaluators_unit_tests/      # Unit tests for evaluator modules
  - samplers_unit_tests/        # Unit tests for sampler modules 
  - datasets_unit_tests/        # Unit tests for dataset modules (including masking functionality)
  - test_examples.py            # Integration tests that run example scripts

The datasets_unit_tests/ directory contains comprehensive tests for:
- DatasetBase masking functionality (NaN and out-of-bounds handling)
- Mask-aware surrogate training
- Integration between datasets and surrogate models
