# Setup

## Requirements
If you are interesting in developing, you will need to install the dev dependencies. Use either `cpu_dev_requirements.txt` or `cuda_dev_requirements.txt` depending on whether CUDA is available on your system.

### Updating Requirements
If you need to add a dependency, DO NOT modify the top level requirements files. Instead, do the following:

1. Add the new dependency to the appropriate file in `tools/requirements` and save it.
1. Change your working directory to `tools/requirements` 
1. Run `./generate_reqs.sh`
1. Make a PR for your changes.

# Tests
Please run tests and ensure they pass before making a PR or pushing your code to main.

## Running Tests
Navigate to the root of the repository on the command line and run:

```
pytest
```

This should automatically find all tests and run them. You should expect to see an output that has the name of each test file and a green dot or a red F for each test. Green dots indicate passing, red Fs indicate failing. It is currently expected to see some warnings about deprecated PyTorch features.

If the tests fail, please make sure to either fix your code or update the tests before opening a PR / pushing your code.