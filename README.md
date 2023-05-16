# CoLLAGE
Codon Likelihoods Learned Against Genome Evolution (CoLLAGE): a deep learning framework for identifying naturally selected patterns of codon preference within a species

## Tests
Please run tests and ensure they pass before making a PR or pushing your code to main.

### Setup
In addition to the needed dependencies for running CoLLAGE itself, please install `pytest`. Run one of the following commands dependig on our package manager. (TODO(auberon): add requirements.txt(s))

```pip install pytest```
```conda install -c anaconda pytest```

### Running Tests
Navigate to the root of the repository on the command line and run:

```
pytest
```

This should automatically find all tests and run them. You should expect to see an output that has the name of each test file and a green dot or a red F for each test. Green dots indicate passing, red Fs indicate failing. It is currently expected to see some warnings about deprecated PyTorch features.

If the tests fail, please make sure to either fix your code or update the tests before opening a PR / pushing your code.