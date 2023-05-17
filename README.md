# CoLLAGE
Codon Likelihoods Learned Against Genome Evolution (CoLLAGE): a deep learning framework for identifying naturally selected patterns of codon preference within a species.

## Using CoLLAGE
### Web
The easiest way to use CoLLAGE is through our free web service located at WEBSITE_URL_TBD. You simply need to upload your FASTA, and the website handles the rest!

### Running locally
If you would like to run CoLLAGE locally, please install the dependencies in either `cpu_requirements.txt` or `cuda_requirements.txt` depending on whether CUDA is available on your system.

TODO(auberon): Add instructions on running scripts for CoLLAGE.
=======
### Test Setup
In addition to the needed dependencies for running CoLLAGE itself, please install `pytest`. Run one of the following commands depending on your package manager. (TODO(auberon): add requirements.txt(s))

```
pip install pytest
```
```
conda install -c anaconda pytest
```


TODO(auberon): Make package CoLLAGE for PyPI and make it pip installable.

### Developing
If you would like to contribute to the development of CoLLAGE, please see [CONTRIBUTING.md](CONTRIBUTING.md) for information on how to install and set up your dev environment.
=======
```
pytest
```

This should automatically find all tests and run them. You should expect to see an output that has the name of each test file and a green dot or a red F for each test. Green dots indicate passing, red Fs indicate failing. It is currently expected to see some warnings about deprecated PyTorch features.

If the tests fail, please make sure to either fix your code or update the tests before opening a PR / pushing your code.

## Formatting
We use the `autopep8` formatter. Please format your code before making PRs or pushing to main. We supply the following addtional arguments to ignore line length limits:
```
--ignore E501
```

### Formatting in VS Code
If you are using VS Code, you may find it convenient to have the formatting automatically applied when you save your files. You can do this with the below steps. If using a Mac, replace `ctrl` with `cmd`.
1. Install the `autopep8` [VS Code extension](https://marketplace.visualstudio.com/items?itemName=ms-python.autopep8)
1. Open your user `settings.json`. Hit `ctrl+shift+p` to open the command palette, then start typing `settings` and select `Preferences: Open User Settings(JSON)` when it pops up.
1. Add the following to your configuration (keeping your existing settings where it says `# YOUR OTHER SETTINGS`)
    ```
    {
        # YOUR_OTHER_SETTINGS
        "[python]": {
            # YOUR OTHER SETTINGS
            "editor.defaultFormatter": "ms-python.autopep8",
            "editor.formatOnSave": true,
        },
        "python.formatting.provider": "none",
        "autopep8.args": ["--ignore", "E501"],
    }
    ```
1. Restart VS Code
1. Check that it works by editing a Python file and adding a bunch of extra newlines to the end. When you save the file with `ctrl+s` it should remove the extra lines automatically!

