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