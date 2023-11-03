# CoLLAGE
Codon Likelihoods Learned Against Genome Evolution (CoLLAGE): a deep learning framework for identifying naturally selected patterns of codon preference within a species.

## Using CoLLAGE
### Web
The easiest way to use CoLLAGE is through our free web service located at WEBSITE_URL_TBD. You simply need to upload your FASTA, and the website handles the rest!

### Running locally
If you would like to run CoLLAGE locally, please install the dependencies in either `cpu_requirements.txt` or `cuda_requirements.txt` depending on whether CUDA is available on your system.

TODO(auberon): Add instructions on running scripts for CoLLAGE.

### Running the Docker image
TODO(auberon): give full instructions
Input file must be named input.fasta. TODO(auberon): remove this limitation.
```
docker run -v /absolute/path/to/input/folder:/input redcliffesalaman/collage-model
```

### Developing
If you would like to contribute to the development of CoLLAGE, please see [CONTRIBUTING.md](CONTRIBUTING.md) for information on how to install and set up your dev environment.
