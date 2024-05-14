# MLOps home assessment
This repository is created for a job interview home assessment. 

Only a Jupyter Notebook was received containing an imaginary Data Scientist's image classification model training scripts. The task was to make those training scripts reproducible in a professional(ish) way.
For more information, see the original Jupyter Notebook (`notebooks/mlops_project_classifier.ipynb`).

## 1. Repository structure
The repository structure is based on [Cookiecutter Data Science guideline](https://drivendata.github.io/cookiecutter-data-science/).

```bash
mlops-home-assessment
├── .git                        # Git related things
│   ├── hooks                   # Installed pre-commit hooks
│   └── ...
├── .gitattributes              # Git LFS config
├── .github                     # CODEOWNERS, GitHub Actions Workflow definitions
│   ├── CODEOWNERS
│   └── workflows
│       ├── pre-commit.yaml
│       └── training.yaml
├── .gitignore                  # gitignore
├── .pre-commit-config.yaml     # Pre-commit hooks related settings
├── MANIFEST.in                 # Python packaging related
├── README.md                   # This file
├── data                        # Folder containing the different datasets
│   └── cifar_dataset
│       ├── cifar-10-batches-py
│       │   ├── batches.meta
│       │   ├── data_batch_1
│       │   ├── data_batch_2
│       │   ├── data_batch_3
│       │   ├── data_batch_4
│       │   ├── data_batch_5
│       │   ├── readme.html
│       │   └── test_batch
│       └── metadata.json
├── docker                      # Folder containing the Docker image build context
│   ├── Dockerfile
│   ├── build.sh
│   ├── cleanup_build_context.sh
│   ├── configure.sh
│   └── create_build_context.sh
├── experiments                 # Folder for tracking the experiments
│   └── local_exp_test
│       └── run_1
│           ├── metrics.png
│           └── model.pth
├── image_classifier            # The Python package folder (src)
│   ├── __init__.py
│   ├── data                    # Folder containing data loader/handling scripts
│   │   ├── __init__.py
│   │   ├── data.py
│   │   └── db_connect.yaml
│   ├── main.py                 # Main script for training
│   ├── model                   # Folder containing scripts for model training
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── training.py
│   │   └── validation.py
│   └── visualization           # Folder containing visualization scripts
│       └── plot.py
├── notebooks                   # Folder containing Jupyter Notebooks
│   └── mlops_project_classifier.ipynb
├── pyproject.toml              # Python packaging / pre-commit hooks related
├── requirements                # Folder containing requirements files
│   ├── requirements-build.in
│   ├── requirements-build.sh
│   ├── requirements-build.txt
│   ├── requirements-cq.in
│   ├── requirements-cq.sh
│   ├── requirements-cq.txt
│   ├── requirements-dev.in
│   ├── requirements-dev.sh
│   ├── requirements-dev.txt
│   ├── requirements-prod.in
│   ├── requirements-prod.sh
│   └── requirements-prod.txt
└── tox.ini                     # Pre-commit hooks related
```

## 2. Prerequisites
Note that for developing, `Ubuntu 22.04 (WSL)` and `VSCode` was being used.

### 2.1. Python
Currently, `Python3.10` is used. Please use this version in your `Dev Containers` and `virtualenvs`.

### 2.2. Docker
If you wish to build and run the Docker images on your local machine, then you will need `Docker` to be installed. See more information about Docker installation in the [official Docker installation guide](https://docs.docker.com/engine/install/ubuntu/).

## 3. Requirements

### 3.1. About `pip-tools`
`pip-tools` is a package management CLI tool, that helps keeping pip-based packages consistent and compatible with one another. When generating a `requirements.txt` file via `pip-tools`, it includes all the implicit dependencies of each of the specified packages using the latest possible package versions that avoid any conflicts. Furthermore, it uses verbose, human-readable comments.

See more info about [pip-tools](http://github.com/jazzband/pip-tools)

**Currently, `pip-tools==7.3.0` version is used for generating the `requirements-*.txt` files from the `requirements-*.in` files.** 

To do it, specify all the required packages in the `requirements-*.in` files (**specify package versions only if absolutely necessary**) and then generate the `requirements-*.txt` files as follows:

```bash
# With Python3.10
cd <PATH_TO_THIS_REPO>/requirements
pip install pip-tools==<CURRENTLY_USED_VERSION>
# Use build/cq/dev/prod instead of "*"
pip-compile --output-file requirements-*.txt requirements-*.in
```

### 3.2. Different requirements
Currently, 4 different `requirements` are being used:
- **build**: containing the necessary packages for Python package building
- **code quality (cq)**: containing the packages for the code quality checking related pre-commit hooks
- **development (dev)**: containing packages necessary for model development
- **production (prod)**: containing only the packages that are necessary for training and production

For each requirements, there is a `requirements-*.sh` shell script which with one can install the relevant requirements with `pip`. The reason behind using shell scripts is that this way one can install multiple `requirements-*.txt` files, e.g. `requirements-dev.sh` installs:
- `requirements-cq.txt`
- `requirements-build.txt`
- `requirements-dev.txt`
- the repository in editable mode (`pip install -v -e <PATH_TO_THIS_REPO>`)

## 4. Pre-commit hooks
