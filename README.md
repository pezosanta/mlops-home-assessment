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
To ensure code quality, the following tools/packages are being used as pre-commit hooks:
- `pre-commit`: file checkers (json, yaml, executables), end of file whitespace checker
- `flake8`: linter
- `black`: code formatter
- `isort`: Python imports formatter

The pre-commit hooks are configured via the `.pre-commit-config.yaml` file. 

Furthermore, the following files are used to configure the tools/packages:
- `pyproject.toml`: configure `black` and `isort` properties 
- `tox.ini`: configure `flake8` properties (`flake8` does not yet support `pyproject.toml` based configuration by default).

To install the pre-commit hooks locally, simply run the `requirements/requirements-dev.sh` script from anywhere via:
```bash
# Wherever you are
source <PATH_TO_THIS_REPO>/requirements/requirements-dev.sh
```

## 5. Python packaging
`setuptools` is being used for building a `Python` package from the `image_classifier` src folder, which is required for both development and production (currently this means GitHub Actions Workflow based model training). The currently used package versions are:
- `setuptools>=69.0`
- `build>=1.0.3`

To install these packages locally **in editable mode**, simply run the `requirements/requirements-dev.sh` script from anywhere via:
```bash
# Wherever you are
source <PATH_TO_THIS_REPO>/requirements/requirements-dev.sh
```

If you want to build a `wheel` locally, run the following command within the repository, that will create a `*.whl` and a `*.tar.gz` file under the `<PATH_TO_THIS_REPO>/dist/` folder:
```bash
python -m build
```

The following files are used by `setuptools` for building the package
- `pyproject.toml`: contains general information, build requirements and default logic on which python modules (`.py` files / folders with an `__init__.py` file in them) and non-python files (package data) must be contained in the package
- `MANIFEST.in`: contains a set of commands with glob-like pattern matching to include (or exclude) relevant files that `setuptools` does not catch by default with the `pyproject.toml` settings

For more information on `setuptools` based packaging, check the following links:
- https://setuptools.pypa.io/en/latest/userguide/quickstart.html
- https://packaging.python.org/en/latest/guides/writing-pyproject-toml/

## 6. Docker
To ensure reproducibility, model trainings are run in Docker containers in the training GitHub Actions Workflow runs.

The `docker` folder contains all the files that are necessary to build the Docker image both locally and in the GitHub Action Workflow runs:
- `Dockerfile`: the Docker image configuration file
- `configure.sh`: shell script for setting up local variables for Docker build
- `create_build_context.sh`: shell script for composing the Docker build context
    - copying the `requirements/requirements-prod.(txt/sh)` files to the `docker` folder
    - building the Python package and copying the resulting `dist/*.whl` file to the `docker` folder
- `cleanup_build_context.sh`: shell script for cleaning up the repository after Docker build:
    - deleting the `docker/requirements-prod.(txt/sh)` and `docker/*.whl` files
    - deleting the `dist` and `*.egg-info` folders from the project root
- `build.sh`: shell script for executing the `configure.sh`, `create_build_context.sh` shell scripts, the `docker build` command and finally the `cleanup_build_context.sh` shell script

To build the Docker image locally, simply just run the `build.sh` shell script from anywhere:
```bash
source <PATH_TO_THIS_REPO>/docker/build.sh
```

## 7. SOURCE / PACKAGE folder
The `image_classifier` folder is the SOURCE / PACKAGE folder that contains all the scripts/files relevant for model training.

To run a model training locally simply run the `image_classifier/main.py` script and set the arguments as needed:
```shell
# With Python3.10
python -m image_classifier.main \
    --experiment_name <your_experiment_name> \ # REQUIRED
    --run_name <your_run_name> \ # REQUIRED
    --dataset_folder <your_dataset> \ # OPTIONAL, defaults to data/cifar_dataset
    --learning_rate <your_learning_rate> \ # OPTIONAL
    --momentum <your_momentum> \ # OPTIONAL
    --num_epochs <your_num_epochs> \ # OPTIONAL
    --num_workers <your_num_workers> \ # OPTIONAL
    --batch_size <your_batch_size> \ # OPTIONAL
    --pretrained_model <path_to_pretrained_model_relative_to_experiments_folder> \ # OPTIONAL, e.g.: local_exp_test/run_1/model.pth 
    --db_pass <your_mocked_password> \ # REQUIRED
```

Note that instead of `python -m image_classifier.main` you can simply use `image_classifier_trainer` as an alias (see in `pyproject.toml`).

The experiments are tracked under the `experiments` folder. Each experiment run outputs a trained `model.pth` file and a corresponding `metrics.png` of the validation accuracy metrics.

Every `*.pth` and `*.png` file in the repository is versioned via `Git LFS` configured in `.gitattributes`.

## 8. GitHub Actions Workflows
Currently, 2 GitHub Actions Workflows are implemented and being used:
- **Code Quality checks with pre-commit** (`.github/workflows/pre-commit.yaml`): 
    - description: runs the configured pre-commit hooks against the modified files
    - triggers: automatic trigger at every git push event of all branches
- **Image Classifier Training** (`.github/workflows/training.yaml`):
    - description: builds the Docker image and runs the model training in Docker container. The experiment run outputs (`model.pth`, `metrics.png`) are published as build artifacts. If you wish to make a build artifact model available to be used in later workflow runs as `pretrained_model`, then simply 
        - download the build artifact
        - unzip it
        - copy the files into your local `<PATH_TO_THIS_REPO>/experiments/<your_experiment_name>/<your_run_name>/` folder
        - commit and push the new files
        - pass the `<your_experiment_name>/<your_run_name>/model.pth` as the `pretrained_model` argument at the next workflow trigger
    - triggers: manual (workflow_dispatch), model training input arguments can be configured on the UI
