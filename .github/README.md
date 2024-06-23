# GitHub folder

This folder contains `GitHub` related files/configs.

## 1. CODEOWNERS file
In this file, you can configure for each file/folder in the repository the mandatory reviewers in case of a PR. It is important that a CODEOWNERS file is always valid within its branch. Therefore, if you modify it in a feature branch and then you open a PR to the default branch, the CODEOWNERS file in the default branch will be used in that PR. This means that for the same files/folders you can configure different OWNERS for different branches. 

For more info about CODEOWNERS files, check the file itself or the official guideline: https://help.github.com/articles/about-codeowners/

## 2. Workflows
The `workflows` folder contains the definition/configuration files of the `GitHub Actions workflows`. 

### 2.1 Code Quality check (`pre-commit.yaml`)
This workflow is triggered by every git commit push (for all branches) and runs the pre-commit hooks ensuring that the code/file modifications of each pushed commits meet our code quality standards.

These pre-commit hooks are just static file checkers (so our codes/scripts are not executed), therefore, it is enough to run them in a virtualenv (so no docker images are needed). To speed up the virtualenv creation, we use (poetry based) caching, which means that
- if the `poetry.lock` file has not changed since last run, the virtualenv is used from the cache that has all the packages installed
- if the `.pre-commit-config.yaml` file has not changed since last run, the virtualenvs for each of the pre-commit hooks will be restored from the cache (pre-commit creates a separate virtualenv for each of the hooks)

When setting up the main virtualenv, the `Python` version is obtained from the `pyproject.toml` file.

For more information, check the `pre-commit.yaml` file.
