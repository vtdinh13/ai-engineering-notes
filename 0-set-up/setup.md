# uv
1. `uv init -p python3.12` to create the scaffolding for a new project using python 3.12. Running this command will generate two files: 
    1. `pyproject.toml` contains:
        - project name
        - project metadata 
        - version
        - dependecy list
        - python version requirement
    2. `.python-version` pins the Python interpreter version to ensure the consistent interpreter version across machines.
2. `uv sync` creates a `.venv` virtual environment if one doesn't exist and the `uv.lock` lockfile.
    
    - `uv sync` installs the packages listed in `pyproject.toml`
    - The lockfile captures the exact versions of all direct dependencies to ensure that the same versions are installed across all machines. This gives reproducible results. `uv` also uses the lockfile to know whether the environment is up to date or needs changes. 
3. `uv add jupyterlab --dev` to add packages needed for development. This command will add the development version of `jupyterlab` to `pyproject.toml`. 
4. `uv sync` to install what's in `pyproject.toml`. `.venv` will also update to reflect these changes.
5. `uv run jupyter lab` to run jupyter lab

# Docker
    