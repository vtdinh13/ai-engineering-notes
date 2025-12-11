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
6. Use the same virtual environment as a Jupyter kernel:
    1. `uv add ipykernel --dev` to install the IPython kernel package inside the VM. But add this to the development kit because it will not be utilized in production.
    2. `uv run python -m ipykernel install --use --name ai-engineering-notes --display-name "ai-engineering-notes"` to register the kernel.
    3. Select this kernel when starting a Jupyter notebook.
7. To remove `uv` -> remove the auto-generated files it created with `uv init`.
    1. `rm pyproject.toml uv.lock` to remove generated files.
    2. `rm -rf .venv` to remove the VM

# Docker
1. Each service now uses the default Docker volumes by mounting to the container paths directly. For example in postgres: `- /var/lib/postgresql/data`. 

    1. To mount to a specified named volume: 
        - declare the name prior to the container path: `postgres_data:/var/lib/postgresql/data`. 
        - declare named volumes at the bottom of the YML file. 
        ``` docker-compose.yml
            volumes:
                postgres_data:
                pgadmin_data:
                elasticsearch_data:
                opensearch_data:
        ```
