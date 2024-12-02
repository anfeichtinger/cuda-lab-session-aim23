# Installation
#### Make sure python 3.9 or later is installed. Can be installed from Windows Store.

```bash
python --version
```

> Note: when there are multiple python versions installed you may need to run all commands with `python3.9` instead of `python` and `pip3.9` instead of `pip`

#### Install venv in python.

```bash
pip install virtualenv
```

#### Close rider and open the project folder with VSCode.

#### Open the terminal in VSCode and switch to the task2 branch.

```bash
git checkout task2
```

#### Create a new virtual environment and switch to it.

```bash
python -m venv deps
.\deps\Scripts\Activate.ps1
```

#### You should see (deps) in front of your shell now.
#### Download and install pytorch and its dependencies.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install ipython ipykernel
```

#### Install the Jupyter extension for VSCode.

#### Open the `kernel.ipynb` file, click on "Select Kernel" -> "Python environments" and choose your environment "deps". If the option is not showing up close the file and open it again. If it is still not working restart the project in VSCode and run this command again: `.\deps\Scripts\Activate.ps1`.

#### You should now be able to run the jupyter notebook. 🚀