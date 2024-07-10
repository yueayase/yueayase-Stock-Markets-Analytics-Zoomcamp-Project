# Final Project for 'Stock-Markets-Analytics-Zoomcamp'

## Setting Up the Project Environment (in Terminal)
* Change to your favorite working directory
* Create a new virtual environment (venv): `python3 -m venv venv` 
(If you use msys2 in Windows, then go to https://www.python.org/downloads/ and download python3.11.
Remeber to add system environment variable for the place your python installer is.
Finally, type: `py -3.11 -m venv {your_favorite_venv_name}`)
* Activate the new virtual environment: `{your_favorite_venv_name}/Scripts/activate`

* Install Ta-lib (on Mac):
  * Step 1: Install Homebrew (if not already installed): `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
  * Step 2: Install TA-Lib using Homebrew: `brew install ta-lib`
  * Step 3: Install the ta-lib Python package (After the TA-Lib C library is installed, you can proceed to install the Python package using pip. Make sure you're in your virtual environment if you're using one):
`pip3 install ta-lib`
  * Step 4: Make sure you have Numpy of version earliar than 2, so that ta-lib can be successfully imported (e.g. "numpy==1.26.4" in requirements.txt). [LINK](https://stackoverflow.com/questions/78634235/numpy-dtype-size-changed-may-indicate-binary-incompatibility-expected-96-from)

* Install Ta-lib (on Windows):
  * Step 1: Follow the instructions of this link: https://stackoverflow.com/a/75503202 to build the ta-lib 
  * Step 2: Create a new folder named `ta-lib` in the folder `C:\ta-lib\c\include`and then move all files to this new created folder
  * Step 3: run `python -m pip install ta-lib` 

* Install all requirements to the new environment (venv): `pip3 install -r requirements.txt`

## Running the Project
* Run `main.py` from the Terminal (or Cron) to simulate one new day of data.

## What is the aim of this project?
* Try different machine learning models to help us determine whether the stock price will grow in the future 5 days
* This project includes:
  * Use `PCA` to predict all collected data from `yFinance` and `pandas_datareader` apis and use it as a bew feature
  * Use `logistic regression` model to select important features to help our `random forest` trained faster and maintain the precision of the prediction
  * Use `random forest` model to train the model we finally want to export
  * Simulate 3 different portfolio management strategies: `equal weights`, `mean-variance optimization`, and `sharpe ratio optimization`
  * By the way, some known human prediction rules are used as the features before training the ML model(You can see them in `_define_dummies` function in `scripts/train.py`)