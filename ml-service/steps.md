# How to train the models on this machine (step-by-step, no coding knowledge needed)


This guide walks you through every single step to train all the models for
this project on a new/different computer. Just follow the steps in order,
copy-pasting the commands exactly as written. You do not need to understand
what the commands do — just type/paste them and press Enter.


Everything in **`code blocks like this`** is a command you type into a
terminal window. Everything else is an explanation.


---


## Before you start


You need:
1. This whole project folder (`project234`, containing the `ml-service`
   folder) copied onto the new computer.
2. **Python** installed on the new computer (version 3.11, 3.12, or 3.13).
   - To check if Python is already installed, open a terminal (see next
     section) and type:
     ```
     python --version
     ```
   - If you see something like `Python 3.13.14`, you're good — skip to
     "Step 0: Open a terminal in the right folder".
   - If you see an error like "python is not recognized", you need to
     install Python first: go to https://www.python.org/downloads/ , click
     the big "Download Python" button, run the installer, and **make sure
     to tick the checkbox that says "Add Python to PATH"** during
     installation. After installing, close and reopen your terminal and try
     the `python --version` command again.


---


## Step 0: Open a terminal in the right folder


A "terminal" (also called "command prompt" or "PowerShell") is a black or
white window where you type commands instead of clicking things.


**On Windows:**
1. Open File Explorer and navigate to the `ml-service` folder inside the
   project folder (`project234\ml-service`).
2. Click once on the empty white space inside the folder (not on any file).
3. Type `cmd` and press Enter — a dark terminal window will open, already
   pointed at this folder. (Alternative: right-click inside the folder and
   look for an option like "Open in Terminal" or "Open PowerShell window
   here".)
4. To check you're in the right place, type:
   ```
   dir
   ```
   You should see files like `train.py`, `requirements.txt`, and folders
   like `data`, `models`, `api`.


Keep this terminal window open — you'll type every command below into it,
one at a time, pressing Enter after each one and waiting for it to finish
before typing the next.


---


## Step 1: Create a clean, isolated Python environment


This creates a private "sandbox" for this project's packages, so it doesn't
interfere with anything else on the computer.


```
python -m venv venv
```


This takes a few seconds and produces no visible output when it's done —
that's normal.


Now "activate" it (this tells the terminal to use this private sandbox):


**On Windows (cmd or PowerShell):**
```
venv\Scripts\activate
```


After running this, you should see `(venv)` appear at the start of the
line in your terminal. This means it worked. **You need to do this every
time you close and reopen the terminal**, before running any other command
in this guide.


---


## Step 2: Install all the required packages


This downloads and installs everything the project needs (this step needs
an internet connection, and can take several minutes — some packages,
especially TensorFlow, are large downloads):


```
pip install -r requirements.txt
```


Wait for it to finish. You'll see a lot of scrolling text — that's normal.
At the end you should see a line like `Successfully installed ...` with a
long list of package names.


**If you see an error mentioning "Long Path" or "WinError 3" or
"filename too long"**: this is a known Windows issue with one of the
packages (TensorFlow). See the "Troubleshooting" section at the bottom of
this guide for the fix, then re-run the `pip install -r requirements.txt`
command above.


**If you see an error about a missing compiler or "Microsoft Visual
C++"**: see "Troubleshooting" below.


Once this step finishes with no errors, you're ready to start training.


---


## Step 3: Generate the training data


This creates the synthetic dataset the models learn from.


```
python data\generate_data.py
```


Expected output (takes a few seconds):
```
Generated 219000 rows for 25 households over 365 days.
Saved to data/synthetic_energy_data.csv
```
If you see this, it worked. Move to the next step.


---


## Step 4: Create the charts/plots (optional but recommended)


This creates some visual charts to sanity-check the data looks right. It
doesn't train anything, just makes pictures.


```
python notebooks\eda.py
```


Expected output (takes a few seconds): some text about missing values and
statistics, ending with:
```
Saved plots to notebooks/plots/
```
You can open the `notebooks\plots` folder afterward and double-click the
`.png` image files to view them if you're curious.


---


## Step 5: Train the "Phase 1" baseline models


This is the main, first round of model training. **This is the slowest
step so far and may take a while** (could be anywhere from several minutes
to longer, depending on the computer) — just let it run, don't close the
terminal.


```
python train.py
```


**How to tell how far along it is:** this step trains 5 different models
for "solar", then 5 more for "consumption" (10 total), one at a time. As
each one starts and finishes, you'll see a line like this appear:
```
[1/5] Training LinearRegression for solar generation (solar_generation_kwh)...
    done in 4.2s - R2=0.9465
[2/5] Training SVR for solar generation (solar_generation_kwh)...
    done in 612.8s - R2=0.9992
[3/5] Training DecisionTreeRegressor for solar generation (solar_generation_kwh)...
    done in 3.1s - R2=0.9993
...
```
The `[2/5]`, `[3/5]` etc. tell you which model out of 5 it's currently on
for the current target ("solar" comes first, then it starts over at
`[1/5]` for "consumption" — so you'll see the `[1/5]` counter twice in
total, once per target). The **`SVR` model is normally by far the slowest
one** — if it's sitting on `[2/5] Training SVR...` for a long time with no
"done" line yet, that is completely normal, just keep waiting.


After all 10 models are done, you'll see two result tables print out
(one per target), followed by these two final lines:
```
Saved best solar model (...) to models/best_solar_model.pkl
Saved best consumption model (...) to models/best_consumption_model.pkl
```


**Do not move to the next step until you see those final "Saved..."
lines and the terminal gives you back a prompt to type into** (i.e. it's
no longer showing a spinning/frozen state).


---


## Step 6: Train the LightGBM model


```
python models\train_lightgbm.py
```


Expected output (should be fairly quick):
```
=== LightGBM for solar generation (solar_generation_kwh) ===
RMSE=... MSE=... MAE=... R2=...
=== LightGBM for consumption (consumption_kwh) ===
RMSE=... MSE=... MAE=... R2=...
```
(The `...` will be actual numbers — that's normal and expected.)


---


## Step 7: Train the CatBoost model


```
python models\train_catboost.py
```


Just like Step 6, this is fast (no special setup needed). Expected output:
```
=== CatBoost for solar generation (solar_generation_kwh) ===
RMSE=... MSE=... MAE=... R2=...
=== CatBoost for consumption (consumption_kwh) ===
RMSE=... MSE=... MAE=... R2=...
```


---


## Step 8: Train the Prophet model


```
python models\train_prophet.py
```


The first time this runs, it may print some one-time setup messages and
take a little longer than usual (up to a minute or two) — this is normal,
it's compiling something in the background the first time only.


Expected output ends with 6 lines that look like:
```
Prophet_hh4_small (solar_generation_kwh): RMSE=... MAE=... R2=...
Prophet_hh0_medium (solar_generation_kwh): RMSE=... MAE=... R2=...
Prophet_hh3_large (solar_generation_kwh): RMSE=... MAE=... R2=...
Prophet_hh4_small (consumption_kwh): RMSE=... MAE=... R2=...
Prophet_hh0_medium (consumption_kwh): RMSE=... MAE=... R2=...
Prophet_hh3_large (consumption_kwh): RMSE=... MAE=... R2=...
```


**If instead you see a line saying `Prophet unavailable ... falling back
to statsmodels SARIMAX`**, that's OK too — it means Prophet couldn't set
itself up on this computer (this can happen behind certain company/office
internet networks), so the program automatically switched to a backup
method. You'll instead see 6 lines starting with `SARIMAX_fallback_...`.
Either way is a successful run — move to the next step.


---


## Optional but recommended: use your GPU to make Step 9 much faster


**Skip this box and go straight to Step 9 if you don't have an NVIDIA
graphics card, or don't want to bother with extra setup** — Step 9 will
still work fine without it, just slower (could take a few hours on CPU
instead of ~10-20 minutes on a GPU).


If your computer has an NVIDIA graphics card (e.g. an RTX 3050 or similar),
by default this project will **NOT** use it — the version of the training
software this project installs only knows how to use NVIDIA graphics cards
on a Windows computer through something called "WSL2" (a way of running a
Linux system inside Windows). Setting it up is a one-time process:


### GPU Step A: Turn on WSL2 (one-time, needs a restart)


1. Click the Windows Start button, type `powershell`, then right-click
   "Windows PowerShell" and choose **"Run as administrator"**. Click "Yes"
   if asked for permission.
2. In the blue window that opens, type:
   ```
   wsl --install
   ```
   and press Enter. This downloads and sets some things up — it can take
   several minutes and will show scrolling text. When it's done (or asks
   you to), **restart your computer**.
3. After restarting, a black "Ubuntu" window should open by itself (if it
   doesn't, click Start, type `Ubuntu`, and open it). The first time, it
   will ask you to create a username and password for this Linux system —
   these can be anything you like (they're separate from your Windows
   login) and won't be shown as you type the password, which is normal.
   **Remember what you typed** — you'll need it later if asked.


### GPU Step B: Check your graphics card is visible


In that same Ubuntu window, type:
```
nvidia-smi
```
You should see a table with your graphics card's name in it (e.g. "RTX
3050"). If you see this, your graphics card is ready to use — you do
**not** need to install any separate NVIDIA driver inside Ubuntu; it uses
the one already installed on Windows.


If this command says "not found" or shows an error, your Windows NVIDIA
driver may be outdated — go to
https://www.nvidia.com/Download/index.aspx , download and install the
latest driver for your graphics card, restart, and try `nvidia-smi` again.


### GPU Step C: Set up Python inside Ubuntu


Still in the Ubuntu window, type each of these one at a time, pressing
Enter after each and waiting for it to finish (it may ask you to press `Y`
or type your Ubuntu password partway through — that's normal):
```
sudo apt update
sudo apt install python3 python3-pip python3-venv -y
```


### GPU Step D: Go to the project folder from inside Ubuntu


Ubuntu can see your normal Windows files under a special `/mnt/c/` path.
Type (adjust only if your project is in a different location than shown
in Step 0 above — the parts after `/mnt/c/` mirror your normal Windows
folder path):
```
cd "/mnt/c/Users/Singh3 Shruti/OneDrive - The Boston Consulting Group, Inc/Desktop/project234/ml-service"
```
To check you're in the right place:
```
ls
```
You should see the same files/folders as in Step 0 (`train.py`, `data`,
`models`, etc.).


### GPU Step E: Create a separate environment and install everything (with GPU support)


This needs its own environment, separate from the Windows one from Step 1
— Windows and Ubuntu can't share the same one.
```
python3 -m venv venv_gpu
source venv_gpu/bin/activate
pip install -r requirements.txt
pip install tensorflow[and-cuda]==2.20.0
```
The last line is what adds real GPU support on top of the plain version
from `requirements.txt`. This step needs internet access and can take
several minutes.


To check it worked, type:
```
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
If you see something like `[PhysicalDevice(name='/physical_device:GPU:0', ...)]`
(anything with `GPU` in it, not an empty `[]`), your graphics card is
ready to be used for training.


**From now on, whenever you come back to do more training later, you only
need to reopen the Ubuntu window and run:**
```
cd "/mnt/c/Users/Singh3 Shruti/OneDrive - The Boston Consulting Group, Inc/Desktop/project234/ml-service"
source venv_gpu/bin/activate
```
**...before running Step 9 below** — you don't need to repeat Steps A-E
again.


---


## Step 9: Train the LSTM model


**This is the slowest and most demanding step in the whole guide** (unless
you set up the optional GPU section above, in which case it should be much
quicker — around 10-20 minutes instead of potentially a few hours). Let it
run undisturbed — don't close the terminal or put the computer to sleep.


**If you set up the GPU section above**, run this command in the
**Ubuntu** window (with `venv_gpu` activated, per GPU Step E):
```
python3 models/train_lstm.py
```


**If you skipped the GPU section**, run this command in your normal
**Windows** terminal from Step 0 instead (note the backslash instead of
forward slash):
```
python models\train_lstm.py
```


Either way, you'll see a lot of scrolling progress output showing
"epochs" (training rounds) — something like:
```
Epoch 1/50
...
Epoch 2/50
...
```
This is completely normal — just let it keep going. It may stop before
reaching "50" on its own if it decides it's learned enough (this is
expected and called "early stopping" — not an error).


Expected final output:
```
LSTM (solar_generation_kwh): RMSE=... MAE=... R2=...
LSTM (consumption_kwh): RMSE=... MAE=... R2=...
```


The saved model files land in the same `models` folder either way (Ubuntu
and Windows share the same underlying files on your C: drive), so **you
can switch back to your normal Windows terminal for every step after
this one** — nothing further needs to run inside Ubuntu.


---


## Step 10: Train the combined "ensemble" model


This combines several of the previous models together. It needs Steps 6,
7, and 9 to have already finished successfully.


```
python models\train_ensemble.py
```


Expected output includes lines like:
```
[solar_generation_kwh] lightgbm included as base model: True/False
[solar_generation_kwh] catboost included as base model: True/False
Ensemble (solar_generation_kwh) base models=[...]: RMSE=... MAE=... R2=...
Ensemble (consumption_kwh) base models=[...]: RMSE=... MAE=... R2=...
```


---


## Step 11: Create the final results tables and charts


This gathers every model's results from all the previous steps into two
easy-to-read summary tables and two charts.


```
python results\model_comparison.py
```


Expected output: two tables printed to the screen (one for solar, one for
consumption), each listing every model tried so far sorted from best to
worst, plus a line at the end:
```
Saved comparison CSVs and R2 bar charts to results/
```


**You're done training!** You can find:
- The result tables: `results\solar_model_comparison.csv` and
  `results\consumption_model_comparison.csv` (open with Excel).
- The result charts: `results\solar_r2_comparison.png` and
  `results\consumption_r2_comparison.png` (double-click to view).


---


## Step 12 (optional): Start the prediction service and try it out


This starts a small web service that lets you ask the trained models for
predictions.


```
uvicorn api.main:app --reload
```


Expected output ends with something like:
```
INFO:     Application startup complete.
```
Leave this terminal window open and running.


To try it out, open a web browser and go to:
```
http://127.0.0.1:8000/docs
```
This opens an interactive page where you can click "Try it out" on any of
the listed endpoints, fill in some example numbers, and click "Execute" to
see a prediction come back.


To stop the service, click back into the terminal window and press
`Ctrl + C`.


---


## Quick reference: run everything in order, one after another


**If you did NOT set up the optional GPU section**, once you've done
Steps 0-2 (opened a terminal, created and activated the environment,
installed packages) you can copy-paste this whole block at once if you
prefer — it runs every training step back to back, all in the same
Windows terminal:


```
python data\generate_data.py
python notebooks\eda.py
python train.py
python models\train_lightgbm.py
python models\train_catboost.py
python models\train_prophet.py
python models\train_lstm.py
python models\train_ensemble.py
python results\model_comparison.py
```


**If you DID set up the optional GPU section**, run everything except
Step 9 in your normal Windows terminal, then switch to the Ubuntu window
for Step 9 only, then switch back to Windows for the rest:
```
:: In your normal Windows terminal:
python data\generate_data.py
python notebooks\eda.py
python train.py
python models\train_lightgbm.py
python models\train_catboost.py
python models\train_prophet.py
```
```
# Then in the Ubuntu window (with venv_gpu activated - see GPU Step E):
python3 models/train_lstm.py
```
```
:: Then back in your normal Windows terminal:
python models\train_ensemble.py
python results\model_comparison.py
```


---


## Troubleshooting


**"python is not recognized as an internal or external command"**
Python isn't installed, or wasn't added to PATH. Reinstall Python from
https://www.python.org/downloads/ and make sure to tick "Add Python to
PATH" during setup.


**Install error mentioning "Long Path", "WinError 3", or a very long
file path that couldn't be found (this happens with TensorFlow)**
Windows has a default limit on how long a file path can be, and one of
the packages (TensorFlow) needs longer paths than that. To fix it (you'll
need to be able to run something "as Administrator" — ask an IT admin if
you don't have that permission on this computer):
1. Press the Windows key, type `gpedit.msc`, and open it (Local Group
   Policy Editor). If this program doesn't exist on your Windows edition,
   use the registry method below instead.
2. Go to: Computer Configuration → Administrative Templates → System →
   Filesystem.
3. Double-click "Enable Win32 long paths", select "Enabled", click OK.
4. Restart the computer.
5. Re-run the `pip install -r requirements.txt` command from Step 2.


If `gpedit.msc` isn't available, ask your IT department to set the
following Windows Registry value to `1` (this does the same thing):
`HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled`


**Install error mentioning a missing compiler / "Microsoft Visual C++
14.0 or greater is required"**
One of the packages needs a small set of build tools. Download and
install "Build Tools for Visual Studio" from
https://visualstudio.microsoft.com/visual-cpp-build-tools/ (choose the
"Desktop development with C++" option during its install), then re-run
the `pip install -r requirements.txt` command.


**Prophet step (Step 8) seems stuck for several minutes the first time**
This is usually normal — the first run needs to download and set up a
small statistics tool in the background. Give it up to 5 minutes. If it's
still stuck after that and you're on a restricted office/company network,
it may be blocked by a firewall — that's fine, the script will
automatically fall back to a backup method (see Step 8's explanation above)
the next time you run it; you can just let it continue or restart the step.


**`wsl --install` fails, says WSL2 needs a feature enabled, or the Ubuntu
window closes immediately**
Some older Windows setups need virtualization turned on in the computer's
BIOS/UEFI settings first (usually called "Intel VT-x" or "SVM Mode"), and/or
need Windows Update run at least once. Run Windows Update fully (Settings →
Windows Update → Check for updates, install everything, restart), then try
`wsl --install` again. If it still fails, search the exact error message
that appears — this is a well-documented one-time Windows setup issue, not
specific to this project.


**`nvidia-smi` inside Ubuntu says "command not found" even after
installing/updating the Windows NVIDIA driver**
Close and fully reopen the Ubuntu window (or restart the computer) after
installing/updating the driver — WSL2 sometimes needs a fresh session to
pick up a driver update.


**`tensorflow.config.list_physical_devices('GPU')` (GPU Step E) prints an
empty `[]` instead of showing the graphics card**
Double check: (1) you ran `nvidia-smi` successfully first (GPU Step B), (2)
you installed with the `[and-cuda]` part included:
`pip install tensorflow[and-cuda]==2.20.0` (not plain `tensorflow==2.20.0`),
and (3) you're running this check from inside the `venv_gpu` environment
(you should see `(venv_gpu)` at the start of the terminal line). If all of
that checks out and it's still empty, the GPU setup didn't fully work —
it's safe to skip it and just run Step 9 on the normal Windows terminal
instead (slower, but will still work correctly).


**"Address already in use" / "only one usage of each socket address"
when starting the service in Step 12**
Something is already using that address. Either close whatever terminal
window is already running `uvicorn`, or start it on a different port
instead:
```
uvicorn api.main:app --reload --port 8001
```
and then visit `http://127.0.0.1:8001/docs` instead.


**Any step seems frozen**
Some steps (especially Step 5 and Step 9) are genuinely slow and can look
"frozen" for several minutes while actually working — check the terminal
window's title bar or task manager to see if the `python.exe` process is
still using CPU. If it's using CPU, it's still working — just wait longer.





