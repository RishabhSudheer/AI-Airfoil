# AI Airfoil

A small neural network that predicts an airfoil's lift coefficient (CL) and drag coefficient (CD) from its shape, the angle of attack and the Reynolds number. It answers in about a millisecond instead of running the XFOIL solver, and it was trained on an ordinary laptop CPU.

Shapes come from the UIUC Airfoil Coordinates Database. The lift and drag values the network learned from were calculated by XFOIL 6.99, run from Python through AeroSandbox.

## The model

| | |
|---|---|
| Inputs (20) | 18 CST (Kulfan) shape numbers, angle of attack (degrees), log10(Reynolds number) |
| Outputs (2) | CL and log10(CD) |
| Network | Fully connected (MLP), 20 -> 128 -> 128 -> 128 -> 2, ReLU, 35,970 parameters |
| Training | Adam optimizer, learning rate 0.001, batch size 256, early stopping (patience 20) |
| Data | 2,141 airfoils, 171,911 XFOIL results; 1,499 training / 321 validation / 321 test airfoils, split by airfoil |
| Training time | 4.8 minutes (122 epochs) on a 2-core CPU |

**How it works:** each airfoil outline is turned into 18 CST numbers (8 for the upper surface, 8 for the lower surface, 1 for the nose and 1 for the trailing-edge thickness). The angle of attack and log10 of the Reynolds number are added, giving 20 inputs. These are scaled and passed through three layers of 128 neurons, which output CL and log10(CD).

## Results on the 321 test airfoils (never seen in training)

| Group | Points | CL mean abs. error | CL R^2 | CD error % |
|---|---|---|---|---|
| All test points | 26,170 | 0.021 | 0.994 | 4.0 |
| Attached flow (alpha <= 8) | 18,659 | 0.019 | 0.993 | 3.5 |
| Near stall (alpha > 8) | 7,511 | 0.027 | 0.956 | 5.1 |

CL error is the average size of the lift error. R^2 is the share of the lift variation the model explains (1.0 = perfect). CD error is the average drag error in percent.

## Project files

```
AI-Airfoil/
├── README.md
├── config.py                  settings: folders, Reynolds numbers, angle range, XFOIL and training settings
├── common.py                  shared code: airfoil files, CST conversion, the network, loading the model
├── 01_prepare_airfoils.py     downloads the UIUC airfoils and converts each shape to 18 CST numbers
├── 02_generate_xfoil_data.py  runs XFOIL at 5 Reynolds numbers and angles -4 to 14 degrees
├── 03_train.py                splits the data by airfoil, scales it and trains the network
├── 04_evaluate.py             scores the network on the test airfoils and draws the charts
├── 05_predict.py              predicts CL, CD and L/D from the command line
├── 06_export_web_model.py     saves the trained network for the website
├── 07_custom_airfoil.py       changes an airfoil's thickness/camber and compares predictions
├── models/
│   ├── surrogate.pt           the trained network weights (PyTorch)
│   ├── scalers.json           input/output scaling numbers and the trained ranges
│   └── web_model.json         the trained network exported for the website
├── web/
│   ├── index.html             calculator page (finished website)
│   ├── charts.html            charts page (finished website)
│   ├── build_ui.py            builds index.html and charts.html from the templates
│   ├── ui_template.html       calculator page template
│   ├── charts_template.html   charts page template
│   └── shape_tools.js         shared website code: shape maths, NACA shapes, thickness/camber controls
└── results/
    ├── data_overview.png      spread of CL, CD, angles and Reynolds numbers, with example curves
    ├── shape_check.png        original outlines vs outlines rebuilt from 18 CST numbers
    ├── loss_curve.png         training and validation error per epoch
    ├── parity_CL.png          predicted vs XFOIL lift on the test airfoils
    ├── parity_CD.png          predicted vs XFOIL drag on the test airfoils
    ├── custom_airfoil.png     e387 before and after changing thickness and camber
    ├── polars/                network vs XFOIL curves for 4 test airfoils
    ├── test_metrics.json      final accuracy scores
    ├── test_predictions.csv   every test point: XFOIL value, prediction and error
    ├── training_history.csv   loss for each of the 122 epochs of the final run
    └── training_runs.csv      log of all training runs and their settings
```

The project comes as two zips. **AI-Airfoil-code.zip** holds the Python and website code. **AI-Airfoil-model-web.zip** holds the trained model, the finished website and the results. Both unzip into a folder called `AI-Airfoil`, so unzipping them in the same place gives the layout above.

The training data (`data/`) is not included. Running steps 1 and 2 below creates it again.

## Website

Open `web/index.html` in any web browser. The trained model is built into the page, so it works offline with nothing to install. Keep `index.html` and `charts.html` in the same folder.

- **Calculator page:** pick an airfoil from the UIUC library or type a NACA 4-digit code (for example 2412 = 2% camber at 40% chord, 12% thick). Change its thickness and camber with the sliders, then set the angle of attack and Reynolds number to get CL, CD and L/D. A changed shape is drawn over the original, and the original's numbers are shown underneath for comparison.
- **Charts page:** CL vs angle, drag polar (CL vs CD) and L/D vs angle at several Reynolds numbers. Click **Pin current** to compare two airfoils.

## Setup

1. Install Python 3.10 or newer.
2. Install the libraries:
   ```
   pip install numpy pandas matplotlib torch aerosandbox
   ```
3. Only needed to make new data: install XFOIL 6.99 so that typing `xfoil` in a terminal starts it. On Windows, put `xfoil.exe` in a folder on your PATH, or set `XFOIL_COMMAND` in `config.py` to its full path. Check it with:
   ```
   python 02_generate_xfoil_data.py --test
   ```

## Use the trained model

These work straight away with the files in `models/`. No data or XFOIL is needed.

```
python 05_predict.py --airfoil naca4412 --alpha 6 --re 500000
python 05_predict.py --airfoil e387 --alpha -2 0 2 4 6 8 --re 200000
python 07_custom_airfoil.py --airfoil naca2412 --thickness 0.15 --alpha 4 --re 5e5
python 07_custom_airfoil.py --airfoil e387 --thickness-scale 1.2 --camber-scale 0.5 --alpha 0 4 8 --re 2e5 --plot
```

- `05_predict.py` prints alpha, CL, CD and L/D. Use `--dat myfile.dat` instead of `--airfoil` for your own coordinate file.
- `07_custom_airfoil.py` gives thickness and camber as fractions of the chord (0.15 = 15%) and prints the old and new predictions side by side. Add `--save new_shape.dat` to save the changed shape.
- Both print a warning if an input is outside the training range.

## Rebuild everything from scratch

Run these from the `AI-Airfoil` folder, in this order:

```
python 01_prepare_airfoils.py
python 02_generate_xfoil_data.py --workers 4
python 03_train.py
python 04_evaluate.py
python 06_export_web_model.py
python web/build_ui.py
```

- Step 1 downloads the UIUC airfoils into `data/raw/` and writes the CST numbers to `data/processed/`.
- Step 2 is the slow one. It saves its progress, so if it stops you can run the same command again and it carries on. Add `--limit 50` to try a small batch first.
- Steps 3 and 4 overwrite `models/` and `results/` with your new model and charts.
- Steps 5 and 6 rebuild the website from the new model.

## Limits

The network learned from XFOIL results. It was not trained on angles below -4 or above 14 degrees, on Reynolds numbers below 100,000 or above 2,000,000, or on compressible flow (Mach 0 only). Outside those conditions, or for shapes very different from the UIUC airfoils, its answers are less reliable. It also inherits XFOIL's weaknesses near stall, and it predicts 2D airfoils only, not 3D wings.

## Credits

UIUC Airfoil Coordinates Database (M. Selig); XFOIL 6.99 (M. Drela, MIT, GPL); AeroSandbox (P. Sharpe); PyTorch, NumPy, pandas, matplotlib.
