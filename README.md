# EDMR denoising

This project uses a neural network to remove noise and baseline drift from synthetic EDMR spectra. Its performance is compared with moving-average, Gaussian and Savitzky–Golay smoothing, polynomial baseline correction, and baseline correction followed by Savitzky–Golay smoothing.

## How to run

The script uses Python 3.12.7. Install packages:

```bash
python -m pip install numpy==2.5.1 pandas==3.0.5 matplotlib==3.11.1 scipy==1.18.0 torch==2.13.0
python EDMR.py
```

## Outputs

The console prints the training and validation errors, restored epoch, selected filter settings and final test errors.

## Matching the report

Run the script with its default settings.

- **Table 1.1:** the noise exponents are defined in `NOISE_BETA`.
- **Table 1.2:** the selected filter settings are printed after validation.
- **Table 1.3 and Figure 1.3:** use the epoch log and the training-history plot. The report uses a logarithmic axis and marks the selected epoch.
- **Table 1.4 and Figure 1.5:** use the final test-error table and bar chart.
- **Figure 1.4:** use the three test reconstruction plots.

Figure 1.1 in the report is a separate two-panel data illustration; the script displays the first training example instead. Figure 1.2 is the network diagram, supplied with the report.

The report is in `main.pdf`. Its completed training run is saved in `analysis/full_run_2026-10-07.ipynb`. The original trained weights were not saved, so retraining may give slightly different results.
