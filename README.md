# EEG Pipeline for TRAM protocol

## EEG Pipeline for TRAM protocol (MATLAT / Python)
A Randomised Controlled Trial of the Theta Burst Stimulation Melbourne Protocol for Alzheimer's Disease (The TRAM Protocol).

### EEG preprocessing (MATLAB)
EEG preprocessing is done using RELAX pipeline (https://github.com/NeilwBailey/RELAX/releases) in MATLAB.

### EEG processing (MATLAB)
Resting EEG sliced into epochs, removed EEG channels are interpolated, bad epochs rejected

### Theta-gamma coupling (Python)
Phase-amplitude coupling (PAC) is done using tensorpac package (https://etiennecmb.github.io/tensorpac)

1. Average re-referenced, downsampled, and baseline corrected (to itself).
2. Theta-gamma coupling calculated
3. Normalisation / statistics
4. Peak frequency determined
5. PSD, Comodulogram and Preferred Phase plotted