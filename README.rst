PF-HMM
----------------

**Brief Description**

This repository contains data, MATLAB code, and Python scripts associated with our theoretical paper, Inferring the causes of noise from binary outcomes:
A normative theory of learning under uncertainty. The repository is structured to clearly separate preprocessing, data analysis, and figure generation.

**File organization**::

  PF-HMM/
    ├── mat_data/
    │   └── data_sealion.mat (preprocessed trial data)
    │   └── hidden_state.mat (the hidden states of the actual time series)
    │   └── reward_stimuli.pkl (the reward location (observation) of the time series)
    │   └── timeseries100.mat (the generated timeseries with 100 trials used in plotting fig2a)
    ├── matlab_code/
    │   └── cbm/ (folder contains cbm fitting code)
    │   └── tools/ (additional tool code)
    │   └── hmm.m (defines the HMM model)
    │   └── hmm_fit.m (run the script to fit HMM)
    │   └── hmm_beta_fit.m (run the script to fit HMM with beta param)
    │   └── model_comparison.m (run the script for model comparison results between HMM and HMM-beta)
    │   └── hmm_recovery.m (run the script for HMM param recovery analysis)
    │   └── pfhmm.m (defines the PF-HMM model)
    │   └── pfhmm_sim.m (run the script to simulate PF and HMM)
    │   └── response_time_analysis.m (run the script for response time analysis)
    │   └── stats_table.m (produce supplementary tables)
    ├── python_code/
    │   └── figures.ipynb (run the script to reproduce the main figures)
    ├── saved_figures/
    │   └── (store mian figures saved from figures.ipynb and hmm_recovery.m)
    └── README.md

**Prerequisites**

- MATLAB R2023b
- Python 3.11.5

**Installation & Setup**

Clone this repository:

  git clone https://github.com/piraylab/PF-HMM.git

  cd PF-HMM

Install Python dependencies:

  pip install -r requirements.txt

**Data Processing Workflow**

1. MATLAB analysis: Use scripts in matlab_code to analyze data stored in mat_data.
2. Python figure generation: Use scripts in python_code to visualize results and save figures to saved_figures.

**Citation**

If you find this work useful, please cite our paper: https://osf.io/preprints/osf/vuc5g_v1
