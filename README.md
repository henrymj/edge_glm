# **Code to produce the analyses of Jones, Yoo, Chun & Rosenberg.**

## Housekeeping
- `envs/` holds the following environmental requirement files:  
    1. `environemt.yml` lists the minimal set of packages required to build the environment.  
    2. `manifest.yml` was generated via the command `conda env export`, and is the most comprehensive, listing all packages in the environment on my machine and their specific versions. However, it may contain packages specific for running the code on Apple Silicon.  

- The Shen atlas (`shen_2mm_268_parcellation.nii.gz`), which is used in the ROI activity visualizations, was downloaded from [here](https://neurovault.org/images/395091/).

- `Shen_network_labels.mat` maps the 268 regions onto 8 networks, and is used in compressing the results from the edge level to the network level.

- `saCPM.mat`, which was used for the additional secondlevel overlap analysis, was downloaded from [here](https://github.com/monicadrosenberg/Rosenberg_PNAS2020).

 - `dataset-2/sublist.csv` lists the 65 participants who completed 2 runs of GradCPT in dataset 2.

# Analysis Reproduction
- **IMPORTANT NOTE:** Dataset 1 cannot be shared publicly. Thus, you can only rerun analyses on dataset 2. For inquiries about Dataset 1, contact the authors of the original manuscript https://doi.org/10.1038/nn.4179.

- Some notebooks can be run from without any modifications. For others, arguments needed to be provided in one of the earliest cells defining the arguments, typically the first line of the 3rd cell. The code would be changed from `get_args()` (the standard analysis) to `get_args(["-arg1", "new_arg1_val", "-arg2", "new_arg2_val", ...etc])`. Because we are only sharing dataset 2, the firstlevel and secondlevel notebooks have been set to run on dataset 2 (`get_args(["-dataset", "2"])`).

## Figure 1

To produce the material for Figure 1, replicating the ROI activity analyses, run the following, adding a `'-replicate'` flag to the arguments of the firstlevels:
1. `firstlevels_glm.ipynb`  
\[set the first line of cell 3 to `get_args(["-replicate" "-dataset", "2"])`] 
2. `firstlevels_VTCcorrs.ipynb`  
\[set the first line of cell 3 to `get_args(["-replicate" "-dataset", "2"])`] 
5. `secondlevels_replicate-Dataset2.ipynb`

## Figure 2
To produce the material for Figure 2, examining edges results for traditional analyses, run the following:
1. `firstlevels_glm.ipynb`
2. `firstlevels_VTCcorrs.ipynb` 
5. `secondlevels.ipynb`
6. `thirdlevels.ipynb`
7. `visualize_results.ipynb`

## Figure 3
To produce the material for Figure 3, comparing edges correlated with the VTC and attention networks predicting individual differences, run:
1. `thirdlevels.ipynb` (The critical section is "*Compare 2ndlevels with CPM edges*")  
2. `visualize_results.ipynb` (the critical section is "*VTC corr \* CPM overlap, dataset 2 NBS*")  

This assumes you've run the notebooks described above for generating Figure 2.

## Figure 4
To produce the material for Figure 4, the estimated hemodynamic response clusters, see the following (but note they can't be run without dataset 1):
1. `FIR_ROIs.ipynb`
2. `FIR_edges.ipynb`

## Unpredicted edges analysis
To identify edges which weren't predicted by ROIs, see `visualize_results.ipynb`. The critical section is  "*Looking for Edges that aren't predicted by ROIs*"

## Additional analyses & supplements

### Running parametric VTC regression
To run the alternative VTC analysis, treating the VTC as a parametric regressor, run:
1. `firstlevels_glm.ipynb`  
[set the first line of cell 3 to: `get_args(["-dataset", "2", "-model", "VTC_convolved"])`]   
Then, run secondlevel notebook onwards as if producing Figure 2.

### Figure S1 - Surface projecting ROIs for visualization 
See `visualize_projected_ROIs.ipynb`.

### Figure S4 - CO vs CE pre-trial differences in Dataset 2
To compare CO against CE in the timepoints leading up to the rare category trials, run the following:
1. `firstlevels_pretrial.ipynb`  
[set the first line of cell 3 to: `get_args(["-dataset", "2"])`]  
2. `firstlevels_pretrial.ipynb`  
[set the first line of cell 3 to: `get_args(["-replicate", "-dataset", "2"])`]
3. `secondlevels.ipynb`