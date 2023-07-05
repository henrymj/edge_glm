# Helper Functions for reading, writing, and setting up of data/results
import argparse
import json
import numpy as np
import pandas as pd
from glob import glob
from os import path, makedirs
from collections import defaultdict
from scipy.io import loadmat


def get_args(args=None):
    """
    Parse arguments, command line style
    """
    parser = argparse.ArgumentParser(description="GLM Script")
    parser.add_argument("-model", default="trialType")
    parser.add_argument("-glm_noise_model", default="ar1")
    parser.add_argument("-use_rois", action="store_true")
    parser.add_argument("-replicate", action="store_true")
    parser.add_argument("-VTC_FWHM", default=9)
    parser.add_argument("-VTC_shift", default=6)
    parser.add_argument("-trim_block_ends", action="store_true")
    parser.add_argument("-t_r", default=1)
    parser.add_argument('-nTrials_preceding', default=6)
    parser.add_argument("-signal_scaling", action="store_true")
    parser.add_argument("-atlas_loc", default="shen_2mm_268_parcellation.nii.gz")
    parser.add_argument("-nROI", default=268)
    parser.add_argument("-dataset", default=1)
    parser.add_argument("-dataset2_censor_thresh", default=0.1)
    parser.add_argument("-results_base", default="results")

    # second level specific
    parser.add_argument('-n_perms', default=10000)  # number of permutations to run, ALSO USED FOR THIRDLEVELS
    parser.add_argument('-perm_p_thresh', default=.05)  # after permutation, what threshold to be above to include
    parser.add_argument('-NBS_p_thresh', default=.01)  # during permutation, what threshold to use before building graphs
    parser.add_argument('-include_s7', action='store_true')
    parser.add_argument('-drop_nan_nodes', action='store_true')
    parser.add_argument('-overwrite', action='store_true')

    args = [] if args is None else args

    return parser.parse_args(args)


def get_firstlevel_dir(args):
    desc = [
        ("trimBlockEnds-True" if args.trim_block_ends else ""),
        ("noiseModel-OLS" if args.glm_noise_model == "ols" else ""),
        ("Replicate-True" if args.replicate else ""),
    ]
    desc = "_".join([s for s in desc if s != ""])
    desc = desc if desc != "" else "standard"

    results_dir = path.join(args.results_base, "1stlevels", f"dataset-{args.dataset}", f"desc-{desc}")
    makedirs(results_dir, exist_ok=True)
    return results_dir


def get_secondlevel_dir(args):
    desc = [
        ("trimBlockEnds-True" if args.trim_block_ends else ""),
        ("noiseModel-OLS" if args.glm_noise_model == "ols" else ""),
        ("Replicate-True" if args.replicate else ""),
        ('include-s7' if args.include_s7 else ''),
        ('drop_nan_nodes-True' if args.drop_nan_nodes else '')
    ]
    desc = '_'.join([s for s in desc if s != ''])
    desc = desc if desc != '' else 'standard'

    secondlevel_dir = path.join(args.results_base, '2ndlevels', f'dataset-{args.dataset}', f'desc-{desc}')
    makedirs(secondlevel_dir, exist_ok=True)
    return secondlevel_dir


def save_args(args, results_dir):
    model_str = f'model-{args.model}_datatype-{"roi" if args.use_rois else "edge"}_args.json'
    with open(path.join(results_dir, model_str), "w") as f:
        json.dump(vars(args), f)


def load_data(args):
    if int(args.dataset) == 1:
        ntr_tossed = 8

        sublist = ["Sub%s" % i for i in range(1, 26)]
        sub_file_map = {}
        for sidx, sub in enumerate(sublist):
            sub_file_map[sidx] = sorted(glob(path.join("dataset-1/behav_data", "%s_*" % sub)))

        roi_data = loadmat("dataset-1/fmri_data.mat")

        # get timeseries, split into a list for each participant
        timeseries_data_unsplit = {idx: arr[0] for idx, arr in enumerate(roi_data["gradCPT_ROImean_FullTask"])}
        timeseries_data = defaultdict(list)
        for sidx, sub in enumerate(sublist):
            sub_files = sub_file_map[sidx]
            sub_trs = np.arange(int(timeseries_data_unsplit[sidx].shape[0] / len(sub_files))) * args.t_r  # assumes runs are equal TRs
            for runidx in range(len(sub_files)):
                curr_timeseries = timeseries_data_unsplit[sidx][runidx * len(sub_trs) : (runidx + 1) * len(sub_trs)]
                timeseries_data[sidx].append(curr_timeseries)

    elif int(args.dataset) == 2:
        ntr_tossed = 3

        fmri_files = sorted(glob("dataset-2/fmri_data/*.mat"))
        timeseries_data = {idx: loadmat(f)["ts"][0].tolist() for idx, f in enumerate(fmri_files)}

        sublist = ["sub%s" % i for i in pd.read_csv("dataset-2/sublist.csv").T.values[0]]

        sub_file_map = {}
        for sidx, sub in enumerate(sublist):
            sub_file_map[sidx] = sorted(glob(path.join("dataset-2/behav_data", "gradCPTdata_%s_*" % sub)))

    return timeseries_data, sub_file_map, sublist, ntr_tossed
