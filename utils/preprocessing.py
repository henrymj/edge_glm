# Helpers for preprocessing the data in preparation of analysis
import numpy as np
import pandas as pd
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter
from nilearn.glm.first_level import make_first_level_design_matrix


def extract_edges(arr, ridx, cidx):
    return (arr[:, :, np.newaxis] * arr[:, np.newaxis, :])[:, ridx, cidx]


def set_get_timeseries(args, ridx, cidx):
    if args.use_rois or args.replicate:

        def get_timeseries(arr):
            return arr

    else:

        def get_timeseries(arr):
            return extract_edges(zscore(arr, axis=0, nan_policy="omit", ddof=1), ridx, cidx)

    return get_timeseries


def trim_block_ends(sub_run_df, args):
    '''Clip first and last trials of each block if args.trim_block_ends'''
    if args.trim_block_ends:  # only run if clipping blocks, otherwise return original df
        first_trials = [0] + list(sub_run_df.query("condition==0").index.values + 1)
        last_trials = list(sub_run_df.query("condition==0").index.values - 1) + [sub_run_df.index[-1]]
        sub_run_df = sub_run_df.drop(first_trials + last_trials).reset_index(drop=True)

    return sub_run_df


def build_sub_run_df(sub_run_data, ntr_tossed, args):
    if int(args.dataset) == 1:
        sub_run_onsets = pd.DataFrame(sub_run_data["onsetTimes"])
    elif int(args.dataset) == 2:
        sub_run_onsets = pd.DataFrame(sub_run_data["data"][:, 11] - sub_run_data["scanner_starttime"][0][0])

    sub_run_onsets -= ntr_tossed * args.t_r  # shift by the dropped 8 or 3 trs (8 or 3s)
    sub_run_resps = pd.DataFrame(
        sub_run_data["response"],
        columns=[
            "condition",
            "kpress",
            "scanner_t",
            "perc_stim",
            "RT",
            "assignLoop",
            "acc",
        ],
    )
    # drop last row when combining, doesn't have behavioral data
    sub_run_df = pd.concat([sub_run_resps, sub_run_onsets], axis=1)[:-1]
    sub_run_df = trim_block_ends(sub_run_df, args)

    return sub_run_df


def compute_VTC(sub_run_df, FWHM):
    sub_run_df = sub_run_df.copy()
    # build up VTC
    sub_run_df.loc[sub_run_df.RT == 0, "RT"] = None  # set nonresponses to None for interpolation
    sub_run_df.loc[(sub_run_df.acc != 1), "RT"] = None  # set probe resps to None

    sub_run_df["zRT"] = zscore(sub_run_df["RT"].values, nan_policy="omit")
    sub_run_df["abs_zRT"] = sub_run_df["zRT"].apply(np.abs)
    sub_run_df["abs_zRT_interpolated"] = sub_run_df["abs_zRT"].interpolate(method="linear", limit_direction="both")
    # https://en.wikipedia.org/wiki/Full_width_at_half_maximum
    # FWHM = 2*sqrt(2*ln(2)) * sigma
    # -> sigma = FWHM / (2*sqrt(2*ln(2)))
    sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
    # TODO - modify filter to adjust weights at edges
    sub_run_df["vtc"] = gaussian_filter(sub_run_df["abs_zRT_interpolated"].values, sigma=sigma)

    return sub_run_df


def compute_VTC_by_block(sub_run_df, FWHM, break_onsets):
    block_end_idx = break_onsets.index.values  # compute all but last

    first_block = sub_run_df.loc[: block_end_idx[0], :].copy()

    remaining_sub_run_df = sub_run_df.loc[~sub_run_df.index.isin(first_block.index)].copy()  # chop off first block

    first_block = first_block.query("condition!=0")

    sub_run_VTC = compute_VTC(first_block, FWHM)  # compute VTC

    for block_end in block_end_idx[1:]:
        curr_block = remaining_sub_run_df.loc[:block_end, :].copy()

        remaining_sub_run_df = remaining_sub_run_df.loc[~remaining_sub_run_df.index.isin(curr_block.index)].copy()  # chop off current block

        curr_block = curr_block.query("condition!=0")

        curr_block = compute_VTC(curr_block, FWHM)

        # append to ongoing DF
        sub_run_VTC = pd.concat([sub_run_VTC, curr_block])

    last_block = remaining_sub_run_df.copy()
    last_block = last_block.query("condition!=0")

    last_block = compute_VTC(last_block, FWHM)

    sub_run_VTC = pd.concat([sub_run_VTC, last_block])

    return sub_run_VTC.query("condition!=0").reset_index(drop=True)


def add_VTC_get_breaks(sub_run_df, args):
    """
    Add VTC to sub run df and return break onsets and offsets
    """

    break_onsets = sub_run_df.query("condition==0").loc[:, 0]
    break_offsets = sub_run_df.loc[break_onsets.index + 1, 0]
    if int(args.dataset) == 1:
        break_trials = sub_run_df.query("condition==0")
        sub_run_df = compute_VTC_by_block(sub_run_df, args.VTC_FWHM, break_onsets)
        sub_run_df = pd.concat([sub_run_df, break_trials]).sort_index()  # add back in the break trials
        break_durations = break_offsets.values - break_onsets.values
    else:
        sub_run_df = compute_VTC(sub_run_df, args.VTC_FWHM)
        break_durations = None

    return sub_run_df, (break_onsets, break_offsets, break_durations)


def exclude_irrelevant_VTC_TRs(sub_run_desmat, curr_timeseries, args, break_onsets, break_offsets):
    if "VTC" in args.model:  # this function only applies for VTC models, will be skipped otherwise
        if int(args.dataset) == 1:
            excludable_times = np.arange(np.floor(break_onsets.values[0]), np.ceil(break_offsets.values[0]))  # exclude first break
            for idx in range(1, len(break_onsets)):
                # Exclude each block break, shifted by 6 seconds
                curr_exclude = np.arange(np.floor(break_onsets.values[idx]), np.ceil(break_offsets.values[idx])) + args.VTC_shift
                excludable_times = np.concatenate((excludable_times, curr_exclude))
            keep_idx = ~sub_run_desmat.index.isin(excludable_times)
            sub_run_desmat = sub_run_desmat[keep_idx].reset_index(drop=True)
            curr_timeseries = curr_timeseries[keep_idx, :]

        # for both datasets, exclude the first 6 TRs
        notnull = sub_run_desmat.vtc.notnull()
        sub_run_desmat = sub_run_desmat[notnull].reset_index()
        curr_timeseries = curr_timeseries[notnull, :]

    return sub_run_desmat, curr_timeseries


def check_dataset2_censored_trs(sub_run_desmat, curr_timeseries, args, return_bad_trs=False):
    skip_session = False
    bad_trs = None
    if int(args.dataset) == 2:  # this function only applies to dataset 2, will be skipped for dataset 1
        bad_trs = np.argwhere(np.apply_along_axis(lambda x: np.isnan(x).all(), axis=1, arr=curr_timeseries)).flatten()

        if len(bad_trs) / curr_timeseries.shape[0] >= args.dataset2_censor_thresh:  # skip subjects with too many censored timepoints
            skip_session = True
            pass

        # OLS model:
        # remove bad TRs from timeseries and design matrix
        if args.glm_noise_model == "ols":
            curr_timeseries = np.delete(curr_timeseries, bad_trs, axis=0)
            sub_run_desmat = sub_run_desmat.drop(bad_trs)

        # AR1 model:
        # interpolate TRs for AR1 whitening
        # add junk regressors for each kept TR
        # remove TRs at bookends which couldn't be interpolated
        elif args.glm_noise_model == "ar1":
            curr_timeseries = pd.DataFrame(curr_timeseries).interpolate(limit_area="inside").values  # interpolate, excluding bookends
            borders_bad_trs = np.argwhere(np.apply_along_axis(lambda x: np.isnan(x).all(), axis=1, arr=curr_timeseries)).flatten()  # still bad

            # clip timeseries and design matrix to exclude those bad border TRs that can't be interpolated
            curr_timeseries = np.delete(curr_timeseries, borders_bad_trs, axis=0)
            sub_run_desmat = sub_run_desmat.drop(borders_bad_trs)

            # add a junk regressor for every non-border bad TR
            bad_trs = [btr for btr in bad_trs if btr not in borders_bad_trs]
            for btr in bad_trs:
                sub_run_desmat[f"censor_{btr}"] = 0
                sub_run_desmat.loc[btr, f"censor_{btr}"] = 1

            # reset desmat index
            sub_run_desmat = sub_run_desmat.reset_index(drop=True)
    if return_bad_trs:
        return curr_timeseries, sub_run_desmat, skip_session, bad_trs
    else:
        return curr_timeseries, sub_run_desmat, skip_session


def convert_df_to_desMat(
    sub_run_df,
    sub_trs,
    model=None,
    VTC_shift=6,
    break_durations=None,
    break_policy="single",
    hrf_model="spm + derivative",
    **kwargs,
):
    assert model in [
        "trialType",
        "VTC_shifted",
        "VTC_convolved",
        "VTC_convolvedDemeaned",
    ]
    sub_run_df = sub_run_df.copy()
    sub_run_df.insert(0, "duration", 1)
    sub_run_df.insert(0, "onset", sub_run_df[0])
    sub_run_df.insert(0, "trial_type", None)

    if model == "VTC_shifted":
        sub_run_df = sub_run_df.query("condition!=0").reset_index(drop=True)  # cut out block breaks

        sub_run_df["trial_type"] = "vtc"

        # Insert rows with TR onsets and then interpolate
        interp_df = sub_run_df.copy()
        TR_onset_df = pd.DataFrame([["vtc"] * len(sub_trs), sub_trs], index=["trial_type", "onset"]).T
        interp_df_w_TR_onsets = pd.concat([interp_df, TR_onset_df]).sort_values(by="onset").set_index("onset")
        interp_df_w_TR_onsets["vtc"] = interp_df_w_TR_onsets["vtc"].interpolate(method="slinear")
        interpolated_vtc = interp_df_w_TR_onsets.loc[sub_trs, "vtc"].values

        interpolated_vtc_shifted = np.full(len(sub_trs), np.nan)
        interpolated_vtc_shifted[VTC_shift:] = interpolated_vtc[: (len(sub_trs) - VTC_shift)]
        out_df = pd.DataFrame(interpolated_vtc_shifted, columns=["vtc"])
        out_df["constant"] = 1

        return out_df

    else:
        # do the work for the Trial Type model as the default, VTC_convolved builds off of it
        censor_breaks = False

        (rare_cond, common_cond) = (1, 2) if len(sub_run_df.query("condition==1")) < len(sub_run_df.query("condition==2")) else (2, 1)
        sub_run_df.insert(0, "modulation", 1)

        sub_run_df.loc[(sub_run_df.condition == rare_cond) & (sub_run_df.acc == 1), "trial_type"] = "probe_success"
        sub_run_df.loc[(sub_run_df.condition == rare_cond) & (sub_run_df.acc == -1), "trial_type"] = "probe_fail"
        sub_run_df.loc[(sub_run_df.condition == common_cond) & (sub_run_df.acc == -1), "trial_type"] = "common_fail"

        if break_durations is not None:
            if break_policy == "single":  # whether to handle block breaks as one
                sub_run_df.loc[(sub_run_df.condition == 0), "trial_type"] = "break"
                sub_run_df.loc[(sub_run_df.condition == 0), "duration"] = break_durations
            elif break_policy == "separate":
                for idx, row in enumerate(sub_run_df.query("condition==0").iterrows()):
                    sub_run_df.loc[row[0], "trial_type"] = f"break_{idx}"
                    sub_run_df.loc[row[0], "duration"] = break_durations[idx]
            elif break_policy == "censor":  # easiest to do this after making the design matrix
                censor_breaks = True

        if "VTC_convolved" in model:
            # add last unmodulated regressor
            sub_run_df.loc[
                (sub_run_df.condition == common_cond) & (sub_run_df.acc == 1),
                "trial_type",
            ] = "common_success"

            vtc_df = sub_run_df.copy()
            vtc_df["vtc"] = vtc_df["abs_zRT_interpolated"]  # converting to pre-smoothed, let HRF smooth
            if model == "VTC_convolvedDemeaned":
                vtc_df["vtc"] = vtc_df["vtc"] - vtc_df["vtc"].mean()

            vtc_df["trial_type"] = "vtc"
            vtc_df["modulation"] = vtc_df["vtc"]

            sub_run_df = pd.concat([sub_run_df, vtc_df])
        else:
            assert model == "trialType"

    # make des_mat
    sub_run_des_mat = make_first_level_design_matrix(
        sub_trs,
        sub_run_df.loc[
            sub_run_df.trial_type.notnull(),
            ["trial_type", "onset", "duration", "modulation"],
        ].reset_index(drop=True),
        hrf_model=hrf_model,
        **kwargs,
    )
    sub_run_des_mat = sub_run_des_mat.loc[:, ~sub_run_des_mat.columns.str.contains("drift")]  # Drop drift, already removed from data!

    if censor_breaks:
        for idx, row in enumerate(sub_run_df.query("condition==0").iterrows()):
            censor_trs = np.where((sub_trs >= row[1]["onset"]) & (sub_trs <= row[1]["onset"] + break_durations[idx]))
            for ct in censor_trs[0]:
                sub_run_des_mat[f"break_censor_{ct}"] = 0
                sub_run_des_mat.loc[ct, f"break_censor_{ct}"] = 1

    return sub_run_des_mat
