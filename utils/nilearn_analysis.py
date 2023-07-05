# Helpers for running actual analysis
# Mostly modified versions of Nilearn functions
import sys
import time
import numpy as np
from warnings import warn
from nilearn.glm.first_level import make_first_level_design_matrix, run_glm
from nilearn.glm.contrasts import (
    _compute_fixed_effect_contrast,
    expression_to_contrast_vector,
)
from nilearn.glm.contrasts import compute_contrast as compute_glm_contrast
from nilearn.glm._utils import _check_run_tables, _check_events_file_uses_tab_separators
from nilearn.glm.regression import RegressionResults, SimpleRegressionResults
from nilearn.glm.second_level.second_level import (
    _check_first_level_contrast,
    _get_con_val,
    _check_output_type,
    _check_effect_maps,
)


def get_contrast_info(args):
    model_contrasts = {
        "trialType": ["probe_fail", "probe_success", "probe_success-probe_fail", "probe_fail+probe_success", "common_fail"],
        "VTC_shifted": [
            "vtc",
        ],
        "VTC_convolved": [
            "vtc",
        ],
        "VTC_convolvedDemeaned": [
            "vtc",
        ],
    }
    contrast_DM_cols = {
        "trialType": [
            "probe_fail",
            "probe_success",
        ],
        "VTC_shifted": [
            "vtc",
        ],
        "VTC_convolved": [
            "vtc",
        ],
        "VTC_convolvedDemeaned": [
            "vtc",
        ],
    }

    return model_contrasts[args.model], contrast_DM_cols[args.model]


# MODIFIED NILEARN FUNCTIONS #
def get_flm_attribute(flm, attribute, result_as_time_series):
    """Transform RegressionResults instances within a dictionary
    (whose keys represent the autoregressive coefficient under the 'ar1'
    noise model or only 0.0 under 'ols' noise_model and values are the
    RegressionResults instances) into input nifti space.
    Parameters
    ----------
    attribute : str
        an attribute of a RegressionResults instance.
        possible values include: residuals, normalized_residuals,
        predicted, SSE, r_square, MSE.
    result_as_time_series : bool
        whether the RegressionResult attribute has a value
        per timepoint of the input nifti image.
    Returns
    -------
    output : list
        A list of Nifti1Image(s).
    """
    # check if valid attribute is being accessed.
    all_attributes = dict(vars(RegressionResults)).keys()
    possible_attributes = [prop for prop in all_attributes if "__" not in prop]
    if attribute not in possible_attributes:
        msg = "attribute must be one of: " "{attr}".format(attr=possible_attributes)
        raise ValueError(msg)

    if flm.minimize_memory:
        raise ValueError(
            "To access voxelwise attributes like "
            "R-squared, residuals, and predictions, "
            "the `FirstLevelModel`-object needs to store "
            "there attributes. "
            "To do so, set `minimize_memory` to `False` "
            "when initializing the `FirstLevelModel`-object."
        )

    if flm.labels_ is None or flm.results_ is None:
        raise ValueError("The model has not been fit yet")

    output = []

    for design_matrix, labels, results in zip(flm.design_matrices_, flm.labels_, flm.results_):
        if result_as_time_series:
            voxelwise_attribute = np.zeros((design_matrix.shape[0], len(labels)))
        else:
            voxelwise_attribute = np.zeros((1, len(labels)))

        for label_ in results:
            label_mask = labels == label_
            voxelwise_attribute[:, label_mask] = getattr(results[label_], attribute)

        output.append(voxelwise_attribute)  # MODIFIED to avoid inverse_transform

    return output


def fit_edge_flm(flm, run_Ys, events=None, confounds=None, design_matrices=None):
    """
    A version of nilearn's FLM fit function modified to work on edge timeseries.
    Skips Nifti masking procedures and some signal transformations should occur prior to being passed in (e.g. mean scaling)
    See their code for more details:
    https://github.com/nilearn/nilearn/blob/master/nilearn/glm/first_level/first_level.py#L332
    """

    # Raise a warning if both design_matrices and confounds are provided
    if design_matrices is not None and (confounds is not None or events is not None):
        warn("If design matrices are supplied, confounds and events will be ignored.")
    # Local import to prevent circular imports

    # Check arguments
    # Check imgs type
    if events is not None:
        _check_events_file_uses_tab_separators(events_files=events)
    if not isinstance(run_Ys, (list, tuple)):
        run_Ys = [run_Ys]
    if design_matrices is None:
        if events is None:
            raise ValueError("events or design matrices must be provided")
        if flm.t_r is None:
            raise ValueError("t_r not given to FirstLevelModel object" " to compute design from events")
    else:
        design_matrices = _check_run_tables(run_Ys, design_matrices, "design_matrices")
    # Check that number of events and confound files match number of runs
    # Also check that events and confound files can be loaded as DataFrame
    if events is not None:
        events = _check_run_tables(run_Ys, events, "events")
    if confounds is not None:
        confounds = _check_run_tables(run_Ys, confounds, "confounds")

    # For each run fit the model and keep only the regression results.
    flm.labels_, flm.results_, flm.design_matrices_ = [], [], []
    n_runs = len(run_Ys)
    t0 = time.time()
    for run_idx, Y in enumerate(run_Ys):
        # Report progress
        if flm.verbose > 0:
            percent = float(run_idx) / n_runs
            percent = round(percent * 100, 2)
            dt = time.time() - t0
            # We use a max to avoid a division by zero
            if run_idx == 0:
                remaining = "go take a coffee, a big one"
            else:
                remaining = (100.0 - percent) / max(0.01, percent) * dt
                remaining = "%i seconds remaining" % remaining

            sys.stderr.write("Computing run %d out of %d runs (%s)\n" % (run_idx + 1, n_runs, remaining))

        # Build the experimental design for the glm
        if design_matrices is None:
            n_scans = Y.shape[1]
            if confounds is not None:
                confounds_matrix = confounds[run_idx].values
                if confounds_matrix.shape[0] != n_scans:
                    raise ValueError("Rows in confounds does not match" "n_scans in Y at index %d" % (run_idx,))
                confounds_names = confounds[run_idx].columns.tolist()
            else:
                confounds_matrix = None
                confounds_names = None
            start_time = flm.slice_time_ref * flm.t_r
            end_time = (n_scans - 1 + flm.slice_time_ref) * flm.t_r
            frame_times = np.linspace(start_time, end_time, n_scans)
            design = make_first_level_design_matrix(
                frame_times,
                events[run_idx],
                flm.hrf_model,
                flm.drift_model,
                flm.high_pass,
                flm.drift_order,
                flm.fir_delays,
                confounds_matrix,
                confounds_names,
                flm.min_onset,
            )
        else:
            design = design_matrices[run_idx]
        flm.design_matrices_.append(design)

        # Mask and prepare data for GLM
        if flm.verbose > 1:
            t_masking = time.time()
            sys.stderr.write("Starting masker computation \r")

        if flm.verbose > 1:
            t_masking = time.time() - t_masking
            sys.stderr.write("Masker took %d seconds       \n" % t_masking)

        if flm.memory:
            mem_glm = flm.memory.cache(run_glm, ignore=["n_jobs"])
        else:
            mem_glm = run_glm

        # compute GLM
        if flm.verbose > 1:
            t_glm = time.time()
            sys.stderr.write("Performing GLM computation\r")
        labels, results = mem_glm(Y, design.values, noise_model=flm.noise_model, bins=100, n_jobs=flm.n_jobs)
        if flm.verbose > 1:
            t_glm = time.time() - t_glm
            sys.stderr.write("GLM took %d seconds         \n" % t_glm)

        flm.labels_.append(labels)
        # We save memory if inspecting model details is not necessary
        if flm.minimize_memory:
            for key in results:
                results[key] = SimpleRegressionResults(results[key])
        flm.results_.append(results)
        del Y

    # Report progress
    if flm.verbose > 0:
        sys.stderr.write("\nComputation of %d runs done in %i seconds\n\n" % (n_runs, time.time() - t0))

    return flm


def compute_edge_contrast(flm, contrast_def, stat_type=None, output_type="z_score"):
    """
    Modified from https://github.com/nilearn/nilearn/blob/master/nilearn/glm/first_level/first_level.py#L527
        in order to skip inverse masking operation. Follow link for details.
    """
    if flm.labels_ is None or flm.results_ is None:
        raise ValueError("The model has not been fit yet")

    if isinstance(contrast_def, (np.ndarray, str)):
        con_vals = [contrast_def]
    elif isinstance(contrast_def, (list, tuple)):
        con_vals = contrast_def
    else:
        raise ValueError("contrast_def must be an array or str or list of" " (array or str)")

    n_runs = len(flm.labels_)
    n_contrasts = len(con_vals)
    if n_contrasts == 1 and n_runs > 1:
        warn("One contrast given, assuming it for all %d runs" % n_runs)
        con_vals = con_vals * n_runs
    elif n_contrasts != n_runs:
        raise ValueError("%n contrasts given, while there are %n runs" % (n_contrasts, n_runs))

    # Translate formulas to vectors
    for cidx, (con, design_mat) in enumerate(zip(con_vals, flm.design_matrices_)):
        design_columns = design_mat.columns.tolist()
        if isinstance(con, str):
            con_vals[cidx] = expression_to_contrast_vector(con, design_columns)

    valid_types = ["z_score", "stat", "p_value", "effect_size", "effect_variance"]
    valid_types.append("all")  # ensuring 'all' is the final entry.
    if output_type not in valid_types:
        raise ValueError("output_type must be one of {}".format(valid_types))
    contrast = _compute_fixed_effect_contrast(flm.labels_, flm.results_, con_vals, stat_type)
    output_types = valid_types[:-1] if output_type == "all" else [output_type]
    outputs = {}
    for output_type_ in output_types:
        # Prepare the returned images
        output = {
            "estimate": getattr(contrast, output_type_)(),
            "descrip": "%s of contrast %s" % (output_type_, str(con_vals)),
        }
        outputs[output_type_] = output

    return outputs if output_type == "all" else output


def compute_2ndlevel_edge_contrast(
    slm,
    second_level_contrast=None,
    first_level_contrast=None,
    second_level_stat_type=None,
    output_type="z_score",
):
    """
    Modified from https://github.com/nilearn/nilearn/blob/master/nilearn/glm/second_level/second_level.py
    Folow URL for details
    """
    if slm.second_level_input_ is None:
        raise ValueError("The model has not been fit yet")

    # check first_level_contrast
    _check_first_level_contrast(slm.second_level_input_, first_level_contrast)

    # check contrast and obtain con_val
    con_val = _get_con_val(second_level_contrast, slm.design_matrix_)

    # check output type
    # 'all' is assumed to be the final entry;
    # if adding more, place before 'all'
    valid_types = [
        "z_score",
        "stat",
        "p_value",
        "effect_size",
        "effect_variance",
        "all",
    ]
    _check_output_type(output_type, valid_types)

    # Check design matrix X and effect maps Y agree on number of rows
    _check_effect_maps(slm.second_level_input_, slm.design_matrix_)

    # Fit an Ordinary Least Squares regression for parametric statistics
    Y = slm.second_level_input_
    if slm.memory:
        mem_glm = slm.memory.cache(run_glm, ignore=["n_jobs"])
    else:
        mem_glm = run_glm
    labels, results = mem_glm(Y, slm.design_matrix_.values, n_jobs=slm.n_jobs, noise_model="ols")

    # We save memory if inspecting model details is not necessary
    if slm.minimize_memory:
        for key in results:
            results[key] = SimpleRegressionResults(results[key])
    slm.labels_ = labels
    slm.results_ = results

    # We compute contrast object
    if slm.memory:
        mem_contrast = slm.memory.cache(compute_glm_contrast)
    else:
        mem_contrast = compute_glm_contrast
    contrast = mem_contrast(slm.labels_, slm.results_, con_val, second_level_stat_type)

    output_types = valid_types[:-1] if output_type == "all" else [output_type]

    outputs = {}
    for output_type_ in output_types:
        contrast_name = str(con_val)
        # Prepare the returned images
        output = {
            "estimate": getattr(contrast, output_type_)(),
            "descrip": "%s of contrast %s" % (output_type_, contrast_name),
        }
        outputs[output_type_] = output

    return outputs if output_type == "all" else output
