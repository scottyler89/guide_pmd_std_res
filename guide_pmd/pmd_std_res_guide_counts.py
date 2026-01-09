import os
import shutil
import gc
import argparse
import numpy as np
import pandas as pd
from scipy.stats import t
#from statsmodels.formula.api import GLM
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Gaussian
from scipy.stats import false_discovery_control as fdr
from percent_max_diff.percent_max_diff import pmd

gc.enable()

###################################
def valid_idxs(in_vect):
    return(np.where(~(np.isnan(in_vect) | np.isinf(in_vect)))[0])

def nan_fdr(in_vect):
    good_idxs = valid_idxs(in_vect)
    out_fdr = np.ones(in_vect.shape)
    adj = fdr(in_vect[good_idxs])
    out_fdr[good_idxs]=adj
    return(out_fdr)
######################################

def run_glm_analysis(normalized_matrix, model_matrix, add_intercept=True):
    """
    Takes as input a pandas dataframe for the input analysis ready data matrix, and your model matrix. 
    An important note on the input model matrix is that an "Intercept" column will automatically be added if
    it does not already exist in the input model matrix. So make sure that you either don't have a different 
    constant column included.
        
    :param normalized_matrix: The matrix to be used for running differential abundance
    :param model_matrix: The model matrix that will be used for the GLM (assumes Gaussian family, as would be expected from PMD standardized residuals).
    """
    # Input verification
    if not isinstance(normalized_matrix, pd.DataFrame) or not isinstance(model_matrix, pd.DataFrame):
        raise ValueError("Both inputs must be pandas DataFrames.")
    if normalized_matrix.shape[1] != model_matrix.shape[0]:
        raise ValueError("Matrices must have the same number of columns in data matrix & rows in model matrix (samples).")
    try:
        normalized_matrix.apply(pd.to_numeric)
        model_matrix.apply(pd.to_numeric)
    except Exception as e:
        raise ValueError("Both matrices must contain numeric values.") from e
    # Adding an intercept column
    if "Intercept" not in model_matrix.columns and add_intercept:
        model_matrix = model_matrix.copy()
        model_matrix.insert(0, 'Intercept', 1)
    t_cols = [f"{var_name}_t" for var_name in model_matrix.columns]
    p_cols = [f"{var_name}_p" for var_name in model_matrix.columns]
    # Preparing lists to store results
    all_res_dict = {}
    residuals_dict = {}
    # Iterating through each feature (row) in the normalized matrix
    for feature_name, feature_data in normalized_matrix.iterrows():
        if feature_data.var()>0:
            # Creating a GLM model
            #data_df = pd.concat([feature_data, model_matrix.T], axis=1, keys=['Response', *model_matrix.index])
            #formula = "Response ~ " + ' + '.join(model_matrix.index)
            #model = GLM.from_formula(formula, data_df, family=Gaussian())
            # Merging with model_matrix
            feature_data_df = pd.DataFrame(feature_data.T)
            combined_data = pd.concat([feature_data_df, model_matrix], axis=1)#, keys=['Response', *model_matrix.columns])
            # Building and fitting the model directly
            model = GLM(combined_data.iloc[:,0], combined_data.loc[:,model_matrix.columns.tolist()], family=Gaussian())
            results = model.fit()
            residuals = pd.Series(results.resid_response, index=combined_data.index)
            residuals_dict[feature_name] = residuals
            # Storing results
            temp_dict = {}
            for var_name, val in results.tvalues.items():
                temp_dict[var_name+"_t"]=val
            for var_name, val in results.pvalues.items():
                temp_dict[var_name+"_p"]=val
            all_res_dict[feature_name]=temp_dict
    # Creating and returning the output DataFrame
    res_table = pd.DataFrame(all_res_dict).T
    res_table = res_table.reindex(columns=t_cols + p_cols)
    del all_res_dict
    gc.collect()
    for p_col in p_cols:
        res_table[p_col+"_adj"]=nan_fdr(res_table[p_col].to_numpy())
    res_table = res_table.reindex(index=normalized_matrix.index)
    residuals_df = pd.DataFrame(residuals_dict).T
    residuals_df = residuals_df.reindex(columns=model_matrix.index)
    residuals_df = residuals_df.reindex(index=normalized_matrix.index)
    return res_table, residuals_df


def get_pmd_std_res(input_file, in_annotation_cols, n_boot = 100, seed = 123456, sep = "\t"):
    """
    This function calculates PMD standardized residuals for a count data with annotations. 
    It also returns another subset of the data. The PMD standardized residuals calculation is 
    performed using the pmd function from the percent_maximum_difference package.

    :param input_file: The path to the input file; this will often be gRNA counts, in a similar format to DrugZ's inputs.
    :type input_file: str
    :param in_annotation_cols: The index from which to start the subset for PMD calculation.
    :type in_annotation_cols: int
    :param n_boot: The number of bootstrap samples to be used in the PMD calculation. Defaults to 100.
    :type n_boot: int, optional
    :param seed: The seed for the random number generator. Defaults to 123456.
    :type seed: int, optional
    :param sep: The separator used in the input file. Defaults to "\t".
    :type sep: str, optional

    :return: The PMD standardized residuals and the annotation only subset of the df.
    :rtype: DataFrame, DataFrame
    """
    # Reading data from input TSV file
    np.random.seed(seed)
    in_annotation_cols = int(in_annotation_cols)
    n_boot = int(n_boot)
    if n_boot < 2:
        raise ValueError(
            "n_boot must be >= 2; n_boot=1 produces a degenerate null standard deviation and invalid z-scores"
        )
    if in_annotation_cols < 1:
        raise ValueError("in_annotation_cols must be >= 1")
    guides = pd.read_csv(input_file, sep=sep, index_col=0)
    # Processing the data (this is where you would implement your specific logic)
    print("getting the PMD standardized residuals")
    pmd_res = pmd(guides.iloc[:,(in_annotation_cols-1):], num_boot=n_boot, skip_posthoc = True)
    std_res = pmd_res.z_scores
    std_res = std_res.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return std_res, guides.iloc[:,:(in_annotation_cols-1)]


def convert_t_to_p(t_values, df):
    p_values = []
    for t_value in t_values:
        # calculate the p-value for a two-tailed test
        p = t.sf(abs(t_value), df) * 2
        p_values.append(p)
    return p_values


def get_stouf_t(t_vect):
    """Stouffer's Method of combining p-values from Z-scores. In this case we're using the t-values, but it's essentially the same.
    
    :param t_vect: A vector of t-values that must have been calculated similarly.
    :type t_vect: numpy.ndarray
    
    :return: 
    :rtype: float
    """
    return np.sum(t_vect)/np.sqrt(len(t_vect))


def get_all_comb_t(in_stats, annotation_table, annotation_col=1):
    """
    This function calculates the combined T-values for all variables in a given input statistics DataFrame and annotation table. 
    It uses the 'get_stouf_t' function to calculate the T-values.

    :param in_stats: The input statistics DataFrame.
    :type in_stats: pandas.DataFrame
    :param annotation_table: The annotation table.
    :type annotation_table: pandas.DataFrame
    :param annotation_col: The 0-based column index in the original input file to be used for grouping.
        Since the input file's first column is used as the index, this value should be >= 1.
        Defaults to 1 (the first non-index column).
    :type annotation_col: int, optional

    :return: A DataFrame with the combined T-values for all variables.
    :rtype: pandas.DataFrame
    """
    print("getting combined T-values for all variables")
    if annotation_table.shape[1] < 1:
        raise ValueError("annotation_table must have at least one column to combine on")
    if annotation_col < 1:
        raise ValueError("annotation_col must be >= 1 (0 is the index/ID column and is not present in annotation_table)")
    ann_idx = annotation_col - 1
    if ann_idx >= annotation_table.shape[1]:
        raise ValueError(f"annotation_col out of range for annotation_table: {annotation_col}")
    all_keys = sorted(list(set(annotation_table.iloc[:,ann_idx])))
    t_cols = in_stats.columns[in_stats.columns.str.endswith("_t")].tolist()
    all_out = {}
    for key in all_keys:
        temp_row_names = annotation_table.index[annotation_table.iloc[:,ann_idx]==key].tolist()
        stats_sub_table = in_stats.loc[temp_row_names,:]
        temp_row_entry = {}
        for temp_t in t_cols:
            temp_row_entry[temp_t]=get_stouf_t(stats_sub_table[temp_t].to_numpy())
        all_out[key]=temp_row_entry
    return(pd.DataFrame(all_out).T)


def combine_p(in_stats, annotation_table, annotation_col, dof):
    """
    This function calculates the combined p-values for all variables in a given input statistics DataFrame and annotation table.
    It uses the 'get_all_comb_t' function to get combined T-values and 'convert_t_to_p' to convert T-values to p-values. 
    It also adjusts the p-values using the 'nan_fdr' function.

    :param in_stats: The input statistics DataFrame.
    :type in_stats: pandas.DataFrame
    :param annotation_table: The annotation table.
    :type annotation_table: pandas.DataFrame
    :param annotation_col: The column index in the annotation table to be used for grouping.
    :type annotation_col: int
    :param dof: The degrees of freedom for the T-distribution.
    :type dof: int

    :return: A DataFrame with the combined p-values for all variables.
    :rtype: pandas.DataFrame
    """
    comb_stats = get_all_comb_t(in_stats, annotation_table, annotation_col)
    t_cols = comb_stats.columns.tolist()
    p_cols = []
    for col in t_cols:
        temp_p_col = col[:-1]+"p"
        comb_stats[temp_p_col]=convert_t_to_p(comb_stats[col], dof)
        p_cols.append(temp_p_col)
    for p_col in p_cols:
        comb_stats[p_col+"_adj"]=nan_fdr(comb_stats[p_col].to_numpy())
    return comb_stats


#comb_stats = combine_p(stats_res, annotation_table, annotation_col, dof)

def pmd_std_res_and_stats(input_file, 
                            output_dir, 
                            model_matrix_file = None, 
                            p_combine_idx = None,
                            in_annotation_cols = 2,
                            pre_regress_vars = None,
                            n_boot = 100, 
                            seed = 123456, 
                            file_sep="tsv",
                            std_res_file: str | None = None,
                            gene_level: bool = True,
                            focal_vars: list[str] | None = None,
                            gene_id_col: int = 1,
                            gene_methods: list[str] | None = None,
                            gene_out_dir: str | None = None,
                            gene_figures: bool = True,
                            gene_figures_dir: str | None = None,
                            gene_forest_genes: list[str] | None = None,
                            gene_progress: bool = False,
                            gene_lmm_scope: str = "meta_or_het_fdr",
                            gene_lmm_q_meta: float = 0.1,
                            gene_lmm_q_het: float = 0.1,
                            gene_lmm_audit_n: int = 50,
                            gene_lmm_audit_seed: int = 123456,
                            gene_lmm_max_genes_per_focal_var: int | None = None,
                            gene_lmm_explicit_genes: list[str] | None = None):
    """
    Takes as input a pd.read_csv readable tsv & optionally a similar model matrix file if you want to run stats.
    This is the main function that will
    
    :param input_file: Path to the input file (assume guid or gene id is in the first column & samples in the remaining cols).
    :param output_file: Path to the directory for the desired output.
    :param model_matrix_file: Path to the model matrix (optional, if included, will perform a GLM on the pmd std res of the guides).
    :param in_annotation_cols: Number of leading columns that allow for row-wise annotations. For example, a guide_id in the first columns, and gene_target in the second column (default = 2).
    :param n_boot: Number of bootstrap shuffles to do for null (default = 100).
    :param seed: random seed (default = 123456).
    :param file_sep: tsv for tab separapted, csv for comma (default = tsv).
    """
    if pre_regress_vars is None:
        pre_regress_vars = []
    os.makedirs(output_dir, exist_ok=True)
    if file_sep == "tsv":
        sep = "\t"
    elif file_sep == "csv":
        sep = ","
    else:
        raise ValueError("file_sep must be either 'tsv' or 'csv'")
    output_file = os.path.join(output_dir, "PMD_std_res.tsv")
    # Make sure the model matrix file actually exists if 
    if model_matrix_file is not None:
        if not os.path.isfile(model_matrix_file):
            raise FileNotFoundError(model_matrix_file)
    # Run the PMD standardized residuals & save file
    if std_res_file is not None:
        if not os.path.isfile(std_res_file):
            raise FileNotFoundError(std_res_file)
        guides = pd.read_csv(input_file, sep=sep, index_col=0)
        std_res = pd.read_csv(std_res_file, sep="\t" if std_res_file.endswith(".tsv") else sep, index_col=0)
        std_res = std_res.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        in_annotation_cols = int(in_annotation_cols)
        if in_annotation_cols < 1:
            raise ValueError("in_annotation_cols must be >= 1")
        annotation_table = guides.iloc[:, : (in_annotation_cols - 1)]
        if annotation_table.index.has_duplicates:
            raise ValueError("annotation_table index must not contain duplicates (guide_id)")
        if std_res.index.has_duplicates:
            raise ValueError("std_res index must not contain duplicates (guide_id)")
        if not annotation_table.index.isin(std_res.index).all():
            raise ValueError("std_res is missing one or more guide ids present in the input file")
        # Preserve the original bytes of the precomputed PMD output on disk.
        shutil.copyfile(std_res_file, output_file)
    else:
        std_res, annotation_table = get_pmd_std_res(
            input_file,
            in_annotation_cols=in_annotation_cols,
            n_boot=n_boot,
            seed=seed,
            sep=sep,
        )
        std_res.to_csv(output_file, sep="\t")
    stats_df = None
    resids_df = None
    comb_stats = None
    first_regressed = None
    # If we're running the stats, then get them going:
    if model_matrix_file is not None:
        print("running the statistics, using the specified model matrix")
        output_stats_file = os.path.join(output_dir, "PMD_std_res_stats.tsv")
        resids_file = os.path.join(output_dir, "PMD_std_res_stats_resids.tsv")
        mm = pd.read_csv(model_matrix_file, sep=sep, index_col = 0)
        design_cols = None
        # If we're doing a hard pre-regress:
        if len(pre_regress_vars)>0:
            ## Separate the vars
            keep_cols = [thing for thing in mm.columns if thing not in pre_regress_vars]
            if not all(var in mm.columns for var in pre_regress_vars):
                print("mm columns:\n",mm.columns)
                print("pre_regress_vars:\n",pre_regress_vars)
                raise ValueError("Not all variables in pre_regress_vars are in the model matrix!")
            _, first_regressed = run_glm_analysis(std_res, mm[pre_regress_vars])
            # In this case, we've already modeled and accounted for the intercept, 
            # so we exclude it from the second model in the serial modeling
            stats_df, resids_df = run_glm_analysis(first_regressed, mm[keep_cols], add_intercept=False)
            design_cols = len(keep_cols)
        else:
            keep_cols = mm.columns
            stats_df, resids_df = run_glm_analysis(std_res, mm[keep_cols])
            # Back-compat: dof for Stouffer-combined p-values is computed from the
            # user-provided model-matrix columns (even if an intercept is added).
            design_cols = len(keep_cols)
        stats_df.to_csv(output_stats_file, sep="\t")
        resids_df.to_csv(resids_file, sep="\t")
        if p_combine_idx is not None:
            p_combine_idx = int(p_combine_idx)
            if p_combine_idx < 1:
                raise ValueError("p_combine_idx must be >= 1 (0 is the index/ID column and is not present in annotation_table)")
            if annotation_table.shape[1] < 1:
                raise ValueError("Cannot combine p-values: annotation_table has no columns")
            if (p_combine_idx - 1) >= annotation_table.shape[1]:
                raise ValueError(f"p_combine_idx out of range for annotation_table: {p_combine_idx}")
            dof = std_res.shape[1] - design_cols
            if dof <= 0:
                raise ValueError(f"Non-positive degrees of freedom for t distribution: {dof}")
            comb_stats = combine_p(stats_df, annotation_table, p_combine_idx, dof)
            output_stats_file = os.path.join(output_dir, "PMD_std_res_combined_stats.tsv")
            comb_stats.to_csv(output_stats_file, sep="\t")

        if gene_level:
            if gene_methods is None:
                gene_methods = ["meta", "lmm", "qc"]
            if gene_out_dir is None:
                gene_out_dir = os.path.join(output_dir, "gene_level")
            from . import gene_level as gene_level_mod

            gene_response = std_res if first_regressed is None else first_regressed
            gene_mm = mm[keep_cols]
            gene_add_intercept = not (len(pre_regress_vars) > 0)
            if focal_vars is None or len(focal_vars) == 0:
                focal_vars = [c for c in gene_mm.columns.tolist() if c != "Intercept"]
            if len(focal_vars) == 0:
                print("gene-level: skipping (no focal vars; model matrix appears to contain only an intercept)")
                return std_res, stats_df, resids_df, comb_stats

            gene_meta = None
            gene_lmm_selection = None
            gene_lmm = None
            gene_qc = None
            per_guide_ols = None

            needs_per_guide_ols = ("meta" in gene_methods) or ("lmm" in gene_methods) or ("qc" in gene_methods) or (
                gene_figures and (gene_forest_genes is not None and len(gene_forest_genes) > 0)
            )
            if needs_per_guide_ols:
                gene_mm_aligned = gene_level_mod._align_model_matrix(gene_mm, list(gene_response.columns))
                per_guide_ols = gene_level_mod.fit_per_guide_ols(
                    gene_response,
                    gene_mm_aligned,
                    focal_vars=focal_vars,
                    add_intercept=gene_add_intercept,
                )

            needs_gene_meta = ("meta" in gene_methods) or ("lmm" in gene_methods)
            if needs_gene_meta:
                gene_meta = gene_level_mod.compute_gene_meta(
                    gene_response,
                    annotation_table,
                    gene_mm,
                    focal_vars=focal_vars,
                    gene_id_col=gene_id_col,
                    add_intercept=gene_add_intercept,
                    per_guide_ols=per_guide_ols,
                )
                if "meta" in gene_methods:
                    os.makedirs(gene_out_dir, exist_ok=True)
                    gene_out_path = os.path.join(gene_out_dir, "PMD_std_res_gene_meta.tsv")
                    gene_meta.to_csv(gene_out_path, sep="\t", index=False)
            if "lmm" in gene_methods:
                from . import gene_level_selection as gene_level_selection_mod
                from . import gene_level_lmm as gene_level_lmm_mod

                sel_cfg = gene_level_selection_mod.GeneLmmSelectionConfig(
                    scope=str(gene_lmm_scope),
                    q_meta=float(gene_lmm_q_meta),
                    q_het=float(gene_lmm_q_het),
                    audit_n=int(gene_lmm_audit_n),
                    audit_seed=int(gene_lmm_audit_seed),
                    max_genes_per_focal_var=gene_lmm_max_genes_per_focal_var,
                    explicit_genes=tuple([] if gene_lmm_explicit_genes is None else [str(g) for g in gene_lmm_explicit_genes]),
                )
                sel_cfg.validate()

                feasibility = gene_level_selection_mod.compute_gene_lmm_feasibility(
                    gene_response,
                    annotation_table,
                    gene_mm,
                    focal_vars=[str(v) for v in focal_vars],
                    gene_id_col=gene_id_col,
                    add_intercept=gene_add_intercept,
                )
                if gene_meta is None:
                    raise RuntimeError("internal error: gene_meta is required for lmm selection")  # pragma: no cover
                gene_lmm_selection = gene_level_selection_mod.compute_gene_lmm_selection(
                    gene_meta,
                    feasibility,
                    config=sel_cfg,
                )
                os.makedirs(gene_out_dir, exist_ok=True)
                gene_out_path = os.path.join(gene_out_dir, "PMD_std_res_gene_lmm_selection.tsv")
                gene_lmm_selection.to_csv(gene_out_path, sep="\t", index=False)

                gene_lmm = gene_level_lmm_mod.compute_gene_lmm(
                    gene_response,
                    annotation_table,
                    gene_mm,
                    focal_vars=focal_vars,
                    gene_id_col=gene_id_col,
                    add_intercept=gene_add_intercept,
                    meta_results=gene_meta,
                    selection_table=gene_lmm_selection,
                )
                os.makedirs(gene_out_dir, exist_ok=True)
                gene_out_path = os.path.join(gene_out_dir, "PMD_std_res_gene_lmm.tsv")
                gene_lmm.to_csv(gene_out_path, sep="\t", index=False)
            if "qc" in gene_methods:
                from . import gene_level_qc as gene_level_qc_mod

                gene_qc = gene_level_qc_mod.compute_gene_qc(
                    gene_response,
                    annotation_table,
                    gene_mm,
                    focal_vars=focal_vars,
                    gene_id_col=gene_id_col,
                    add_intercept=gene_add_intercept,
                    residual_matrix=resids_df,
                    per_guide_ols=per_guide_ols,
                )
                os.makedirs(gene_out_dir, exist_ok=True)
                gene_out_path = os.path.join(gene_out_dir, "PMD_std_res_gene_qc.tsv")
                gene_qc.to_csv(gene_out_path, sep="\t", index=False)

            if gene_figures:
                from . import gene_level_figures as gene_level_figures_mod

                if gene_figures_dir is None:
                    gene_figures_dir = os.path.join(output_dir, "figures", "gene_level")
                figures_ok = True
                try:
                    gene_level_figures_mod.write_gene_level_figures(
                        gene_figures_dir,
                        prefix="PMD_std_res",
                        gene_meta=gene_meta,
                        gene_lmm=gene_lmm,
                        gene_qc=gene_qc,
                    )
                except ImportError as exc:
                    figures_ok = False
                    print(f"gene-level figures: skipped ({exc})")
                if gene_forest_genes is not None and len(gene_forest_genes) > 0:
                    if not figures_ok:
                        print("gene-level forest plots: skipped (matplotlib not available)")
                    else:
                        if per_guide_ols is None:
                            raise RuntimeError("internal error: per_guide_ols is required for forest plots")  # pragma: no cover
                        gene_ids = gene_level_mod._get_gene_ids(annotation_table, gene_id_col)
                        per_guide = per_guide_ols.merge(gene_ids, left_on="guide_id", right_index=True, how="left")
                        gene_level_figures_mod.write_gene_forest_plots(
                            per_guide,
                            gene_figures_dir,
                            prefix="PMD_std_res",
                            forest_genes=[str(g) for g in gene_forest_genes],
                            focal_vars=[str(v) for v in focal_vars],
                        )

            if gene_progress:
                msg_parts = []
                if gene_meta is not None:
                    msg_parts.append(f"meta={gene_meta.shape[0]}")
                if gene_lmm_selection is not None:
                    msg_parts.append(f"lmm_selection={gene_lmm_selection.shape[0]}")
                if gene_lmm is not None:
                    msg_parts.append(f"lmm={gene_lmm.shape[0]}")
                if gene_qc is not None:
                    msg_parts.append(f"qc={gene_qc.shape[0]}")
                if msg_parts:
                    print("gene-level outputs:", ", ".join(msg_parts), f"(dir={gene_out_dir})")
                if gene_lmm_selection is not None and (not gene_lmm_selection.empty):
                    n_total = int(gene_lmm_selection.shape[0])
                    n_feasible = int(gene_lmm_selection["feasible"].sum())
                    n_selected = int(gene_lmm_selection["selected"].sum())
                    print(
                        "gene-level lmm selection:",
                        f"selected={n_selected}/{n_total}",
                        f"(feasible={n_feasible}; scope={gene_lmm_scope}; q_meta={gene_lmm_q_meta}; q_het={gene_lmm_q_het}; audit_n={gene_lmm_audit_n}; cap={gene_lmm_max_genes_per_focal_var})",
                    )
                    reason_counts = (
                        gene_lmm_selection.loc[gene_lmm_selection["selection_reason"] != "", "selection_reason"]
                        .value_counts(dropna=False)
                        .to_dict()
                    )
                    if reason_counts:
                        print("gene-level lmm selection reasons:", reason_counts)
                    skip_counts = (
                        gene_lmm_selection.loc[gene_lmm_selection["skip_reason"] != "", "skip_reason"]
                        .value_counts(dropna=False)
                        .to_dict()
                    )
                    if skip_counts:
                        print("gene-level lmm skip reasons:", skip_counts)
                if gene_lmm is not None and (not gene_lmm.empty):
                    method_counts = gene_lmm["method"].value_counts(dropna=False).to_dict()
                    print("gene-level lmm methods:", method_counts)
                    if "lrt_ok" in gene_lmm.columns:
                        lrt_ok_counts = gene_lmm["lrt_ok"].value_counts(dropna=False).to_dict()
                        print("gene-level lmm lrt_ok:", lrt_ok_counts)
                    if "wald_ok" in gene_lmm.columns:
                        wald_ok_counts = gene_lmm["wald_ok"].value_counts(dropna=False).to_dict()
                        print("gene-level lmm wald_ok:", wald_ok_counts)
    elif gene_level:
        print("gene-level: skipping (requires model_matrix_file)")
    return std_res, stats_df, resids_df, comb_stats


def main():
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Process a TSV file and save to an output file.")
    # Adding arguments for input file and output file
    parser.add_argument("-in_file", "-i", type=str, required=True, help= "Path to the input file")
    parser.add_argument("-out_dir", "-o", type=str, required=True, help= "Path to the desired output directory")
    parser.add_argument("-model_matrix_file","-mm", type=str, help= "Path to the input TSV file", default=None)
    parser.add_argument("-pre_regress_vars", "-prv", type=str, nargs="+", help= "Any variables to do full pre-regression rather than joint modeling.", default=None)
    parser.add_argument("-annotation_cols", "-ann_cols", type=int, help= "Number of annotation columns in the input file (including the ID/index column). Default=2", default=2)
    parser.add_argument("-p_combine_idx", type=int, help= "0-based column index in the input file for Stouffer combining. Default=1 (first non-index column). Set to None to disable in the Python API.", default=1)
    parser.add_argument("-n_boot", type = int, help= "the number of bootstrap shuffled nulls to run. (Default=100)", default = 100)
    parser.add_argument("-seed", type = int, help= "set the seed for reproducibility (Default=123456)", default = 123456)
    parser.add_argument("-file_type", type = str, help= "tsv or csv, defulat is tsv", default = "tsv")
    parser.add_argument(
        "--std-res-file",
        dest="std_res_file",
        type=str,
        default=None,
        help="Optional precomputed PMD_std_res.tsv (skips PMD bootstrap computation).",
    )
    parser.add_argument(
        "--gene-level",
        dest="gene_level",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable gene-level outputs (default: enabled when a model matrix is provided).",
    )
    parser.add_argument(
        "--focal-vars",
        dest="focal_vars",
        type=str,
        nargs="+",
        default=None,
        help="Model-matrix column name(s) to compute gene-level effects for (default: all non-Intercept columns).",
    )
    parser.add_argument(
        "--gene-id-col",
        dest="gene_id_col",
        type=int,
        default=1,
        help="0-based column index in the original input file for the gene id (default=1; 0 is the guide id/index).",
    )
    parser.add_argument(
        "--gene-methods",
        dest="gene_methods",
        type=str,
        nargs="+",
        default=["meta", "lmm", "qc"],
        help="Gene-level methods to run (default: meta lmm qc).",
    )
    parser.add_argument(
        "--gene-out-dir",
        dest="gene_out_dir",
        type=str,
        default=None,
        help="Optional output directory for gene-level files (default: <out_dir>/gene_level).",
    )
    parser.add_argument(
        "--gene-figures",
        dest="gene_figures",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate gene-level figures (default: enabled; requires matplotlib).",
    )
    parser.add_argument(
        "--gene-figures-dir",
        dest="gene_figures_dir",
        type=str,
        default=None,
        help="Optional output directory for gene-level figures (default: <out_dir>/figures/gene_level).",
    )
    parser.add_argument(
        "--gene-forest-genes",
        dest="gene_forest_genes",
        type=str,
        nargs="+",
        default=None,
        help="Optional gene id(s) to generate per-guide forest plots for (requires --gene-figures).",
    )
    parser.add_argument(
        "--gene-progress",
        dest="gene_progress",
        action="store_true",
        help="Print a short summary of gene-level execution (counts + fallbacks).",
    )
    parser.add_argument(
        "--gene-lmm-scope",
        dest="gene_lmm_scope",
        type=str,
        choices=["all", "meta_fdr", "meta_or_het_fdr", "explicit", "none"],
        default="meta_or_het_fdr",
        help="Plan A (LMM) scope selection policy (default: meta_or_het_fdr).",
    )
    parser.add_argument(
        "--gene-lmm-q-meta",
        dest="gene_lmm_q_meta",
        type=float,
        default=0.1,
        help="Plan A selection: meta FDR threshold q_meta (default: 0.1).",
    )
    parser.add_argument(
        "--gene-lmm-q-het",
        dest="gene_lmm_q_het",
        type=float,
        default=0.1,
        help="Plan A selection: heterogeneity FDR threshold q_het (default: 0.1).",
    )
    parser.add_argument(
        "--gene-lmm-audit-n",
        dest="gene_lmm_audit_n",
        type=int,
        default=50,
        help="Plan A selection: deterministic audit sample size per focal var (default: 50).",
    )
    parser.add_argument(
        "--gene-lmm-audit-seed",
        dest="gene_lmm_audit_seed",
        type=int,
        default=123456,
        help="Plan A selection: audit RNG seed (default: 123456).",
    )
    parser.add_argument(
        "--gene-lmm-max-genes-per-focal-var",
        dest="gene_lmm_max_genes_per_focal_var",
        type=int,
        default=None,
        help="Optional cap on selected genes per focal var (default: no cap).",
    )
    parser.add_argument(
        "--gene-lmm-explicit-genes",
        dest="gene_lmm_explicit_genes",
        type=str,
        nargs="+",
        default=None,
        help="Gene id(s) to fit with Plan A when --gene-lmm-scope=explicit.",
    )
    # Parsing the arguments
    args = parser.parse_args()
    # Call the processing function with the parsed arguments
    std_res, stats_res, resids_df, comb_stats = pmd_std_res_and_stats(args.in_file, 
                          args.out_dir, 
                          model_matrix_file = args.model_matrix_file, 
                          p_combine_idx = args.p_combine_idx,
                          in_annotation_cols = args.annotation_cols,
                          pre_regress_vars = args.pre_regress_vars,
                          n_boot = args.n_boot, 
                          seed = args.seed,
                          file_sep=args.file_type,
                          std_res_file=args.std_res_file,
                          gene_level=args.gene_level,
                          focal_vars=args.focal_vars,
                          gene_id_col=args.gene_id_col,
                          gene_methods=args.gene_methods,
                          gene_out_dir=args.gene_out_dir,
                          gene_figures=args.gene_figures,
                          gene_figures_dir=args.gene_figures_dir,
                          gene_forest_genes=args.gene_forest_genes,
                          gene_progress=args.gene_progress,
                          gene_lmm_scope=args.gene_lmm_scope,
                          gene_lmm_q_meta=args.gene_lmm_q_meta,
                          gene_lmm_q_het=args.gene_lmm_q_het,
                          gene_lmm_audit_n=args.gene_lmm_audit_n,
                          gene_lmm_audit_seed=args.gene_lmm_audit_seed,
                          gene_lmm_max_genes_per_focal_var=args.gene_lmm_max_genes_per_focal_var,
                          gene_lmm_explicit_genes=args.gene_lmm_explicit_genes)



if __name__ == "__main__":
    main()
