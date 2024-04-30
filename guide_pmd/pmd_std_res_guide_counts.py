import os
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

def run_glm_analysis(normalized_matrix, model_matrix):
    """
    Takes as input a pd.read_csv readable tsv & optionally a similar model matrix file if you want to run stats.
    This is the main function that will
    
    :param normalized_matrix: The matrix to be used for running differential abundance
    :param model_matrix: The model matrix that will be used for the GLM
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
    if "Intercept" not in model_matrix.columns:
        model_matrix.insert(0, 'Intercept', 1)
    # Preparing lists to store results
    all_res_dict = {}
    no_var_feat = []
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
            # Storing results
            temp_dict = {}
            for var_name, val in results.tvalues.items():
                temp_dict[var_name+"_t"]=val
            for var_name, val in results.pvalues.items():
                temp_dict[var_name+"_p"]=val
            all_res_dict[feature_name]=temp_dict
        else:
            no_var_feat.append(feature_name)
    p_cols = [var_name+"_p" for var_name, val in results.pvalues.items()]
    # Creating and returning the output DataFrame
    res_table = pd.DataFrame(all_res_dict).T
    del all_res_dict
    gc.collect()
    for p_col in p_cols:
        res_table[p_col+"_adj"]=nan_fdr(res_table[p_col].to_numpy())
    dummy_mat = np.zeros((len(no_var_feat),res_table.shape[1]))
    dummy_mat[:,:]=np.nan
    dummy_df = pd.DataFrame(dummy_mat, index = no_var_feat, columns = res_table.columns)
    res_table = pd.concat((res_table,dummy_df),axis=0)
    res_table = res_table.loc[normalized_matrix.index,:]
    return res_table


def get_pmd_std_res(input_file, in_annotation_cols, n_boot = 100, seed = 123456, sep = "\t"):
    # Reading data from input TSV file
    np.random.seed(seed)
    guides = pd.read_csv(input_file,sep="\t", index_col=0)
    # Processing the data (this is where you would implement your specific logic)
    print("getting the PMD standardized residuals")
    pmd_res = pmd(guides.iloc[:,(in_annotation_cols-1):], num_boot=n_boot, skip_posthoc = True)
    std_res = pmd_res.z_scores
    std_res[np.isnan(std_res)]=0.0
    return std_res, guides.iloc[:,:(in_annotation_cols-1)]


def convert_t_to_p(t_values, df):
    p_values = []
    for t_value in t_values:
        # calculate the p-value for a two-tailed test
        p = t.sf(abs(t_value), df) * 2
        p_values.append(p)
    return p_values


def get_stouf_t(t_vect):
    """Stouffer's Method of combining p-values from Z-scores. In this case we're using the t-values, but it's essentially the same."""
    return np.sum(t_vect)/np.sqrt(len(t_vect))


def get_all_comb_t(in_stats, annotation_table, annotation_col=1):
    print("getting combined T-values for all variables")
    all_keys = sorted(list(set(annotation_table.iloc[:,annotation_col-1])))
    t_cols = in_stats.columns[in_stats.columns.str.endswith("_t")].tolist()
    all_out = {}
    for key in all_keys:
        temp_row_names = annotation_table.index[annotation_table.iloc[:,annotation_col-1]==key].tolist()
        stats_sub_table = in_stats.loc[temp_row_names,:]
        temp_row_entry = {}
        for temp_t in t_cols:
            temp_row_entry[temp_t]=get_stouf_t(stats_sub_table[temp_t].to_numpy())
        all_out[key]=temp_row_entry
    return(pd.DataFrame(all_out).T)


def combine_p(in_stats, annotation_table, annotation_col, dof):
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
                            n_boot = 100, 
                            seed = 123456, 
                            file_sep="tsv"):
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
    if not os.path.isdir(output_dir): os.mkdir(output_dir)
    output_file = os.path.join(output_dir, "PMD_std_res.tsv")
    # Make sure the model matrix file actually exists if 
    if model_matrix_file is not None:
        assert os.path.isfile(model_matrix_file)
    if file_sep == "tsv":
        sep="\t"
    elif file_sep == "csv":
        sep = ","
    else:
        raise "file_sep must by either 'tsv' or 'csv'"
    # Run the PMD standardized residuals & save file
    std_res, annotation_table = get_pmd_std_res(input_file, in_annotation_cols = in_annotation_cols, n_boot = n_boot, seed = seed, sep=sep)
    std_res.to_csv(output_file,sep=sep)
    stats_res = None
    collapsed_sig = None
    # If we're running the stats, then get them going:
    if model_matrix_file is not None:
        print("running the statistics, using the specified model matrix")
        output_stats_file = os.path.join(output_dir, "PMD_std_res_stats.tsv")
        mm = pd.read_csv(model_matrix_file, sep=sep, index_col = 0)
        stats_df = run_glm_analysis(std_res, mm)
        stats_df.to_csv(output_stats_file, sep="\t")
        if p_combine_idx is not None:
            # Note that we already have the intercept, so we're already -1'ing for dof calc
            dof = std_res.shape[1]-mm.shape[1]
            comb_stats = combine_p(stats_res, annotation_table, p_combine_idx, dof)
            output_stats_file = os.path.join(output_dir, "PMD_std_res_combined_stats.tsv")
            comb_stats.to_csv(output_stats_file, sep="\t")
    return std_res, stats_res, comb_stats


def main():
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Process a TSV file and save to an output file.")
    # Adding arguments for input file and output file
    parser.add_argument("-in_file", type=str, help= "Path to the input TSV file")
    parser.add_argument("-out_file", type=str, help= "Path to the desired output file")
    parser.add_argument("-n_boot", type = int, help= "the number of bootstrap shuffled nulls to run. (Default=100)", default = 100)
    parser.add_argument("-seed", type = int, help= "set the seed for reproducibility (Default=123456)", default = 123456)
    # Parsing the arguments
    args = parser.parse_args()
    # Call the processing function with the parsed arguments
    process_file(args.in_file, args.out_file, args.n_boot, args.seed)


if __name__ == "__main__":
    main()

