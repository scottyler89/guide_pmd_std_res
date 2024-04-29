import os
import argparse
import numpy as np
import pandas as pd
from statsmodels.formula.api import GLM
from statsmodels.genmod.families import Gaussian
from scipy.stats import false_discovery_control as fdr
from percent_max_diff.percent_max_diff import pmd

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
    # Input verification
    if not isinstance(normalized_matrix, pd.DataFrame) or not isinstance(model_matrix, pd.DataFrame):
        raise ValueError("Both inputs must be pandas DataFrames.")
    if normalized_matrix.shape[1] != model_matrix.shape[1]:
        raise ValueError("Matrices must have the same number of columns (samples).")
    try:
        normalized_matrix.apply(pd.to_numeric)
        model_matrix.apply(pd.to_numeric)
    except Exception as e:
        raise ValueError("Both matrices must contain numeric values.") from e
    # Preparing lists to store results
    feature_names = []
    model_statistics = []
    p_values = []
    # Iterating through each feature (row) in the normalized matrix
    for feature_name, feature_data in normalized_matrix.iterrows():
        # Creating a GLM model
        data_df = pd.concat([feature_data, model_matrix.T], axis=1, keys=['Response', *model_matrix.index])
        formula = "Response ~ " + ' + '.join(model_matrix.index)
        model = GLM.from_formula(formula, data_df, family=Gaussian())
        results = model.fit()
        # Storing results
        feature_names.append(feature_name)
        model_statistics.append(results.tvalues[1])
        p_values.append(results.pvalues[1])
    # Creating and returning the output DataFrame
    return pd.DataFrame({
        'feature': feature_names,
        't': model_statistics,
        'p': p_values,
        "BH_adj_p":nan_fdr(p_values)
    })


def run_glm_from_file(normalized_matrix, model_matrix, sep="\t"):
    pd.read_csv()



def get_pmd_std_res(input_file, n_boot = 100, seed = 123456, sep = "\t"):
    # Reading data from input TSV file
    np.random.seed(seed)
    guides = pd.read_csv(input_file,sep="\t", index_col=0)
    # Processing the data (this is where you would implement your specific logic)
    pmd_res = pmd(guides.iloc[:,1:], num_boot=n_boot, skip_posthoc = True)
    # For simplicity, let's write it directly to the output file as is
    return pmd_res.z_scores


def process_file(input_file, 
                 output_dir, 
                 model_matrix_file = None, 
                 n_boot = 100, 
                 seed = 123456, 
                 file_sep="tsv"):
    """
    Takes as input a pd.read_csv readable tsv.
    
    :param input_file: Path to the input file (assume guid or gene id is in the first column & samples in the remaining cols).
    :param output_file: Path to the directory for the desired output.
    :param model_matrix_file: Path to the model matrix (optional, if included, will perform a GLM on the pmd std res of the guides).
    :param n_boot: Number of bootstrap shuffles to do for null (default = 100).
    :param seed: random seed (default = 123456).
    :param file_sep: tsv for tab separapted, csv for comma (default = tsv).
    """
    if not os.path.isdir(output_dir): os.mkdir(output_dir)
    output_file = os.path.join(output_dir, "PMD_standardized_residuals.tsv")
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
    std_res = get_pmd_std_res(input_file, n_boot = n_boot, seed = seed, sep=sep)
    std_res.to_csv(output_file,sep=sep)
    stats_res = None
    # If we're running the stats, then get them going:
    if model_matrix_file is not None:
        run_glm_from_file(std_res, model_matrix_file)
    return std_res, stats_res


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


