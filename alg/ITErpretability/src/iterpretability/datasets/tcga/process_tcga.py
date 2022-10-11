import os
import numpy as np
import pandas as pd
import pickle
from scipy.stats import entropy


def filter_genes(gene_data_df, num_genes, selection_criteria):
    genes = gene_data_df.columns.values

    genes_information = pd.DataFrame(columns=genes)

    row_info = []
    for gene in genes:
        expression_values = gene_data_df[gene].values
        gene_information = 0
        if selection_criteria == "cv":
            gene_information = np.std(expression_values) / np.mean(expression_values)
        elif selection_criteria == "entropy":
            gene_information = entropy(expression_values)

        row_info.append(gene_information)

    genes_information.loc[0] = row_info
    genes_information = genes_information.sort_values(by=0, ascending=False, axis=1)
    genes_information = genes_information[genes_information.columns.values[:num_genes]]

    return gene_data_df[genes_information.columns.values]


def normalize_data(X):
    X_normalized = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

    return X_normalized


def process_tcga(max_num_genes, file_location=""):
    try:
        tcga_dataset = pickle.load(
            open(os.path.join(file_location, "tcga_full_dataset.p"), "rb")
        )
    except:
        raise FileNotFoundError(
            "Full TCGA dataset needs to be downloaded from: https://drive.google.com/file/d/1NveePKQscxJ-VZacOm9MHEAVvPKOQJW8/view?usp=sharing"
        )

    # Log normalize the gene expression data
    tcga_dataset["rnaseq"] = np.log(np.array(tcga_dataset["rnaseq"]) + 1.0)

    # Remove genes that have constant gene expression across all patients
    tcga_dataset["rnaseq"] = tcga_dataset["rnaseq"][
        :,
        np.where(
            np.min(tcga_dataset["rnaseq"], axis=0)
            - np.max(tcga_dataset["rnaseq"], axis=0)
            != 0
        )[0],
    ]

    # Select max_num_genes
    filtered_gene_data = filter_genes(
        pd.DataFrame(
            tcga_dataset["rnaseq"], columns=range(tcga_dataset["rnaseq"].shape[1])
        ),
        num_genes=max_num_genes,
        selection_criteria="entropy",
    )

    # Normalize data to [0, 1]
    tcga_dataset["rnaseq"] = normalize_data(filtered_gene_data.values)

    pickle.dump(
        tcga_dataset,
        open(os.path.join(file_location, "tcga_" + str(max_num_genes) + ".p"), "wb"),
    )


if __name__ == "__main__":
    process_tcga(max_num_genes=100)
