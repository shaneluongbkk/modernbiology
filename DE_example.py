import pandas as pd
import numpy as np
import requests
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

# ------------------------------------------------------------------------------
# Download data
# ------------------------------------------------------------------------------

# Download raw counts
counts_url = "https://www.ebi.ac.uk/gxa/experiments-content/E-MTAB-5244/resources/DifferentialSecondaryDataFiles.RnaSeq/raw-counts"
counts = pd.read_csv(counts_url, sep='\t')
print(counts.head())

# Download metadata
metadata_url = "https://www.ebi.ac.uk/gxa/experiments-content/E-MTAB-5244/resources/ExperimentDesignFile.RnaSeq/experiment-design"
metadata = pd.read_csv(metadata_url, sep='\t')
print(metadata.head())

# ------------------------------------------------------------------------------
# Wrangle data for DESeq2 (Python equivalent)
# ------------------------------------------------------------------------------

# DESeq expects the counts to have gene IDs as row names
counts.set_index('Gene ID', inplace=True)
print(counts.head())

# Remove unused columns (Gene Name)
genes = pd.DataFrame({
    'Gene ID': counts.index,
    'Gene Name': counts['Gene Name']
})
counts.index.name = None
counts = counts.drop(columns=['Gene Name'])
print(counts.head())

# DESeq expects the metadata matrix to have sample IDs in the rownames
metadata.set_index('Run', inplace=True)
metadata = metadata[['Sample Characteristic[genotype]']]
metadata.columns = ['genotype']
metadata['genotype'] = metadata['genotype'].replace({'wild type genotype': 'wildtype', 'Snai1 knockout': 'knockout'})
metadata['genotype'] = metadata['genotype'].astype('category')
print(metadata.head())

# ------------------------------------------------------------------------------
# Spot check expression for knockout gene SNAI1
# ------------------------------------------------------------------------------

gene_id = genes['Gene ID'][genes['Gene Name'] == 'SNAI1'].values[0]
gene_counts = counts.loc[gene_id]
gene_data = pd.concat([metadata, pd.Series(gene_counts, name='counts')], axis=1)

# Plot gene expression for knockout vs wildtype
plt.figure(figsize=(8, 6))
sns.boxplot(x='genotype', y='counts', data=gene_data, palette="Set2")
plt.title('Gene expression for SNAI1 (Knockout vs Wildtype)')
plt.show()

# ------------------------------------------------------------------------------
# Run Differential Expression (DESeq2)
# ------------------------------------------------------------------------------

# DESeq2 requires the counts matrix to be in the form of samples x genes
counts = counts.transpose()

# Filter out lowly expressed genes
counts = counts.loc[:, counts.sum(axis=0) > 10]

# Create a DESeqDataSet object
dds = DeseqDataSet(
    counts=counts,  # counts should be a pandas DataFrame
    metadata=metadata,  # metadata should be a pandas DataFrame indexed by sample IDs
    design="~ genotype",  # specify the design formula with genotype
    refit_cooks=True,  # refit cooks outliers
    min_mu=0.5  # minimum mean estimate
)

# Run DESeq2
dds.deseq2()

# Compare expression with a specific contrast
contrast = ("genotype", "knockout", "wildtype")
results = DeseqStats(dds, contrast=contrast, alpha=1e-5)

# Obtain the result DataFrame
res = results.summary()