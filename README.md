# bayes_diffexpr
## Bayesian Inference of Differential Expression

This is the repository for the research associated with the publication:
Puelma Touzel M, Walczak AM, Mora T (2020) Inferring the immune response from repertoire sequencing. PLOS Computational Biology 16(4): e1007873. https://doi.org/10.1371/journal.pcbi.1007873 

The `master` branch is an initial, compact version of the codebase. The `depaper` branch is has the files used the publication. This includes a more elaborate codebase with analysis in `jupyter` notebooks converted into `.py` files using `jpuytext`, as well the `.tex` files used to generate the initial submission and, finally, some short `EdgeR` code, used for comparison with some results in Pogorelyy, M et al.PNAS 2018. Example datasets used in with this code can be found in the repository I set up for that paper https://github.com/mptouzel/pogorelyy_et_al_2018.

# Main files
`infer_diffexpr_main.py` takes 2 pairs of `.txt` files as input. 

Each file is a csv table with each row referring to one observed clone. What is pulled from these files is the nucleotide and amino acid sequence as well as the observed clone count (n.b. file header information, e.g. column order, as well as the paths to these files, has been hard-coded to work with a particular dataset; any application to another dataset with require changing this information). Each pair is merged into a data set of pair counts, one sample for each observed clone.

Using functions in the lib folder, `infer_diffexpr_main.py` learns a null model of variability based on the first pair (e.g. two replicates), and then learns the parameters of a distribution of log fold change, using the second pair. Finally, the script computes posteriors of log fold change for each clone and writes a csv table of summary statistics for these posteriors.  

`infer_diffexpr_main.py` for required libraries. All are standard.

There are additional files for processing the output of `infer_diffexpr_main.py`. Some are `jupyter` notebooks, which have accompanying version-controlled `.py` scripts managed by `jupytext`.

# Other files
##plotting
`plot_output.ipynb` reads in the outputed data and plots the likelihood surface over the computed grid of parameter values of the particular P(s) used.

## EdgeR comparison
In the directory `edgeR` are single and batch versions of `R` scripts that use the `edgeR` package to compute a list of significantly expanded clones. 

## manuscript
In the directory `paper` is the latex that renders the current version of the manuscipt.
