# bayes_diffexpr
## Bayesian Inference of Differential Expression

`infer_diffexpr_main.py` takes 2 pairs of `.txt` files as input. 

Each file is a table with each row referring to one observed clone. What is pulled from these files is the nucleotide and amino acid sequence as well as the observed clone count (n.b. file header information, e.g. column order, as well as the paths to these files, has been hard-coded to work with a particular dataset; any application to another dataset with require changing this information).

Using functions in `infer_diffexpr_lib.py`, `infer_diffexpr_main.py` learns a null model of variability based on the first pair (e.g. two replicates), and then learns the parameters of a distribution of log fold change, using the second pair. Finally, the script computes posteriors of log fold change for each clone and writes a csv table of summary statistics for these posteriors.  

