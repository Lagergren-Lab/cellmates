# Tree reconstruction benchmark
> Reproduce the tree reconstruction benchmark on simulated data for the Cellmates paper

The workflow consists of three main steps:
- simulate trees with associated leaves distance matrix (varying number of leaves, mutation rates and tree shapes)
- reconstruct trees from the distance matrices using different methods and compute scores against the ground truth trees
- combine results and generate plots to visualize the performance of the different methods

## Execution
To execute the workflow on HPC, run
```bash
snakemake --profile workflow/profile
```
This will use the conda environments specified in the Snakefile and the cluster configuration in `workflow/profile/config.yaml`.
The parameters for the simulation and reconstruction can be modified in the `config/config.yaml` file.
The results will be generated in `benchmark_results/` folder as follows
```
- benchmark_results/
    - partials/
        - dm_<params>.npy               # [TEMP] distance matrix for given parameters
        - tree_<params>.nwk             # [TEMP] ground truth tree for given parameters
        - res_<params>.csv              # reconstruction results for given parameters
    - plots.pdf                         # combined plots of the benchmark results
    - res_merged.csv                    # combined results of the benchmark
```
While distance matrices and trees are deleted after reconstruction, the scores for each parameter set are kept so that
when re-running workflow with additional methods or parameters, only the missing reconstructions are performed.

Test execution on local machine can be done with
```bash
snakemake --profile --configfile config/demo.yaml
```

## Environment

The environment requires `cellmates` to be installed as a local package.
After letting snakemake create the conda environments `cellmates-min.yaml`,
you can install cellmates in the relevant environments with
```bash
conda activate <env_name>
pip install -e ../../..
```
where `<env_name>` is the name of the environment created by snakemake, which you can
find in the `.snakemake/conda` folder.

This intermediate step will be removed in future versions of the workflow.