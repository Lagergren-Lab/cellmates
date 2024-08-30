# Cellmates – a proper distance method for single-cell cancer data

## Project status

This project is currently under development. The functionalities that have been implemented are described below.

- [x] Data simulation (main code in function `rand_dataset()` at [datagen.py](src/simulation/datagen.py))
- [x] EM algorithm for root-to-centroid distance (main code in function `jcb_em()` at [em.py](src/inference/em.py))
- [x] Tree reconstruction algorithm from distance matrix
(main code in function `build_tree()` at [em.py](src/inference/em.py))
- [x] Testing for the execution of all three steps described above in [test_tree_inference_synth()](tests/test_inference/test_em.py)

The pseudo-algorithm with the first distance model (with $\varepsilon \in [0,1]$) is explained in the
[general writeup](https://www.overleaf.com/project/62e11c46a9cd5d7659fc29b4)
at Cellmates chapter, while an in-depth explanation of the JCB EM
algorithm using additive lengths is in the [Cellmates article](https://www.overleaf.com/project/63a43d8ae9c4f58d4f48653c).

The next steps are:

- [ ] Merge two documents in order to have only one standalone Cellmates writeup
- [ ] Check that both EM algorithms correctly estimate distances in a single quadruplet case (2 cells only) assuming known CN profiles
- [ ] Implement NJ for tree reconstruction comparisons (move from notebook to code)
- [ ] Add observation model parameters inference (whether with EM or via other softwares e.g. SCOPE)

## Requirements

- Python 3.10
- NumPy
- DendroPy
- NetworkX
- anndata

```bash
pip install -r requirements.txt
```