# Cellmates

> A maximum likelihood method for single-cell phylogeny reconstruction with proper evolutionary distances from copy numbers

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Overview

Cellmates is a computational tool for reconstructing phylogenetic trees from single-cell copy number data. It uses an Expectation-Maximization (EM) algorithm to estimate evolutionary distances between cells based on their copy number profiles, enabling accurate reconstruction of tumor evolution.

### Key Features

- **Maximum likelihood phylogenetic inference** from single-cell copy number data
- **Centroid-based distance estimation** for improved tree topology accuracy
- **Copy number prediction** for internal nodes (ancestors) in the tree
- **Support for multiple observation models** (Normal, Poisson, Negative Binomial)
- **Parallel processing** for faster inference on large datasets

## Requirements

- **Python**: 3.10 or higher
- **Operating System**: Linux, macOS, or Windows with WSL

### Core Dependencies

- NumPy >= 1.26
- SciPy >= 1.15
- NetworkX >= 3.4
- DendroPy >= 5.0
- anndata >= 0.11
- scikit-bio >= 0.6
- pomegranate >= 1.1
- BioPython >= 1.85
- tqdm
- matplotlib
- seaborn

## Installation

### Option 1: Using Conda (Recommended)

The easiest way to install Cellmates is using conda to manage dependencies:

```bash
# Clone the repository
git clone https://github.com/Lagergren-Lab/cellmates.git
cd cellmates

# Create conda environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate cellmates

# Install Cellmates in editable mode
pip install -e .
```

### Option 2: Using pip with Virtual Environment

```bash
# Clone the repository
git clone https://github.com/Lagergren-Lab/cellmates.git
cd cellmates

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Cellmates
pip install -e .
```

### Option 3: Quick Installation Test

Use the provided installation script to automatically set up and test the environment:

```bash
./scripts/install_and_test.sh
```

### Development Installation

For development, install with optional dev dependencies:

```bash
pip install -e ".[dev]"
```

## Quick Start

### Command Line Interface

Run Cellmates from the command line:

```bash
cellmates --input <input_file> --output <output_path> [options]
```

For detailed usage instructions:

```bash
cellmates --help
```

### Python API

```python
from cellmates.inference.pipeline import run_inference_pipeline

# Run the complete inference pipeline
results = run_inference_pipeline(
    input="path/to/data.h5ad",
    output="path/to/output/",
    n_states=7,
    max_iter=30,
    num_processors=4
)

# Access results
print(f"Distance matrix saved to: {results['distances']}")
print(f"Tree saved to: {results['tree']}")
```

### Example with Demo Data

An example input file is provided in the `reproducibility` directory:

```bash
cellmates --input reproducibility/demo.h5ad --output results/ --n-states 8 --num-processors 4 --predict-cn
```

## Input Format

The input data should be an HDF5 AnnData file (`.h5ad`) containing:

- **`X`** or **`layers/copy`**: Cell-by-bin corrected read counts matrix
- **`var`**: DataFrame with bin information (must include `chr` column for chromosome)
- **`obs`** (optional): DataFrame with cell metadata
  - `obs_names`: Cell identifiers (used in output tree)
  - `normal` column: Boolean indicating normal cells (will be excluded from analysis)

### Data Preparation Tips

1. **Remove normal cells** before running Cellmates, or add an annotation in `obs['normal']` to identify them
2. Ensure read counts are **GC-corrected and normalized**
3. For **CNAsim-generated data**, correction is applied automatically

## Output Files

Running Cellmates produces the following output files:

| File | Description |
|------|-------------|
| `distance_matrix.npy` | Pairwise triplet-distance matrix (n_cells × n_cells × 3) |
| `tree.nwk` | Reconstructed phylogenetic tree in Newick format |
| `cell_names.txt` | Cell names in order matching the distance matrix |
| `predicted_copy_numbers.npz` | Predicted CN profiles for cells and internal nodes (optional) |
| `em_diagnostics.pkl` | EM algorithm diagnostics (optional, with `--save-diagnostics`) |

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n-states` | 7 | Number of copy number states |
| `--max-iter` | 30 | Maximum EM iterations |
| `--num-processors` | 1 | Number of parallel processors |
| `--rtol` | 1e-3 | Relative tolerance for convergence |
| `--tau` | 10.0 | Precision parameter for Normal observation model |
| `--alpha` | 1.0 | Rate parameter for evolutionary model |
| `--predict-cn` | False | Predict CN profiles for internal nodes |
| `--use-copynumbers` | False | Use copy number states directly (instead of read counts) |
| `--save-diagnostics` | False | Save EM diagnostic data |

## Testing

Run the test suite to verify installation:

```bash
# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_inference/ -v

# Run quick tests only
pytest tests/ -v -m "not slow"
```

## Citation

If you use Cellmates in your research, please cite:

```bibtex
@article{cellmates2024,
  title={Cellmates: Maximum likelihood phylogenetic inference from single-cell copy number data},
  author={Zampinetti, Vittorio and Melin, Harald and Abdolhamdi, Marzie and Lagergren, Jens},
  journal={},
  year={2024}
}
```

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contact

- **Vittorio Zampinetti** - vz@kth.se
- **Harald Melin** - haralme@kth.se
- **Marzie Abdolhamdi** - marziea@kth.se

Project Link: [https://github.com/Lagergren-Lab/cellmates](https://github.com/Lagergren-Lab/cellmates)
