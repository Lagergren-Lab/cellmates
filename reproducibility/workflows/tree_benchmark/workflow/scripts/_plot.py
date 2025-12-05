import os
import pandas as pd
import seaborn as sns

from _utils import plot_comparison

def main(input_csv, output_plot_file):

    sns.set_theme(style="whitegrid")
    # input_list: list of csv files to merge
    df = pd.read_csv(input_csv)
    g = plot_comparison(df)
    g.savefig(output_plot_file, dpi=300)
    return


if __name__ == '__main__':
    workflow_dir = "/Users/zemp/PycharmProjects/cellmates-cmp/reproducibility/workflows/tree_benchmark/"
    main(
        input_csv=os.path.join(workflow_dir, "benchmark_results/res_merged.csv"),
        output_plot_file=os.path.join(workflow_dir, "benchmark_results/res_plot.png"),
    )

