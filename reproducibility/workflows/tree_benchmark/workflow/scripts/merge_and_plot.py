import pandas as pd

from _utils import plot_comparison

def main(input_list, output_csv_file, output_plot_file):
    # input_list: list of csv files to merge
    merged_res = []
    for input_file in input_list:
        df = pd.read_csv(input_file)
        merged_res.append(df)
    df = pd.concat(merged_res, ignore_index=True)
    # save to csv
    df.to_csv(output_csv_file, index=False)

    plot_analysis_pdf(df, output_plot_file)
    return


if __name__ == '__main__':
    # snakemake object available
    main(
        input_list=snakemake.input,
        output_csv_file=snakemake.output.csv_file,
        output_plot_file=snakemake.output.plot_file,
    )
