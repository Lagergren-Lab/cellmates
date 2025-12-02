import pandas as pd

from _utils import plot_comparison

def main(input_list, output_csv_file, output_plot_file):
    # input_list: list of csv files to merge
    merged_res = []
    for input_file in input_list:
        with open(input_file, 'r') as f:
            header = f.readline().strip().split(',')
            values = f.readline().strip().split(',')
            res_dict = {header[i]: values[i] for i in range(len(header))}
            merged_res.append(res_dict)
    df = pd.DataFrame(merged_res)
    # save to csv
    df.to_csv(output_csv_file, index=False)

    plot_comparison(df, output_plot_file)
    return


if __name__ == '__main__':
    # snakemake object available
    main(
        input_list=snakemake.input,
        output_csv_file=snakemake.output.csv_file,
        output_plot_file=snakemake.output.plot_file,
    )
