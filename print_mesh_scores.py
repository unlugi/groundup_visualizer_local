import json
import os
import click

import numpy as np

import pandas as pd
from tabulate import tabulate

# click options with a flag for scores files that allows multiple
@click.command()
@click.option('--scores_file', '-s', multiple=True, type=click.Path(exists=True),
              help='Scores file(s) to print.')
@click.option('--pretty_names', '-p', multiple=True, type=str,
              help='pretty_names', default=None)
def cli(
    scores_file: list[str],
    pretty_names: list[str] = None,
):
    print("HELLOOOOOO")
    if len(pretty_names) != 0:
        assert len(scores_file) == len(pretty_names), "Number of pretty names must match number of scores files"
    score_data = {}
    # loop through all the scores json files and load them
    for file_ind, file in enumerate(scores_file):
        with open(file, 'r') as f:
            data = json.load(f)
            if len(pretty_names) != 0:
                pretty_name = pretty_names[file_ind]
            else:
                pretty_name = data["sample_path"]
                
            score_data[pretty_name] = data
            
            data.pop("sample_path")
            data.pop("num_samples")

    df = pd.DataFrame.from_dict(score_data, orient='index')

    # Set the desired column order
    column_order = ['acc↓', 'compl↓', 'chamfer↓', 'precision↑', 'recall↑', 'f1_score↑']  # Replace with your desired column names

    # Reindex the DataFrame with the specified column order
    df = df.reindex(columns=column_order)
    
    print(tabulate(df, headers = 'keys', tablefmt = 'psql'))
    print(tabulate(df, headers = 'keys', tablefmt = 'latex_raw'))
    
    
    # df = pd.DataFrame.from_dict(score_data, orient='index')
    
            
    # Loop over each column
    for column in df.columns:
        # Apply rounding based on column name
        if column == 'acc↓' or column == 'compl↓' or column == 'chamfer↓':
            df[column] = df[column].apply(lambda x: f"{100*x:.2f}")
        elif column == 'precision↑' or column == 'recall↑' or column == 'f1_score↑':
            df[column] = df[column].apply(lambda x: f"{100*x:.1f}")
            

    # Loop over each column again to find the maximum value
    for column in df.columns:
        # Find the maximum value in the column
        if column == 'acc↓' or column == 'compl↓' or column == 'chamfer↓':
            max_value = df[column].min()
        elif column == 'precision↑' or column == 'recall↑' or column == 'f1_score↑':
            max_value = df[column].max()
    
        # Apply formatting to the entry if it's the largest value
        df[column] = df[column].apply(lambda x: f"\\textbf{{{x}}}" if x == max_value else x)

    print(tabulate(df, headers='keys', tablefmt='latex_raw'))


if __name__ == '__main__':
    cli()
