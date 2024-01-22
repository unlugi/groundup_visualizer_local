import json
import os
import click

import numpy as np


# click options with a flag for scores files that allows multiple
@click.command()
@click.option('--scores_file', '-s', multiple=True, type=click.Path(exists=True),
              help='Scores file(s) to print.')
def cli(
    scores_file: list[str]
):
    score_data = {}
    # loop through all the scores json files and load them
    for file in scores_file:
        with open(file, 'r') as f:
            data = json.load(f)
            score_data[data["sample_path"]] = data
            data.pop("sample_path")
            data.pop("num_samples")

    for name in score_data:
        print(f"Scores for {name}:")
        for key in score_data[name]:
            print(f"{key}: {score_data[name][key]}")
        print() 
    
    
if __name__ == '__main__':
    cli()