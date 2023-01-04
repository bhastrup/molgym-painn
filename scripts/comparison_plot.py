import argparse
from dataclasses import dataclass
from json import load
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from molgym.tools.analysis import (
    parse_json_lines_file,
    parse_results_filename
)

# Styling
fig_width = 0.45 * 5.50107
fig_height = 2.1

plt.style.use('ggplot')
plt.rcParams.update({'font.size': 6})

colors = [
    '#1f77b4',  # muted blue
    '#d62728',  # brick red
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf',  # blue-teal
]

@dataclass
class Args:
    name: str
    models: List[str]
    labels: List[str]
    log_dir: str
    fig_scale: float
    show: bool
    no_save: bool
    mode: str


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description=("Plotting MolGym output across models. Automatically "
		     "includes all runs in subfolders that matches the "
		     "specified model(s).")
    )

    parser.add_argument('--name', help="The run name to plot",
                        required=True)
    parser.add_argument('--models', help="The models to plot",
                        required=True, nargs='+', type=str)
    parser.add_argument('--labels',
                        help="The legend labels in the order of `models`",
                        nargs='+', default=None, type=str)
    parser.add_argument('--log_dir',
                        help="The same as was given for running the model",
                        default='logs', type=str)
    parser.add_argument('--fig_scale', help="Scaling factor for plot",
                        default=1.0, type=float)
    parser.add_argument('--show', help="Show the plot", action='store_true')
    parser.add_argument('--no-save', help="Don't save the plot",
                        action='store_true')
    parser.add_argument('--mode',
                        help='train or eval mode',
                        required=False,
                        type=str,
                        choices=['train', 'eval'],
                        default='eval')

    return parser.parse_args(namespace=Args)

def get_data(
    models: List[str], name: str, mode: str, log_dir: str
) -> pd.DataFrame:
    # Find all the files that matches the name and mode
    paths = list(Path.cwd().glob(f'**/{name}_run-*_{mode}.txt'))

    assert len(paths) > 0

    id_paths: List[Path, dict] = []
    for path in paths:
        info = parse_results_filename(path.name)
        log_path = (path.parent.parent /
                    log_dir /
                    f"{name}_run-{info['seed']}.json")

        # Load run parameters, remove the models not considered and "sort"
        # the paths by model
        with open(log_path) as fd:
            parameters = load(fd)
        if parameters['model'] not in models:
            continue

        id_paths.append((path, parameters))

    frames = []
    for path, params in id_paths:
        df = pd.DataFrame(parse_json_lines_file(path))

        df['model'] = params['model']
        df['seed'] = params['seed']
        df['mode'] = mode

        frames.append(df)

    data = pd.concat(frames)

    # Compute average and std over seeds
    data = data.groupby(
        ['model', 'mode', 'total_num_steps']
    ).agg([np.mean, np.std]).reset_index()

    return data

def main():
    args = parse_args()
    if not args.labels:
        args.labels = args.models
    
    assert len(args.models) == len(args.labels), "When given, the number of labels must match number of models"

    legend_labels = {m: l for m, l in zip(args.models, args.labels)}

    data = get_data(args.models, args.name, args.mode, args.log_dir)

    fig, ax = plt.subplots(
        nrows=1, ncols=1,
        figsize=(args.fig_scale*fig_width, args.fig_scale*fig_height),
        constrained_layout=True
    )
    color_iter = iter(colors)

    prop = 'return_mean'
    for j, (model, group) in enumerate(data.groupby('model')):
        color = next(color_iter)

        if group[prop]['mean'].isna().all():
            continue
        # The mean line
        ax.plot(
            group['total_num_steps'] / 1000,
            group[prop]['mean'],
            alpha=0.7,
            zorder=2 * j + 3,
            label=legend_labels[model],
            color=color,
            linewidth=0.7,
        )
        # The indication of the std
        ax.fill_between(
            x=group['total_num_steps'] / 1000,
            y1=group[prop]['mean'] - group[prop]['std'],
            y2=group[prop]['mean'] + group[prop]['std'],
            alpha=0.2,
            zorder=2 * j + 2,
            color=color,
        )
    
    ax.set_title(args.name)
    ax.set_ylabel('Average Return')
    ax.set_xlabel('Steps x 1000')

    ax.legend(loc='lower right')

    if args.show: plt.show()
    if not args.no_save: fig.savefig(f'return_comparison_{args.name}.pdf')

if __name__ == '__main__':
    main()
