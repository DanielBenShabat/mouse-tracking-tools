"""
Run the preprocessing on many experiments at once.

Expected layout (one folder per experiment, as organized in Dropbox):
    <experiments_directory>/<experiment_name>/data/<any name>.zip   (one zip with one CSV per participant)

Each experiment is unzipped into a temporary folder, preprocessed, and saved as
<output_directory>/<experiment_name>.csv

Usage:
    python code/run_experiments.py <experiments_directory> [--output <output_directory>] [--only name1 name2 ...]
"""
import argparse
import glob
import os
import sys
import tempfile
import traceback
import zipfile

from process_across_subjects import process_across_subjects

# Settings used for every experiment, unless overridden in EXPERIMENT_SETTINGS
DEFAULT_SETTINGS = {
    'x_cord_column': 'x_cord',
    'y_cord_column': 'y_cord',
    'rt_column': 'rt',
    'time_column': 'time_cord',
    'response_column': 'response',  # slider after each trial; experiments without one get an empty column
    'columns_to_preserve': [],
    'practice_mode': 'marker',  # drop everything before the "You are now ready to begin the task." screen
    'practice_marker': 'ready to begin the task',
    'num_practice_trials': 0,
    'num_trials': None,
}

# Per-experiment overrides of DEFAULT_SETTINGS
EXPERIMENT_SETTINGS = {
    'sntr_3blcks': {
        # no slider after each trial; keep the willingness-to-pay bids and their confidence ratings as extra rows
        'response_column': '',
        'columns_to_preserve': ['bid_stimulus', 'confidence_bid'],
    },
}


def find_csv_directory(root):
    """Return the single folder under `root` that contains the participant CSV files."""
    directories = set()
    for path in glob.glob(os.path.join(root, '**', '*.csv'), recursive=True):
        if '__MACOSX' not in path:
            directories.add(os.path.dirname(path))
    if len(directories) != 1:
        raise ValueError('expected one folder with CSV files in the zip, found %d: %s' % (len(directories), sorted(directories)))
    return directories.pop()


def run_experiment(experiment_directory, output_directory):
    name = os.path.basename(os.path.normpath(experiment_directory))
    zips = glob.glob(os.path.join(experiment_directory, 'data', '*.zip'))
    if len(zips) != 1:
        raise ValueError('expected one zip file in %s, found %d' % (os.path.join(experiment_directory, 'data'), len(zips)))

    settings = dict(DEFAULT_SETTINGS)
    settings.update(EXPERIMENT_SETTINGS.get(name, {}))

    with tempfile.TemporaryDirectory() as temp_directory:
        with zipfile.ZipFile(zips[0]) as archive:
            archive.extractall(temp_directory)
        data_directory = find_csv_directory(temp_directory)
        return process_across_subjects(data_directory, output_directory, output_name=name, **settings)


def main():
    parser = argparse.ArgumentParser(description='Preprocess every experiment folder in a directory.')
    parser.add_argument('experiments_directory')
    parser.add_argument('--output', help='output folder (default: "output" next to the experiments folder)')
    parser.add_argument('--only', nargs='+', help='names of the experiments to run (default: all)')
    args = parser.parse_args()

    experiments_directory = os.path.abspath(args.experiments_directory)
    output_directory = args.output or os.path.join(os.path.dirname(experiments_directory), 'output')
    os.makedirs(output_directory, exist_ok=True)

    names = sorted(entry for entry in os.listdir(experiments_directory)
                   if os.path.isdir(os.path.join(experiments_directory, entry)))
    if args.only:
        missing = set(args.only) - set(names)
        if missing:
            sys.exit('unknown experiments: %s' % ', '.join(sorted(missing)))
        names = [name for name in names if name in args.only]

    failed = []
    for name in names:
        print('\n=== %s ===' % name)
        try:
            print('saved', run_experiment(os.path.join(experiments_directory, name), output_directory))
        except Exception:
            traceback.print_exc()
            failed.append(name)

    print('\nDone: %d succeeded, %d failed%s' % (len(names) - len(failed), len(failed),
                                                 (': ' + ', '.join(failed)) if failed else ''))
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
