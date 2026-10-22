# Mouse tracking preprocessing and analysis tools
In this repository, you will find scripts for preprocessing, extracting measurements, and visualizing raw mouse-tracking data collected from online experiments.
Based on [Freeman & Ambady (2010)](https://link.springer.com/article/10.3758/BRM.42.1.226), mouse trajectories are interpolated to 100 time points, rescaled, and remapped for each trial. 
Additionally, it calculates classical mouse tracking measures, such as the area under the curve and x-flips, as well as new ones. A variety of data visualization options are also available. 

## Usage guidelines
The scripts were originally built for experimental data obtained in the Kleiman lab at the Hebrew University of Jerusalem, using a jsPsych-based implementation of a mouse tracker. 

Thus, for it to work on your dataset:
- The dataset should be structured so that there are individual .csv files for each participant. 
- All x coordinates should be in a specific column, and all y coordinates should be in another column. The names of these columns should be stated in the main.py file, using the global variables X_CORD_COLUMN and Y_CORD_COLUMN

The visualization functionality can work with up to 2 condition factors. The first factor controls the color, and the second factor controls the subplot. The number of conditions in each factor is unlimited. Each factor's condition information should be in a different column, and the names of these columns should be stated in the main.py file, using the global variables FIRST_CONDITION_COLUMN and SECOND_CONDITION_COLUMN.

Please feel free to contact me (ariel.levy2@mail.huji.ac.il) for any further clarification.

## Contents

### Data
The data folder includes mock data of three 'participants' in a generic experiment with one factor containing two conditions. This is only for demonstration purposes.
You can copy your own data files to this folder in order to process them. Alternatively, you can define 

### Code:

#### Preprocessing.py

This class contains all preprocessing-related functions for mouse tracking data, including normalization, rescaling (to deal with data from different browsers and coordinate systems), and remapping. It also extracts six mouse-mouse-tracking-based measures:
1. x-flips: the number of times a participant shifted direction during the trial.
2. RPB (Returns to the Point of Balance): the number of times a participant crossed the line from one side of the screen to the other.
3. AUC (Area Under the Curve): the area between the actual trajectory and a straight line connecting the starting point and the chosen option.
4. MD (Maximal Deviation): the maximal deviation from the actual trajectory to a straight line connecting the starting position and the chosen option.
5. initiation angle: the angle between the starting of the trajectory to the x-axis
6. initiation correspondence: determines whether the initiation angle is below 90, i.e., whether it corresponds to the direction of the chosen option.

It also extracts kinematic measures. `mean_velocity`, `max_velocity`, `mean_acceleration` and the profiles `v_0..v_100` / `a_0..a_100` assume that the 101 points are equally spaced in time (RT / 100), so `mean_velocity` equals trajectory length / RT.

When a timestamp column is given (`TIME_COLUMN`, e.g. `time_cord`), velocity is also calculated from the real timestamps of the samples. Each trial is resampled every 10 ms in real time, and velocity is calculated with a Savitzky-Golay filter over 50 ms. The browser sometimes records two samples only 1-4 ms apart, and without smoothing this timing jitter shows up as velocity spikes. All measures are in the rescaled space (velocity in units/s, acceleration in units/s², times in ms). A trial is "moving" while its speed is at least 5% of its peak speed.
- `tv_0..tv_100`: velocity profile over real time, at 101 points equally spaced from the first to the last recorded sample.
- `velocity_peak`: peak velocity.
- `velocity_time_to_peak`: time from movement onset to peak velocity.
- `velocity_peak_relative`: when the peak happens within the movement, from 0 (at onset) to 1 (at the end). Values around 0.5 mean a symmetric, bell-shaped profile.
- `velocity_num_peaks`: number of velocity peaks (prominence of at least 10% of the peak). 1 means a single smooth movement, more means corrections or sub-movements.
- `idle_time`: time the mouse stood still between movement onset and the end of the movement.
- `initiation_time`: time from the start of the trial until the movement started. Recording starts when the mouse first moves, so this is RT minus the recorded duration, plus the time until the speed crossed the threshold.
- `recorded_duration`: time from the first to the last recorded sample.
- `peak_acceleration` / `peak_deceleration`: maximal acceleration and maximal deceleration (both positive).

Subjects who only ever chose one side cannot be rescaled. They are marked with `is_OK = False`, a warning is printed, and their measures are left empty.

#### process_across_subjects.py

This function runs preprocessing iteratively over all data files in the data folder and outputs a unified data file with all mouse measures calculated.

#### Visualization.py

This class contains functions that visualize the raw mouse trajectories and mean trajectories of participants in the experiment.

Example plots made by this script can be found in the 'output' folder.

#### main.py

Use this file as the “control panel” for the entire toolkit. You do not need to modify the other Python files—just adjust the settings here and run the script.

1. **Open your data folder**  
   Place one CSV per participant inside `data/`. Each CSV must have the mouse-coordinate columns (see next step).

2. **Tell the script how to read your columns**  
   Set the names of your x/y coordinate columns and (optionally) condition columns:
   ```python
   X_CORD_COLUMN = 'x_cord'
   Y_CORD_COLUMN = 'y_cord'
   FIRST_CONDITION_COLUMN = 'trajectory'
   SECOND_CONDITION_COLUMN = ''
   ```
   - Leave a column blank (`''`) if you do not collect that information.
   - `COLUMNS_TO_PRESERVE` lets you keep non-trajectory rows (e.g., questionnaires) in the final CSV.

3. **Handle practice trials**  
   Choose how to drop practice trials:
   ```python
   PRACTICE_MODE = 'auto'    # removes rows whose test_part starts with “practice”
   # or
   PRACTICE_MODE = 'manual'
   NUM_PRACTICE_TRIALS = 2   # number of practice trajectories to discard
   NUM_TRIALS = 66           # number of experimental trajectories to keep
   ```
   - In manual mode, set both `NUM_PRACTICE_TRIALS` and `NUM_TRIALS` to match your task.
   - A third mode, `'marker'`, removes everything up to and including the "You are now ready to begin the task." screen. It works even when the practice trials are labelled like the real ones. In this mode, each slider response is attached to the trial right before it.

4. **Customize the plots (optional)**  
   Update the visualization settings (titles, colors, fonts, subject to inspect). These only affect the charts saved to the `output/` folder.

5. **Choose what to run**  
   - `PREPROCESS = True` runs the full pipeline (process data + create plots).  
   - Set `PREPROCESS = False` if you already have a processed CSV and only want to visualize it by pointing `ALTERNATIVE_VIS_PATH` to that file.

6. **Run the script**  
   From the project root:
   ```bash
   python3 code/main.py
   ```
   - The processed dataset is saved to `output/all_subjects_processed<DATE>.csv`.
   - Plots such as “Average trajectories” and “Subject X trajectories” are also saved there.

#### run_experiments.py

Runs the preprocessing on many experiments at once. Each experiment is a folder that contains `data/<zip file>` with one CSV per participant:
```bash
python code/run_experiments.py <experiments folder> [--output <output folder>] [--only bltr_confidence sntr_3blcks]
```
Each experiment is saved as `<experiment name>.csv`, in an `output` folder next to the experiments folder by default. The settings for all experiments are in `DEFAULT_SETTINGS`, and experiments that differ get overrides in `EXPERIMENT_SETTINGS`.

If something fails, double-check that the column names match exactly, that your CSV files contain mouse coordinates, and that the required Python packages from `requirements.txt` are installed.

### Output
This folder includes example output files: preprocessed unified data file and visualization results.

