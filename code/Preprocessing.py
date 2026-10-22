import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.signal import find_peaks, savgol_filter
import math


class Preprocessing(object):
    """
    This class gets csv path for a single subject in the experiment,
    handles preprocessing of the mouse-trajectory data, and
    extracts trajectory-based measures.

    Parameters
    ----------
    path : str
        Path to the subject CSV file.
    x_cord_column / y_cord_column : str
        Column names that contain the serialized x/y coordinates.
    response_column : str, optional
        Column that stores additional slider/response data to keep.
    columns_to_preserve : list[str], optional
        Columns that should be preserved even when the row has no trajectory data.
    practice_mode : {'auto', 'manual', 'marker'}
        - 'auto'   → drop practice trials using the `test_part` column (default behaviour)
        - 'manual' → drop practice trials using NUM_PRACTICE_TRIALS/NUM_TRIALS settings
        - 'marker' → drop every row up to and including the screen whose `stimulus` contains `practice_marker`
    num_practice_trials : int
        Number of leading trajectory rows that should be discarded (manual mode only).
    num_trials : int | None
        Number of experimental trajectory rows to keep after discarding practice rows (manual mode only).
    time_column : str, optional
        Column that stores the serialized timestamp (ms) of every x/y sample.
        Needed for the time-based velocity measures.
    practice_marker : str
        Text of the screen that separates practice from the real task (marker mode only).
    """
    normalized_x = 1
    normalized_y = 1.5
    # a trial is "moving" while its speed is at least this fraction of its peak speed
    MOVEMENT_THRESHOLD = 0.05
    # a velocity peak is counted if its prominence is at least this fraction of the trial's peak speed
    PEAK_PROMINENCE = 0.1
    # time-based velocity: samples are resampled every SAMPLE_STEP_MS and smoothed over SMOOTHING_WINDOW points (50 ms)
    SAMPLE_STEP_MS = 10
    SMOOTHING_WINDOW = 5
    TIME_MEASURES = ['recorded_duration', 'initiation_time', 'velocity_peak', 'velocity_time_to_peak',
                     'velocity_peak_relative', 'velocity_num_peaks', 'idle_time', 'peak_acceleration', 'peak_deceleration']
    def __init__(self,path,x_cord_column,y_cord_column, rt_column = "", response_column = "", columns_to_preserve = [],
                 practice_mode='auto', num_practice_trials=0, num_trials=None,
                 time_column = "", practice_marker = "ready to begin the task"):
        self.isOK = True
        self.normalized_x = 1
        self.normalized_y = 1.5
        self.max_length = 0
        self.NUM_TIMEPOINTS = 101
        self.path = path
        self.x_cord_column = x_cord_column
        self.y_cord_column = y_cord_column
        self.rt_column = rt_column
        self.practice_mode = practice_mode.lower() if isinstance(practice_mode, str) else 'auto'
        if self.practice_mode not in ('auto', 'manual', 'marker'):
            self.practice_mode = 'auto'
        self.num_practice_trials = max(0, num_practice_trials or 0)
        self.num_trials = num_trials if num_trials and num_trials > 0 else None
        self.practice_marker = practice_marker
        csv = pd.read_csv (path,index_col=None, header=0)
        #csv = self._drop_invalid_trials(csv)
        original_csv = csv.copy()  # Keep original for add_slider_data method

        csv = self._filter_practice_trials(csv).reset_index(drop=True)

        # In marker mode every response is attached to the trial right before it,
        # so a missing trial or response cannot shift the remaining responses.
        if self.practice_mode == 'marker' and response_column != "":
            csv['explicit_slider'] = self._pair_responses_with_trials(csv, response_column)


        # Smart filtering: keep rows with trajectory data OR rows with data in columns_to_preserve
        if columns_to_preserve:
            # Keep rows that have trajectory data OR have data in any of the preserve columns
            has_trajectory = csv[x_cord_column].notna()
            has_preserve_data = csv[columns_to_preserve].notna().any(axis=1)
            rows_to_keep = has_trajectory | has_preserve_data
            self.df = csv[rows_to_keep].copy()
        else:
            # Default behavior: only keep rows with trajectory data
            self.df = csv.dropna(subset = [x_cord_column]).copy()

        self.df = self.df.reset_index(drop=True)
        
        # Track which rows have trajectory data vs non-trajectory data
        self.trajectory_rows = self.df[x_cord_column].notna().values
        self.num_trajectory_rows = self.trajectory_rows.sum()
        if response_column != "" and self.practice_mode != 'marker': #  if there is response
            self.df['explicit_slider'] = self.add_slider_data(original_csv,response_column)
        
        # Only process x/y coordinates for rows that have trajectory data
        trajectory_df = self.df[self.trajectory_rows]
        self.x = trajectory_df[x_cord_column]
        self.y = trajectory_df[y_cord_column]
        self.x = self.x.str.split(',', expand=True)
        self.x = self.x.reset_index(drop=True)
        self.x = self.x.to_numpy()  # convert to numpy matrix for future processing
        self.x = self.x.astype(np.float64)  # convert type from strings to floats
        self.x = np.transpose(self.x)  # transpose for easier processing (now matrix size is NUM_TIMEPOINTS*NUM_TRAJECTORY_ROWS)

        self.y = self.y.str.split(',', expand=True)
        self.y = self.y.reset_index(drop=True)
        self.y = self.y.to_numpy()
        self.y = self.y.astype(np.float64)
        self.y = np.transpose(self.y)

        # Keep the raw samples (with their timestamps) for the time-based velocity measures,
        # since normalize_time_points() replaces self.x/self.y with 101 index-based points.
        self.x_raw = self.x.copy()
        self.y_raw = self.y.copy()
        self.t_raw = None
        if time_column != "" and time_column in self.df.columns:
            t = trajectory_df[time_column].astype(str).str.split(',', expand=True)
            self.t_raw = np.transpose(t.reset_index(drop=True).replace({'': np.nan, 'nan': np.nan}).to_numpy().astype(np.float64))

    def _filter_practice_trials(self, csv):
        """
        Drop practice trials automatically via the 'test_part' column,
        manually via the NUM_PRACTICE_TRIALS/NUM_TRIALS settings,
        or via the screen that marks the start of the real task.
        """
        if self.practice_mode == 'manual':
            return self._drop_manual_practice_trials(csv)

        if self.practice_mode == 'marker':
            return self._drop_trials_before_marker(csv)

        if 'test_part' not in csv.columns:
            return csv

        return csv[~csv['test_part'].astype(str).str.startswith('practice', na=False)]

    def _drop_invalid_trials(self, csv):
        """
        Remove rows where `trial_num` is negative (applies to all datasets).
        """
        if 'trial_num' not in csv.columns:
            return csv

        trial_nums = pd.to_numeric(csv['trial_num'], errors='coerce')
        mask = (trial_nums.isna()) | (trial_nums >= 0)
        return csv[mask].reset_index(drop=True)

    def _drop_manual_practice_trials(self, csv):
        """
        Remove the leading practice trials (based on the number of valid trajectories)
        and optionally cap the number of remaining experimental trials.
        """
        csv = csv.copy()
        if self.num_practice_trials > 0:
            trajectory_indices = csv[csv[self.x_cord_column].notna()].index
            practice_indices = trajectory_indices[:self.num_practice_trials]
            csv = csv.drop(index=practice_indices)

        if self.num_trials:
            trajectory_indices = csv[csv[self.x_cord_column].notna()].index
            excess_indices = trajectory_indices[self.num_trials:]
            csv = csv.drop(index=excess_indices)

        return csv

    def _drop_trials_before_marker(self, csv):
        """
        Remove every row up to and including the first screen whose `stimulus` contains the practice marker.
        If the marker is missing, nothing is removed and a warning is printed.
        """
        if 'stimulus' not in csv.columns:
            print("WARNING: no 'stimulus' column, practice trials were not removed in", self.path)
            return csv
        marker_rows = csv.index[csv['stimulus'].astype(str).str.contains(self.practice_marker, case=False, regex=False)]
        if len(marker_rows) == 0:
            print("WARNING: practice marker not found, practice trials were not removed in", self.path)
            return csv
        return csv.loc[marker_rows[0] + 1:]

    def _pair_responses_with_trials(self, csv, response_column):
        """
        Return a column aligned with `csv` in which each trajectory row gets the first response
        that follows it (before the next trajectory row), divided by 100 as in add_slider_data.
        Only slider rows are used, so other responses (e.g. button presses) are never paired.
        """
        paired = np.full(csv.shape[0], np.nan)
        if response_column not in csv.columns:
            return paired
        responses = pd.to_numeric(csv[response_column], errors='coerce')
        if 'trial_type' in csv.columns:
            responses = responses.where(csv['trial_type'].astype(str).str.contains('slider', case=False))
        responses = responses.to_numpy()
        has_trajectory = csv[self.x_cord_column].notna().to_numpy()
        current_trial = None
        for i in range(csv.shape[0]):
            if has_trajectory[i]:
                current_trial = i
            elif current_trial is not None and not np.isnan(responses[i]):
                paired[current_trial] = responses[i] / 100
                current_trial = None
        return paired

    def get_coordinate_arrays(self):
        """
        Create full coordinate arrays matching the dataframe shape.
        Trajectory rows get interpolated coordinates, non-trajectory rows get NaN.
        Returns: (x_full, y_full) where each has shape (NUM_TIMEPOINTS, total_rows)
        """
        # Always return arrays matching dataframe shape
        x_full = np.full([self.NUM_TIMEPOINTS, self.df.shape[0]], np.nan)
        y_full = np.full([self.NUM_TIMEPOINTS, self.df.shape[0]], np.nan)

        # If subject is valid and has trajectory data, fill in the coordinates
        if self.isOK and hasattr(self, 'x') and hasattr(self, 'y') and self.x is not None and self.y is not None:
            # Fill in trajectory row coordinates
            trajectory_indices = np.where(self.trajectory_rows)[0]
            if len(trajectory_indices) > 0:
                x_full[:, trajectory_indices] = self.x
                y_full[:, trajectory_indices] = self.y

        return x_full, y_full
    def add_slider_data (self, csv, response_column):
        """
        This function adds the slider information that comes after each trial to the data frame
        while respecting the selected practice-handling mode.
        """
        response_df = csv.dropna(subset=[response_column])

        if self.practice_mode == 'auto':
            if 'test_part' in response_df.columns:
                response_df = response_df[~response_df['test_part'].astype(str).str.startswith('practice', na=False)]
        else:  # manual mode
            responses = response_df[response_column].dropna().tolist()
            if self.num_practice_trials > 0:
                responses = responses[self.num_practice_trials:]
            if self.num_trials:
                responses = responses[:self.num_trials]
            return [x/100 for x in responses]

        responses = response_df[response_column].dropna().tolist()
        return [x/100 for x in responses]

    def normalize_time_points(self):
        """
        This function uses linear interpolation to normalize all of the trajectories to 100 time points from start to finish.
        """
        #innitialize matrix
        xnew = np.empty([self.NUM_TIMEPOINTS,self.num_trajectory_rows])
        ynew = np.empty([self.NUM_TIMEPOINTS,self.num_trajectory_rows])

        for i in range (self.num_trajectory_rows): #run over trajectory trials
            x_for_interp = self.x[:,i]
            x_for_interp = x_for_interp[~np.isnan(x_for_interp)] #remove Nan values in the end of cordinates vectors
            y_for_interp = self.y[:,i]
            y_for_interp = y_for_interp[~np.isnan(y_for_interp)]
            fx = interpolate.interp1d(range(x_for_interp.shape[0]),x_for_interp) #create interpolation function
            fy = interpolate.interp1d(range(y_for_interp.shape[0]),y_for_interp)
            norm_tp_x = np.linspace(0,x_for_interp.shape[0]-1,self.NUM_TIMEPOINTS) #create linear space of 100 time points
            norm_tp_y = np.linspace(0,y_for_interp.shape[0]-1,self.NUM_TIMEPOINTS)
            xnew[:,i] = fx(norm_tp_x) #apply interpolation function on the linear space
            ynew[:,i] = fy(norm_tp_y)

        self.x = xnew
        self.y = ynew

    def rescale(self):
        """
        This function rescales the coordiantes space so that the continue button is in [0,0]
        and the right and left targets are where normalized_x (plus and minus) and normalized_y are.
        default and recommended is - [1,1.5] and [-1,1.5] respectively
        """

        if self.num_trajectory_rows == 0:
            print("WARNING: subject has no trials, marked as is_OK=False:", self.path)
            self.isOK = False
            return
        self.continue_x = np.mean(self.x[0,:])#calculating the x cordinates for the continue button
        left_subset = self.x[:,self.x[self.NUM_TIMEPOINTS-1,:]<self.continue_x] #subsetting all trials where the left option is chosen
        right_subset = self.x[:,self.x[self.NUM_TIMEPOINTS-1,:]>self.continue_x]
        if left_subset.shape[1] == 0 or right_subset.shape[1] == 0:
            # the target positions cannot be estimated without choices on both sides
            print("WARNING: subject never chose one of the sides, marked as is_OK=False:", self.path)
            self.isOK = False
            return
        self.right_x, self.left_x = np.mean(right_subset[-1,:]),np.mean(left_subset[-1,:])#calculating the x cordinates for the choices
        self.x -= self.continue_x #center x coordinates
        self.x  = (self.x / ((self.right_x-self.left_x)/2))*self.normalized_x #rescale x coordinates
        self.continue_y =  np.mean(self.y[0,:])
        self.targets_y = np.mean(self.y[self.NUM_TIMEPOINTS-1,:])
        self.y -=self.continue_y
        self.y = (-self.y / ((self.continue_y-self.targets_y)))*self.normalized_y

    def remap_trajectories(self):
        """
        This function remaps all of the trajectories to the right.
        IMPORTANT: The function will only work if you apply the rescale() function first!
        """
        for i in range(self.num_trajectory_rows):
            if self.x[self.NUM_TIMEPOINTS-1,i] < 0: #i.e., if the trajectory ends in the left target
                self.x[:,i] = -self.x[:,i]

    def get_x_flips(self):
        """
        This function calculates number of x flips trial by trial and saves it as a variable 'flips' of the class.
        """
        self.flips = []
        for trial in range(self.num_trajectory_rows):
            bigger, smaller = False, False
            x_count = 0
            last_i = 0
            cur_trial_xcor = []
            for i in range(1,self.NUM_TIMEPOINTS):
                if self.x[last_i,trial] > self.x[i,trial]:
                    last_i += 1
                    if not bigger:
                        bigger = True
                        smaller = False
                        if i != 1:
                            x_count += 1
                            cur_trial_xcor.append (self.x[last_i,trial])
                    else:
                        continue
                if self.x[last_i,trial] < self.x[i,trial]:
                    last_i += 1
                    if not smaller:
                        bigger = False
                        smaller = True
                        if i != 1:
                            x_count += 1
                            cur_trial_xcor.append (self.x[last_i,trial])
                    else:
                        continue
            self.flips.append(x_count)



    def get_RPB(self):
        """
        This funcction calculates the number of times the mouse cursor cross the middle of the X-axis
        """
        self.RPB = []
        for trial in range(self.num_trajectory_rows):
            crosses = 0
            crosses_cut = 0
            last_i = 0
            cur_crosses_ycor = []
            for i in range(1,self.NUM_TIMEPOINTS):
                if (self.x[last_i, trial] > 0 > self.x[i, trial]) or (self.x[last_i, trial] < 0 < self.x[i, trial]):
                    crosses += 1
                last_i += 1
            self.RPB.append(crosses)

    def get_AUC(self):
        """
        This function calculates all area under the curve for the trajectories
        """
        self.AUC = []
        for i in range(self.num_trajectory_rows):
            cur_x = self.x[:, i]
            cur_y = self.y[:, i]
            line_x, line_y = self.normalized_x*cur_x, self.normalized_y *cur_x  # create a straight line from (0,0) to the normalized location (default and recommended - 1,1.5).
            # integrate on the distance between the actual trajectory and a straight line
            cur_integral = np.trapz(cur_y-line_y, dx=1/self.NUM_TIMEPOINTS)
            self.AUC.append(cur_integral)

            #possibility - uncomment this section to visualize the integrated part and compare it to the calculated MD and AUC

            #print('AUC='+str(cur_integral))
            #print('MD=' +str(self.max_deviations[i]))
            #plt.plot(cur_x,cur_y,color = 'r')
            #plt.plot(line_x,line_y)
            #plt.fill_between(cur_x,cur_y,line_y,alpha = 0.4)
            #plt.ylim(0,1.7)
            #plt.show()

    def get_max_deviation(self):
        """
        This function calculates the maximal deviation from the actual trajectory to a
        straight line connecting the starting position and the target.
        IMPORTANT: this function will only work if you apply the rescale and remap functions first.
        """
        self.max_deviations = []
        for i in range(self.num_trajectory_rows):
            line_x, line_y = (np.linspace(self.x[0, i], self.x[self.NUM_TIMEPOINTS-1, i], self.NUM_TIMEPOINTS),
                              np.linspace(self.y[0, i], self.y[self.NUM_TIMEPOINTS-1, i], self.NUM_TIMEPOINTS))
            self.line = np.column_stack((line_x, line_y))
            distances = []
            for j in range(self.NUM_TIMEPOINTS):
                cur_point = np.array([self.x[j, i], self.y[j, i]])
                norm = np.linalg.norm(self.line-cur_point, axis=1)
                cur_distance = min(norm)
                distances.append(cur_distance)
            self.max_deviations.append(max(distances))

    def get_initiation_angle(self):
        """
        This function calculates the angle of the starting movement of the mouse trajectory relative to the y-axis
        """
        self.initiation_angle = []
        self.initiation_correspondence = []
        point_of_angle = 10
        for trial in range(self.num_trajectory_rows):
            # calculate angle of the beginning of trajectory from the vector [1,0](x-axis).
            # see https://stackoverflow.com/questions/42258637/how-to-know-the-angle-between-two-vectors
            angle = math.atan2(self.y[0, trial]-self.y[point_of_angle, trial],
                               self.x[point_of_angle, trial]-self.x[0, trial])
            angle = math.degrees(angle)
            angle = abs(angle)
            self.initiation_angle.append(angle)
            self.initiation_correspondence.append(angle < 90)

            # # uncomment the next lines in order to plot and check the measure validity.
            # plt.plot(self.x[:,trial],self.y[:,trial],'--o')
            # plt.plot([self.x[0,trial],self.x[point_of_angle,trial]],[self.y[0,trial],self.y[point_of_angle,trial]])
            # plt.title(angle)
            # plt.show()

    def get_kinematics(self):
        self.mean_velocity = []
        self.max_velocity = []
        self.mean_acceleration = []
        self.velocities_full = []
        self.accelerations_full = []

        has_rt = self.rt_column != "" and self.rt_column in self.df.columns
        trajectory_indices = np.where(self.trajectory_rows)[0]

        for trial in range(self.num_trajectory_rows):
            if not has_rt:
                self.mean_velocity.append(np.nan)
                self.max_velocity.append(np.nan)
                self.mean_acceleration.append(np.nan)
                self.velocities_full.append(np.full(self.NUM_TIMEPOINTS, np.nan))
                self.accelerations_full.append(np.full(self.NUM_TIMEPOINTS, np.nan))
                continue

            original_row_idx = trajectory_indices[trial]
            rt_val = self.df.loc[original_row_idx, self.rt_column]

            if pd.isna(rt_val) or float(rt_val) <= 0:
                self.mean_velocity.append(np.nan)
                self.max_velocity.append(np.nan)
                self.mean_acceleration.append(np.nan)
                self.velocities_full.append(np.full(self.NUM_TIMEPOINTS, np.nan))
                self.accelerations_full.append(np.full(self.NUM_TIMEPOINTS, np.nan))
                continue

            rt = float(rt_val)
            x_array = self.x[:, trial]
            y_array = self.y[:, trial]

            x_diff = np.diff(x_array)
            y_diff = np.diff(y_array)
            distances = np.sqrt(x_diff ** 2 + y_diff ** 2)

            delta_t = rt / (self.NUM_TIMEPOINTS - 1)

            velocities = distances / delta_t
            accelerations = np.diff(velocities) / delta_t

            self.mean_velocity.append(np.mean(velocities))
            self.max_velocity.append(np.max(velocities))
            self.mean_acceleration.append(np.mean(accelerations))

            padded_vel = np.full(self.NUM_TIMEPOINTS, np.nan)
            padded_vel[1:] = velocities
            self.velocities_full.append(padded_vel)

            padded_acc = np.full(self.NUM_TIMEPOINTS, np.nan)
            padded_acc[2:] = accelerations
            self.accelerations_full.append(padded_acc)

    def get_time_kinematics(self):
        """
        This function calculates velocity measures from the real sample timestamps (time_column).
        Each trial is resampled every SAMPLE_STEP_MS in real time and velocity is calculated with a
        Savitzky-Golay filter. The velocity profile is then stored as NUM_TIMEPOINTS equally spaced points
        in real time. Positions are in the rescaled space, velocity is in units/s,
        acceleration in units/s^2 and times in ms.
        IMPORTANT: this function will only work if you apply the rescale function first.
        """
        names = self.TIME_MEASURES
        self.time_measures = {name: [] for name in names}
        self.time_velocities_full = []

        has_rt = self.rt_column != "" and self.rt_column in self.df.columns
        trajectory_indices = np.where(self.trajectory_rows)[0]

        for trial in range(self.num_trajectory_rows):
            values = self._time_kinematics_of_trial(trial, trajectory_indices[trial], has_rt)
            if values is None:
                for name in names:
                    self.time_measures[name].append(np.nan)
                self.time_velocities_full.append(np.full(self.NUM_TIMEPOINTS, np.nan))
                continue
            measures, velocities = values
            for name in names:
                self.time_measures[name].append(measures[name])
            self.time_velocities_full.append(velocities)

    def _time_kinematics_of_trial(self, trial, original_row_idx, has_rt):
        """
        Returns (measures dict, velocity profile) for one trial, or None if the timestamps are unusable.
        """
        if self.t_raw is None:
            return None
        t = self.t_raw[:, trial]
        x = self.x_raw[:, trial]
        y = self.y_raw[:, trial]
        t, x, y = t[~np.isnan(t)], x[~np.isnan(x)], y[~np.isnan(y)]
        # every sample needs its own timestamp
        if len(t) != len(x) or len(x) != len(y) or len(x) < 2:
            return None
        if np.any(np.diff(t) < 0) or t[-1] <= t[0]:
            return None

        # same linear transform as rescale(); remapping does not change speed
        x = ((x - self.continue_x) / ((self.right_x - self.left_x) / 2)) * self.normalized_x
        y = (-(y - self.continue_y) / (self.continue_y - self.targets_y)) * self.normalized_y

        # Resample to a fixed time step and differentiate with a Savitzky-Golay filter: the browser sometimes
        # records two samples 1-4 ms apart, and plain differences turn that timing jitter into velocity spikes.
        t = t - t[0]
        grid = np.arange(0, t[-1] + 1e-9, self.SAMPLE_STEP_MS)
        if len(grid) < 3:
            return None
        window = min(self.SMOOTHING_WINDOW, len(grid) if len(grid) % 2 == 1 else len(grid) - 1)
        x_grid = np.interp(grid, t, x)
        y_grid = np.interp(grid, t, y)
        vx = savgol_filter(x_grid, window, 2, deriv=1, delta=self.SAMPLE_STEP_MS)
        vy = savgol_filter(y_grid, window, 2, deriv=1, delta=self.SAMPLE_STEP_MS)
        velocities = np.hypot(vx, vy) * 1000  # units/ms -> units/s
        accelerations = savgol_filter(velocities, window, 2, deriv=1, delta=self.SAMPLE_STEP_MS) * 1000
        peak = np.max(velocities)
        if not np.isfinite(peak) or peak <= 0:
            return None

        moving = np.where(velocities >= self.MOVEMENT_THRESHOLD * peak)[0]
        onset, offset = moving[0], moving[-1]
        peak_idx = int(np.argmax(velocities))
        step = grid[1] - grid[0]
        idle_points = np.count_nonzero(velocities[onset:offset + 1] < self.MOVEMENT_THRESHOLD * peak)

        # time between the start of the trial and the first recorded sample (recording starts when the mouse moves)
        pre_recording = np.nan
        if has_rt:
            rt_val = self.df.loc[original_row_idx, self.rt_column]
            if not pd.isna(rt_val) and float(rt_val) >= t[-1]:
                pre_recording = float(rt_val) - t[-1]

        measures = {
            'recorded_duration': t[-1],
            'initiation_time': pre_recording + grid[onset],
            'velocity_peak': peak,
            'velocity_time_to_peak': grid[peak_idx] - grid[onset],
            'velocity_peak_relative': (grid[peak_idx] - grid[onset]) / (grid[offset] - grid[onset]) if offset > onset else np.nan,
            'velocity_num_peaks': len(find_peaks(velocities, prominence=self.PEAK_PROMINENCE * peak)[0]),
            'idle_time': idle_points * step,
            'peak_acceleration': np.max(accelerations),
            'peak_deceleration': -np.min(accelerations),
        }
        # profile of NUM_TIMEPOINTS points, equally spaced in real time from the first to the last sample
        profile = np.interp(np.linspace(0, grid[-1], self.NUM_TIMEPOINTS), grid, velocities)
        return measures, profile

    def get_time_kinematics_arrays(self):
        tv_full = np.full([self.NUM_TIMEPOINTS, self.df.shape[0]], np.nan)
        if self.isOK and hasattr(self, 'time_velocities_full') and len(self.time_velocities_full) > 0:
            trajectory_indices = np.where(self.trajectory_rows)[0]
            if len(trajectory_indices) > 0:
                tv_full[:, trajectory_indices] = np.array(self.time_velocities_full).T
        return tv_full

    def get_kinematics_arrays(self):
        v_full = np.full([self.NUM_TIMEPOINTS, self.df.shape[0]], np.nan)
        a_full = np.full([self.NUM_TIMEPOINTS, self.df.shape[0]], np.nan)
        if self.isOK and hasattr(self, 'velocities_full') and len(self.velocities_full) > 0:
            trajectory_indices = np.where(self.trajectory_rows)[0]
            if len(trajectory_indices) > 0:
                v_full[:, trajectory_indices] = np.array(self.velocities_full).T
                a_full[:, trajectory_indices] = np.array(self.accelerations_full).T
        return v_full, a_full

    def measure_trajectory_length(self):
      """
        This function measures the length of the trajectory course for each trial.
        note that it essential to rescale the data before using this function!
      """
      self.length = []
      for j in range (self.num_trajectory_rows):
        length = 0
        #every row is in size of the longest row where all the extra cells are nan
        for i in range(len (self.x[:,j])-1):
            #finish when the values become nan
         #   if (np.isnan(self.x[i + 1,j])):
         #     continue
            length += math.sqrt((self.x[i, j] - self.x[i + 1,j]) ** 2 + (self.y[i, j] - self.y[i+1, j]) ** 2)
        self.length.append(length)
    def calculate_all_measures(self):
        """
        This function calculates the measures (e,g. x flips, max deviation...)
        and saves each measure as a column in the data frame.
        For trajectory rows: calculates all measures
        For non-trajectory rows: fills with NaN
        """
        self.df['is_OK'] = self.isOK
        if self.isOK:
            # Calculate measures for trajectory rows only
            self.get_x_flips()
            self.get_max_deviation()
            self.get_RPB()
            self.get_AUC()
            self.get_initiation_angle()
            self.measure_trajectory_length()
            self.get_kinematics()
            self.get_time_kinematics()

            # Calculate the real minimum trajectory length for each trajectory trial
            real_min_lengths = []
            for i in range(self.num_trajectory_rows):
                start_x = self.x[0, i]
                start_y = self.y[0, i]
                end_x = self.x[self.NUM_TIMEPOINTS-1, i]
                end_y = self.y[self.NUM_TIMEPOINTS-1, i]
                real_min_length = math.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
                real_min_lengths.append(real_min_length)
            
            # Create arrays for all rows (trajectory + non-trajectory), filled with NaN
            flips_all = np.full(self.df.shape[0], np.nan)
            max_dev_all = np.full(self.df.shape[0], np.nan)
            rpb_all = np.full(self.df.shape[0], np.nan)
            auc_all = np.full(self.df.shape[0], np.nan)
            init_angle_all = np.full(self.df.shape[0], np.nan)
            init_corr_all = np.full(self.df.shape[0], np.nan)
            traj_length_all = np.full(self.df.shape[0], np.nan)
            real_min_all = np.full(self.df.shape[0], np.nan)
            mean_vel_all = np.full(self.df.shape[0], np.nan)
            max_vel_all = np.full(self.df.shape[0], np.nan)
            mean_accel_all = np.full(self.df.shape[0], np.nan)
            
            # Fill in values only for trajectory rows
            trajectory_indices = np.where(self.trajectory_rows)[0]
            flips_all[trajectory_indices] = self.flips
            max_dev_all[trajectory_indices] = self.max_deviations
            rpb_all[trajectory_indices] = self.RPB
            auc_all[trajectory_indices] = self.AUC
            init_angle_all[trajectory_indices] = self.initiation_angle
            init_corr_all[trajectory_indices] = self.initiation_correspondence
            traj_length_all[trajectory_indices] = self.length
            real_min_all[trajectory_indices] = real_min_lengths
            mean_vel_all[trajectory_indices] = self.mean_velocity
            max_vel_all[trajectory_indices] = self.max_velocity
            mean_accel_all[trajectory_indices] = self.mean_acceleration
            
            # Assign to dataframe
            self.df['flips'] = flips_all
            self.df['max_deviation'] = max_dev_all
            self.df['RPB'] = rpb_all
            self.df['AUC'] = auc_all
            self.df['initiation_angle'] = init_angle_all
            self.df['initiation_correspondence'] = init_corr_all
            self.df['trajectory_length'] = traj_length_all
            self.df['real_min_length'] = real_min_all
            self.df['mean_velocity'] = mean_vel_all
            self.df['max_velocity'] = max_vel_all
            self.df['mean_acceleration'] = mean_accel_all

            for name, values in self.time_measures.items():
                column = np.full(self.df.shape[0], np.nan)
                column[trajectory_indices] = values
                self.df[name] = column

        else:  # if data is not ok, fill all columns with nan
            self.df['flips'] = (np.full([self.df.shape[0]],np.nan))
            self.df['max_deviation'] = (np.full([self.df.shape[0]],np.nan))
            self.df['RPB'] = (np.full([self.df.shape[0]],np.nan))
            self.df['AUC'] = (np.full([self.df.shape[0]],np.nan))
            self.df['initiation_angle'] = (np.full([self.df.shape[0]],np.nan))
            self.df['initiation_correspondence'] = (np.full([self.df.shape[0]],np.nan))
            self.df['trajectory_length'] = (np.full([self.df.shape[0]],np.nan))
            self.df['real_min_length'] = (np.full([self.df.shape[0]],np.nan))
            self.df['mean_velocity'] = np.full([self.df.shape[0]], np.nan)
            self.df['max_velocity'] = np.full([self.df.shape[0]], np.nan)
            self.df['mean_acceleration'] = np.full([self.df.shape[0]], np.nan)
            for name in self.TIME_MEASURES:
                self.df[name] = np.full([self.df.shape[0]], np.nan)

# exmp = Preprocessing(r'C:\Users\mhavi\shalevproject\mouse-tracking-tools-master\data\5a0c4184fe645f0001e9f5dd.csv',3,120,"x_cord","y_cord")
# exmp.normalize_time_points()
# exmp.rescale()
# exmp.remap_trajectories()
# exmp.plot_by_condition()
