import numpy as np
import pandas as pd

zeros = np.zeros((9, 77))


print(zeros)

features = ['pitch', 'X', 'yaw', 'Y', 'Z', 'kill_distance', 'pitch_delta_at_kill',
       'yaw_delta_at_kill', 'player_speed', 'pitch_velocity', 'yaw_velocity',
       'pitch_acceleration', 'yaw_acceleration', 'pitch_jerk', 'yaw_jerk',
       'snap_magnitude', 'speed_rolling_std', 'speed_rolling_mean',
       'position_delta', 'position_jumpiness', 'cumulative_pitch',
       'cumulative_yaw', 'angle_magnitude', 'yaw_change_sign',
       'pitch_change_sign', 'direction_flips', 'flip_rate', 'yaw_rolling_std',
       'pitch_rolling_std', 'yaw_rolling_mean', 'pitch_rolling_mean',
       'pitch_peaks', 'yaw_peaks', 'pitch_mean', 'pitch_std', 'pitch_min',
       'pitch_max', 'pitch_range', 'pitch_skew', 'pitch_kurtosis', 'yaw_mean',
       'yaw_std', 'yaw_min', 'yaw_max', 'yaw_range', 'yaw_skew',
       'yaw_kurtosis', 'pitch_velocity_mean', 'pitch_velocity_std',
       'pitch_velocity_min', 'pitch_velocity_max', 'pitch_velocity_range',
       'pitch_velocity_skew', 'pitch_velocity_kurtosis', 'yaw_velocity_mean',
       'yaw_velocity_std', 'yaw_velocity_min', 'yaw_velocity_max',
       'yaw_velocity_range', 'yaw_velocity_skew', 'yaw_velocity_kurtosis',
       'angle_magnitude_mean', 'angle_magnitude_std', 'angle_magnitude_min',
       'angle_magnitude_max', 'angle_magnitude_range', 'angle_magnitude_skew',
       'angle_magnitude_kurtosis', 'weapon_grenade', 'weapon_lmg',
       'weapon_melee', 'weapon_pistol', 'weapon_rifle', 'weapon_shotgun',
       'weapon_smg', 'weapon_sniper', 'weapon_unknown']

padding = pd.DataFrame(zeros, columns=features)

print(padding)

X = np.array(padding)

X_flat = X.reshape(-1, X.shape[-1]); print(X_flat)