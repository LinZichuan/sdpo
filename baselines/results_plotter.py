import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # Can change to 'Agg' for non-interactive mode

import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'

from baselines.common import plot_util

X_TIMESTEPS = 'timesteps'
X_EPISODES = 'episodes'
X_WALLTIME = 'walltime_hrs'

Y_REWARD = 'reward'
Y_DISCOUTED_RETURN = 'eprewmean'
Y_RATIO_MAX = 'loss/ratio_max'
Y_RATIO_MIN = 'loss/ratio_min'
Y_TV_MEAN = 'loss/tv_mean'
Y_TIMESTEPS = 'timesteps'
Y_LEFT_CLIPRANGE = 'misc/left_cliprange'
Y_RIGHT_CLIPRANGE = 'misc/right_cliprange'
POSSIBLE_X_AXES = [X_TIMESTEPS, X_EPISODES, X_WALLTIME]
EPISODES_WINDOW = 100
COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'black', 'purple', 'pink',
        'brown', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise',
        'darkgreen', 'tan', 'salmon', 'gold', 'darkred', 'darkblue']

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def window_func(x, y, window, func):
    yw = rolling_window(y, window)
    yw_func = func(yw, axis=-1)
    return x[window-1:], yw_func

def ts2xy_monitor(ts, xaxis, yaxis):
    if xaxis == X_TIMESTEPS:
        x = np.cumsum(ts.l.values)
    elif xaxis == X_EPISODES:
        x = np.arange(len(ts))
    elif xaxis == X_WALLTIME:
        x = ts.t.values / 3600.
    else:
        raise NotImplementedError

    if yaxis == Y_REWARD:
        y = ts.r.values
    elif yaxis == Y_TIMESTEPS:
        y = ts.l.values
    else:
        raise NotImplementedError
    return x, y

def ts2xy_progress(ts, xaxis, yaxis):
    if xaxis == X_TIMESTEPS:
        if 'misc/total_timesteps' in ts:
            x = ts['misc/total_timesteps']
        elif 'TimestepsSoFar' in ts:
            x = ts['TimestepsSoFar']
        else: 
            raise NotImplementedError
    else:
        raise NotImplementedError

    if yaxis == Y_DISCOUTED_RETURN:
        if 'eprewmean' in ts:
            y = ts['eprewmean']
        elif 'EpRewMean' in ts:
            y = ts['EpRewMean']
        else:
            raise NotImplementedError
    elif yaxis == Y_RATIO_MAX:
        if 'loss/ratio_max' in ts:
            y = np.log(ts['loss/ratio_max'])
        elif 'max_ratio' in ts: # trpo logs
            y = np.log(ts['max_ratio'])
        else:
            raise NotImplementedError
    elif yaxis == Y_RATIO_MIN:
        if 'loss/ratio_min' in ts:
            y = np.log(ts['loss/ratio_min'] + 1e-10)
        elif 'min_ratio' in ts: # trpo logs
            y = np.log(ts['min_ratio'] + 1e-10)
        else:
            raise NotImplementedError
    elif yaxis == Y_TV_MEAN:
        y = ts['loss/tv_mean']
    elif yaxis == Y_LEFT_CLIPRANGE:
        y = ts['misc/left_cliprange']
    elif yaxis == Y_RIGHT_CLIPRANGE:
        y = ts['misc/right_cliprange']
    else:
        raise NotImplementedError
    return x, y

def plot_curves(xy_list, xaxis, yaxis, title):
    fig = plt.figure(figsize=(8,2))
    maxx = max(xy[0][-1] for xy in xy_list)
    minx = 0
    for (i, (x, y)) in enumerate(xy_list):
        color = COLORS[i % len(COLORS)]
        plt.scatter(x, y, s=2)
        x, y_mean = window_func(x, y, EPISODES_WINDOW, np.mean) #So returns average of last EPISODE_WINDOW episodes
        plt.plot(x, y_mean, color=color)
    plt.xlim(minx, maxx)
    plt.title(title)
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.tight_layout()
    fig.canvas.mpl_connect('resize_event', lambda event: plt.tight_layout())
    plt.grid(True)

def split_by_task(taskpath):
    return taskpath.dirname.split('/')[-1].split('-')[0]

def plot_cumulative_rewards(results, save_path):
    xaxis = X_TIMESTEPS
    yaxis = Y_REWARD
    plt.clf()
    plot_util.plot_results(results, xy_fn=lambda r: ts2xy_monitor(r.monitor, xaxis, yaxis), \
                           split_fn=split_by_task, average_group=True, shaded_std=False,
                           tiling='horizontal')
    plt.savefig('%s_cumulative_reward.pdf'%save_path)

def plot_discounted_returns(results, save_path):
    xaxis = X_TIMESTEPS
    yaxis = Y_DISCOUTED_RETURN
    plt.clf()
    plot_util.plot_results(results, xy_fn=lambda r: ts2xy_progress(r.progress, xaxis, yaxis), \
                           split_fn=split_by_task, average_group=True, shaded_std=False,
                           tiling='horizontal')
    plt.savefig('%s_discounted_returns.pdf'%save_path)

def plot_ratios_min_max(results, save_path):
    xaxis = X_TIMESTEPS
    yaxis = Y_RATIO_MAX
    plt.clf()
    fig, axarr = plot_util.plot_results(results, \
                           xy_fn=lambda r: ts2xy_progress(r.progress, xaxis, yaxis), \
                           split_fn=split_by_task, average_group=True, shaded_std=False, \
                           tiling='horizontal')
    # overlay
    yaxis = Y_RATIO_MIN
    plot_util.plot_results(results, fig=fig, axarr=axarr, linestyle='dashed', \
                           xy_fn=lambda r: ts2xy_progress(r.progress, xaxis, yaxis), \
                           split_fn=split_by_task, average_group=True, shaded_std=False, \
                           tiling='horizontal')

    plt.savefig('%s_ratios.pdf'%save_path)

def plot_clip_range_left_right(results, save_path):
    xaxis = X_TIMESTEPS
    yaxis = Y_LEFT_CLIPRANGE
    plt.clf()
    fig, axarr = plot_util.plot_results(results, \
                           xy_fn=lambda r: ts2xy_progress(r.progress, xaxis, yaxis), \
                           split_fn=split_by_task, average_group=True, shaded_std=False, \
                           tiling='horizontal')
    # overlay
    yaxis = Y_RIGHT_CLIPRANGE
    plot_util.plot_results(results, fig=fig, axarr=axarr, linestyle='dashed', \
                           xy_fn=lambda r: ts2xy_progress(r.progress, xaxis, yaxis), \
                           split_fn=split_by_task, average_group=True, shaded_std=False, \
                           tiling='horizontal')

    plt.savefig('%s_clip_range.pdf'%save_path)

def plot_tv_mean(results, save_path):
    xaxis = X_TIMESTEPS
    yaxis = Y_TV_MEAN 
    plt.clf()
    plot_util.plot_results(results, xy_fn=lambda r: ts2xy_progress(r.progress, xaxis, yaxis), \
                           split_fn=split_by_task, average_group=True, shaded_std=False,
                           tiling='horizontal')
    plt.savefig('%s_tv_mean.pdf'%save_path)


# Example usage in jupyter-notebook
# from baselines.results_plotter import plot_results
# %matplotlib inline
# plot_results("./log")
# Here ./log is a directory containing the monitor.csv files

def main():
    log_dir = './exp_logs/1214_roboschool_trpo' 

    # env_list = ['Ant-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Humanoid-v2', 'Walker2d-v2']
    # env_list = ['Ant-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Humanoid-v2', 'HumanoidStandup-v2', \
    #             'InvertedDoublePendulum-v2', 'InvertedPendulum-v2', 'Reacher-v2', 'Swimmer-v2', 'Walker2d-v2']

    # env_list = ['BipedalWalker-v2', 'BipedalWalkerHardcore-v2', 'LunarLanderContinuous-v2']
    # env_list = ['HumanoidStandup-v2']
    # env_list = ['dm_humanoid.run-v0', 'dm_humanoid.stand-v0', 'dm_humanoid.walk-v0', 'dm_swimmer.swimmer15-v0']
    # env_list = ['HumanoidBulletEnv-v0', 'HumanoidFlagrunBulletEnv-v0', 'HumanoidFlagrunHarderBulletEnv-v0']
    env_list = ['RoboschoolAnt-v1', 'RoboschoolHalfCheetah-v1', 'RoboschoolHopper-v1', 'RoboschoolHumanoid-v1', 'RoboschoolHumanoidFlagrun-v1',
                'RoboschoolHumanoidFlagrunHarder-v1', 'RoboschoolInvertedDoublePendulum-v1', 'RoboschoolInvertedPendulum-v1', 'RoboschoolInvertedPendulumSwingup-v1',
                'RoboschoolPong-v1', 'RoboschoolReacher-v1', 'RoboschoolWalker2d-v1']


    dirs = [os.path.join(log_dir, env) for env in env_list]
    abs_dirs = [os.path.abspath(dir) for dir in dirs]
    save_path = log_dir

    # load all data
    results = plot_util.load_results(abs_dirs)

    ## plot cumulative rewards in training
    plot_cumulative_rewards(results, save_path)

    ## plot discounted returns in training
    plot_discounted_returns(results, save_path)

    ## plot ratios max
    plot_ratios_min_max(results, save_path)

    ## plot clipping range
    # plot_clip_range_left_right(results, save_path)

    ## plot tv mean estimate
    plot_tv_mean(results, save_path)


if __name__ == '__main__':
    main()