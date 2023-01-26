import os
import time
import numpy as np
import os.path as osp
from baselines import logger
from collections import deque
from baselines.common import explained_variance, set_global_seeds
from baselines.common.policies import build_policy
try:
    from mpi4py import MPI
    NWORKERS = MPI.COMM_WORLD.Get_size()
except ImportError:
    MPI = None
    NWORKERS = 1
from baselines.ppo2.runner import Runner

def constfn(val):
    def f(_):
        return val
    return f

def allreduce(x, op_type='reduce_mean'):
    op_map = {'reduce_mean': MPI.SUM, 'reduce_min': MPI.MIN, 'reduce_max': MPI.MAX}
    assert op_type in op_map
    reduce_op = op_map[op_type]
    
    assert isinstance(x, np.ndarray)
    if MPI is not None:
        out = np.empty_like(x)
        MPI.COMM_WORLD.Allreduce(x, out, op=reduce_op)
        if op_type == 'reduce_mean':
            out /= NWORKERS
    else:
        out = np.copy(x)
    return out

def learn(*, network, env, total_timesteps, eval_env = None, seed=None, nsteps=2048, ent_coef=0.0, lr=3e-4,
            vf_coef=0.5, adv_clip_range=None, max_grad_norm=0.5, gamma=0.99, lam=0.95, adaptive_clipping=False, inverse_range=False,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,left_cliprange=0.8, right_cliprange=1.2,
            save_interval=0, load_path=None, model_type='clip', tv_threshold=0.3, target_kl=0.01, reduce_type='reduce_mean',
            reg_coef=1.0, margin_size=0.1,
            reg_type='average', update_fn=None, init_fn=None, mpi_rank_weight=1, use_cd=False, use_3stage=False, sd_delta=0.25, sd_value=False, no_abs=False, rminus=False, unbias=False, lclip=False, **network_kwargs):

    set_global_seeds(seed)

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    if isinstance(left_cliprange, float): left_cliprange = constfn(left_cliprange)
    else: assert callable(left_cliprange)

    if isinstance(right_cliprange, float): right_cliprange = constfn(right_cliprange)
    else: assert callable(right_cliprange)

    total_timesteps = int(total_timesteps)

    policy = build_policy(env, network, **network_kwargs)

    # Get the nb of env
    nenvs = env.num_envs

    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space

    drop = False
    noclip = False

    use_FB_estimate = False
    cond1 = cond2 = False
    sg = False
    if use_cd: cond2 = True
    if use_cd:
        algo_type = 'full_batch_update'  # sampling_active / full_batch_update / none
    else:
        algo_type = 'none'  # sampling_active / full_batch_update / none

    if model_type == 'clip':
        nminibatches = 4

    # Calculate the batch_size
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches
    is_mpi_root = (MPI is None or MPI.COMM_WORLD.Get_rank() == 0)

    # Instantiate the model object (that creates act_model and train_model)
    if model_type == 'clip':
        from baselines.ppo2.model import Model
        model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, 
                      nbatch_train=None,
                      nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef, adv_clip_range=adv_clip_range,
                      max_grad_norm=max_grad_norm, mpi_rank_weight=mpi_rank_weight, cond1=cond1, cond2=cond2, use_3stage=use_3stage, drop=drop, noclip=noclip, sd_delta=sd_delta, sd_value=sd_value)
    elif model_type == 'dropout':
        from baselines.ppo2.model import DropoutModel
        model = DropoutModel(policy=policy, nbatch_act=nenvs, 
                      nbatch_train=None,
                      nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                      max_grad_norm=max_grad_norm, mpi_rank_weight=mpi_rank_weight, cond1=cond1, cond2=cond2, sg=sg, use_3stage=use_3stage, sd_delta=sd_delta, sd_value=sd_value, no_abs=no_abs, rminus=rminus, unbias=unbias, lclip=lclip)
    elif model_type == 'reg':
        from baselines.ppo2.model import RegularizedModel
        model = RegularizedModel(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, 
                                 nbatch_train=None,
                                 nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                                 max_grad_norm=max_grad_norm, mpi_rank_weight=mpi_rank_weight,
                                 reg_coef=reg_coef, margin_size=margin_size, reg_type=reg_type)
    else:
        raise NotImplementedError

    if load_path is not None:
        model.load(load_path)
    # Instantiate the runner object
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)
    if eval_env is not None:
        eval_runner = Runner(env = eval_env, model = model, nsteps = nsteps, gamma = gamma, lam= lam)

    epinfobuf = deque(maxlen=100)
    if eval_env is not None:
        eval_epinfobuf = deque(maxlen=100)

    if init_fn is not None:
        init_fn()

    # Start total timer
    tfirststart = time.perf_counter()

    nupdates = total_timesteps//nbatch
    for update in range(1, nupdates+1):
        if inverse_range:
            left_cliprangenow = 1.0 / right_cliprangenow

        assert nbatch % nminibatches == 0
        # Start timer
        tstart = time.perf_counter()

        frac = 1.0 - (update - 1.0) / nupdates
        # Calculate the learning rate
        lrnow = lr(frac)

        # Calculate the cliprange
        cliprangenow = cliprange(frac)

        # Calculate the cliprange
        left_cliprangenow = left_cliprange(frac)
        right_cliprangenow = right_cliprange(frac)

        if update % log_interval == 0 and is_mpi_root: logger.info('Stepping environment...')

        # Get minibatch
        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run() #pylint: disable=E0632
        if eval_env is not None:
            eval_obs, eval_returns, eval_masks, eval_actions, eval_values, eval_neglogpacs, eval_states, eval_epinfos = eval_runner.run() #pylint: disable=E0632

        if update % log_interval == 0 and is_mpi_root: logger.info('Done.')

        epinfobuf.extend(epinfos)
        if eval_env is not None:
            eval_epinfobuf.extend(eval_epinfos)

        # Here what we're going to do is for each minibatch calculate the loss and append it.
        mblossvals = []
        mbratio_max, mbratio_min = [], []
        mbratio_mean = []
        cond2sum_list = []

        dropout = False
        # if use_FB_estimate:
        #     has_break = True
        #     algo_type = 'none'

        assert (algo_type in ['sampling_active', 'full_batch_update', 'none'])
        if algo_type == 'sampling_active':
            sampling_active = True  # use mini-batch (64) to update
            full_batch_update = False  # use full-batch (2048) to update
        elif algo_type == 'full_batch_update':
            sampling_active = False  # use mini-batch (64) to update
            full_batch_update = True  # use full-batch (2048) to update
        else:
            sampling_active = False  # use mini-batch (64) to update
            full_batch_update = False  # use full-batch (2048) to update
        inds = np.arange(nbatch)
        noptepochs = 10
        update_times = 0
        for e in range(noptepochs):
            # Randomize the indexes
            np.random.shuffle(inds)
            break_from_curr_epoch = False
            # 0 to batch_size with batch_train_size step
            for start in range(0, nbatch, nbatch_train):
                end = start + nbatch_train
                mbinds = inds[start:end]
                fullinds = inds
                # if sampling_active:
                #     # full batch estimate
                #     full_slices = (arr[fullinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                #     full_batch_estimate_deviations, condition2 = model.full_batch_estimate(lrnow, cliprangenow,
                #                                                                            *full_slices)
                #     # if np.sum(condition2) < nbatch_train:
                #     if full_batch_estimate_deviations > 0.25:
                #         break_from_curr_epoch = True
                #         print("Break due to full_batch_estimate_deviations > 0.25, full_batch_estimate_deviations:",
                #               full_batch_estimate_deviations)
                #         print("noptepochs:", e, "update_times:", update_times)
                #         break
                #     active_inds = np.where(condition2 == 1)
                #     np.random.shuffle(active_inds)
                #     active_inds_mbinds = active_inds[:nbatch_train]
                #     slices = (arr[active_inds_mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                if use_cd and full_batch_update and model_type == 'dropout':
                    # full batch estimate
                    # full_slices = (arr[fullinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    # full_batch_estimate_deviations, condition2 = model.full_batch_estimate(lrnow, cliprangenow,
                    #                                                                        *full_slices)
                    # # if np.sum(condition2) < nbatch_train:
                    # if full_batch_estimate_deviations > 0.25:
                    #     break_from_curr_epoch = True
                    #     print("Break due to full_batch_estimate_deviations > 0.25, full_batch_estimate_deviations:",
                    #           full_batch_estimate_deviations)
                    #     print("noptepochs:", e, "update_times:", update_times)
                    #     break
                    slices = (arr[fullinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                else:
                    slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                update_times += 1
                #slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                if model_type == 'clip':
                    lossval, cond2sum, ratio_max, ratio_min, ratio_mean = model.train(lrnow, left_cliprangenow, right_cliprangenow, *slices)
                elif model_type == 'reg':
                    lossval = model.train(lrnow, *slices)
                elif model_type == 'dropout':
                    lossval, curr_tv_mean, curr_adv_mean, cond2sum, ratio_max, ratio_min, ratio_mean = model.train(lrnow, *slices)
                    curr_tv_mean = allreduce(np.array([curr_tv_mean]), op_type=reduce_type)[0]
                    if curr_tv_mean >= tv_threshold:
                        logger.info('TV estimate: ', curr_tv_mean)
                        logger.info('Stop updating: %d out of %d'%(e, noptepochs))
                        dropout = True
                        break

                mblossvals.append(lossval[:-2]) # [..., ratio_max, ratio_min]
                # mbratio_max.append(lossval[-2])
                # mbratio_min.append(lossval[-1])
                mbratio_max.append(ratio_max)
                mbratio_min.append(ratio_min)
                mbratio_mean.append(ratio_mean)
                cond2sum_list.append(cond2sum)

            if dropout: # stop updating
                break
            if break_from_curr_epoch:
                print("Break from curr epoch!")
                break
        # Feedforward --> get losses --> update
        lossvals = np.mean(mblossvals, axis=0).tolist()
        ratio_max_now = np.max(mbratio_max)
        ratio_min_now = np.min(mbratio_min)
        ratio_mean_now = np.mean(mbratio_mean)
        lossvals += [ratio_max_now, ratio_min_now, ratio_mean_now]

        # End timer
        tnow = time.perf_counter()
        # Calculate the fps (frame per second)
        fps = int(nbatch / (tnow - tstart))

        if update_fn is not None:
            update_fn(update)

        if update % log_interval == 0 or update == 1:
            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            ev = explained_variance(values, returns)
            logger.logkv("misc/serial_timesteps", update*nsteps)
            logger.logkv("misc/lr", lrnow)
            logger.logkv("misc/nupdates", update)
            logger.logkv("misc/cond2sum", np.mean(cond2sum_list))
            logger.logkv("misc/total_timesteps", update*nbatch)
            logger.logkv("fps", fps)
            logger.logkv("misc/explained_variance", float(ev))
            if model_type == 'clip' or model_type == 'kl_clip':
                logger.logkv('misc/left_cliprange', left_cliprangenow)
                logger.logkv('misc/right_cliprange', right_cliprangenow)
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eprewmax', safemax([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eprewmin', safemin([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.logkv('eplenmax', safemax([epinfo['l'] for epinfo in epinfobuf]))
            logger.logkv('eplenmin', safemin([epinfo['l'] for epinfo in epinfobuf]))
            if eval_env is not None:
                logger.logkv('eval_eprewmean', safemean([epinfo['r'] for epinfo in eval_epinfobuf]) )
                logger.logkv('eval_eplenmean', safemean([epinfo['l'] for epinfo in eval_epinfobuf]) )
            logger.logkv('misc/time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv('loss/' + lossname, lossval)

            logger.dumpkvs()

        ## adaptive clipping
        if model_type == 'clip' or model_type == 'kl_clip':
            if adaptive_clipping:
                if ratio_max_now > 2.0 and right_cliprangenow > 0.0:
                    right_cliprangenow -= 0.01
                elif ratio_max_now < 1.4 and right_cliprangenow < 1.0:
                    right_cliprangenow += 0.01

                # inversing the left_cliprange is disabled
                if not inverse_range:
                    if ratio_min_now < 0.4 and left_cliprangenow > 0.0:
                        left_cliprangenow -= 0.01
                    elif ratio_min_now > 0.8 and left_cliprangenow < 1.0:
                        left_cliprangenow += 0.01

        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir() and is_mpi_root:
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i'%update)
            print('Saving to', savepath)
            model.save(savepath)

    return model

# Avoid division error when calculate the mean (in our case if epinfo is empty returns np.nan, not return an error)
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)
def safemax(xs):
    return np.nan if len(xs) == 0 else np.max(xs)
def safemin(xs):
    return np.nan if len(xs) == 0 else np.min(xs)
