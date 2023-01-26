import tensorflow as tf
import functools

from baselines.common.tf_util import get_session, save_variables, load_variables
from baselines.common.tf_util import initialize

from baselines.common.mpi_adam_optimizer import MpiAdamOptimizer
from mpi4py import MPI
from baselines.common.mpi_util import sync_from_root
import numpy as np

def flat_concat(xs):
    return tf.concat([tf.reshape(x,(-1,)) for x in xs], axis=0)

class Model(object):
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm, mpi_rank_weight=1,  
                microbatch_size=None, adv_clip_range=None, cond1=False, cond2=False, use_3stage=False, drop=False, noclip=False, sd_delta=1.0, sd_value=False):
        self.sess = sess = get_session()

        self.sd_value = sd_value

        if MPI is not None:
            comm = MPI.COMM_WORLD

        with tf.variable_scope('ppo2_model', reuse=tf.AUTO_REUSE):
            # CREATE OUR TWO MODELS
            # act_model that is used for sampling
            act_model = policy(nbatch_act, 1, sess)

            # Train model for training
            if microbatch_size is None:
                train_model = policy(nbatch_train, nsteps, sess)
            else:
                train_model = policy(microbatch_size, nsteps, sess)

        # CREATE THE PLACEHOLDERS
        self.A = A = train_model.pdtype.sample_placeholder([None])
        self.ADV = ADV = tf.placeholder(tf.float32, [None])
        self.R = R = tf.placeholder(tf.float32, [None])
        # Keep track of old actor
        self.OLDNEGLOGPAC = OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        # Keep track of old critic
        self.OLDVPRED = OLDVPRED = tf.placeholder(tf.float32, [None])
        self.LR = LR = tf.placeholder(tf.float32, [])
        # Cliprange
        self.LEFT_CLIPRANGE = LEFT_CLIPRANGE = tf.placeholder(tf.float32, [])
        self.RIGHT_CLIPRANGE = RIGHT_CLIPRANGE = tf.placeholder(tf.float32, [])
        self.use_3stage = use_3stage
        # if self.use_3stage:
        self.condition2_ph = tf.placeholder(tf.float32, [None])
        self.use_cd = cond2

        neglogpac = train_model.pd.neglogp(A)

        # Calculate the entropy
        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # CALCULATE THE LOSS
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Clip the value to reduce variability during Critic training
        # Get the predicted value
        vpred = train_model.vf


        # Calculate ratio (pi current policy / pi old policy)
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)

        CLIPRANGE = 0.2

        # cilp太大的
        # delta & ep
        # 1.0   & 0.3, 0.4, 0.5
        # 0.5   & 0.3, 0.4
        DELTA2 = sd_delta  # 1.0, 0.5
        new_epsilon = 0.2  # 0.3, 0.4, 0.5
        condition = tf.cast(tf.less(tf.abs(ratio - 1.0), DELTA2), tf.float32)
        a = tf.cast(tf.equal(tf.reduce_mean(condition), 0), tf.float32)
        condition2 = a * tf.ones_like(condition) + (1 - a) * condition
        print("condition:", tf.reduce_mean(condition2))
        self.condition2_sum = tf.reduce_sum(condition2)
        self.condition2 = condition2

        use_condition2 = cond2
        print('use_condition2:', use_condition2)

        if use_condition2:
            self.ratio_max = ratio_max = tf.reduce_max(ratio * self.condition2_ph)
            self.ratio_min = ratio_min = tf.reduce_min(ratio * self.condition2_ph + 10.0 * (1.0-self.condition2_ph))
            self.ratio_mean = ratio_mean = tf.reduce_mean(ratio)
        else:
            self.ratio_max = ratio_max = tf.reduce_max(ratio)
            self.ratio_min = ratio_min = tf.reduce_min(ratio)
            self.ratio_mean = ratio_mean = tf.reduce_mean(ratio)

        self.tv_mean = tv_mean = tf.reduce_mean(tf.abs(ratio - 1.0))

        if sd_value:
            adv_mean = tf.reduce_sum(ADV * condition2) / (tf.reduce_sum(condition2) + 1e-8)
            adv_std = tf.sqrt(tf.reduce_sum(ADV * ADV * condition2) / (tf.reduce_sum(condition2) + 1e-8) - adv_mean * adv_mean + 1e-8)
            ADV = (ADV - adv_mean) / (adv_std + 1e-8)

        if adv_clip_range is not None:
            ADV = tf.clip_by_value(ADV, -adv_clip_range, adv_clip_range)

        pg_losses = -ADV * ratio

        # Defining indicator
        self.adv_mean = adv_mean = tf.reduce_mean(tf.abs(pg_losses))

        pg_losses2 = -ADV * tf.clip_by_value(ratio, LEFT_CLIPRANGE, RIGHT_CLIPRANGE)

        # Final PG loss
        if use_condition2:
            pg_losses2_new = -ADV * tf.clip_by_value(ratio, 1.0 - new_epsilon, 1.0 + new_epsilon)
            pg_loss = tf.reduce_sum(tf.maximum(pg_losses, pg_losses2_new) * condition2) / (tf.reduce_sum(condition2) + 1e-8)
        else:
            pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        # pg_loss = tf.reduce_mean(pg_losses)

        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        self.spinup_approxkl = spinup_approxkl = tf.reduce_mean(neglogpac - OLDNEGLOGPAC)

        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(ratio, RIGHT_CLIPRANGE)))
        clipfrac += tf.reduce_mean(tf.to_float(tf.less(ratio, LEFT_CLIPRANGE)))

        # Unclipped value
        if sd_value:
            vf_losses = tf.square(vpred - R)
            vf_loss = .5 * tf.reduce_sum(vf_losses * condition2) / (tf.reduce_sum(condition2) + 1e-8)
        else:
            vf_losses = tf.square(vpred - R)
            vf_loss = .5 * tf.reduce_mean(vf_losses)

        # Total loss
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        ### Compute variance ###############################################################
        # Final PG loss
        surr = tf.maximum(pg_losses, pg_losses2)
        if use_condition2:
            obj_mean = tf.reduce_sum(surr * condition2) / (tf.reduce_sum(condition2) + 1e-8)
            obj_std = tf.reduce_sum((surr * condition2) * (surr * condition2)) / (
                        tf.reduce_sum(condition2) + 1e-8) - obj_mean * obj_mean
        else:
            obj_mean = tf.reduce_mean(surr)
            obj_std = tf.reduce_mean(surr * surr) - obj_mean * obj_mean
        ####################################################################################

        # UPDATE THE PARAMETERS USING LOSS
        # 1. Get the model parameters
        params = tf.trainable_variables('ppo2_model')
        self.params = params
        # 2. Build our trainer
        if comm is not None and comm.Get_size() > 1:
            self.trainer = MpiAdamOptimizer(comm, learning_rate=LR, mpi_rank_weight=mpi_rank_weight, epsilon=1e-5)
        else:
            self.trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        # 3. Calculate the gradients
        grads_and_var = self.trainer.compute_gradients(loss, params)
        grads, var = zip(*grads_and_var)

        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_var = list(zip(grads, var))
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

        self.grads = grads
        self.var = var
        self._train_op = self.trainer.apply_gradients(grads_and_var)
        self.loss_names = ['adv_mean', 'tv_mean', 'policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'spinup_approxkl', \
                           'clipfrac', 'obj_mean', 'obj_std', 'ratio_mean', 'ratio_max', 'ratio_min']
        self.stats_list = [adv_mean, tv_mean, pg_loss, vf_loss, entropy, approxkl, spinup_approxkl, \
                           clipfrac, obj_mean, obj_std, ratio_mean, ratio_max, ratio_min]

        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state

        self.save = functools.partial(save_variables, sess=sess)
        self.load = functools.partial(load_variables, sess=sess)

        initialize()
        global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
        if MPI is not None:
            sync_from_root(sess, global_variables, comm=comm) #pylint: disable=E1101

        if self.use_3stage:
            self.flat_params = flat_concat(self.params)
            self.v_ph = tf.placeholder(tf.float32, shape=self.flat_params.shape)
            self.set_params = self.assign_params_from_flat(self.v_ph, params)

    def assign_params_from_flat(self, x, params):
        flat_size = lambda p: int(np.prod(p.shape.as_list()))
        splits = tf.split(x, [flat_size(p) for p in params])
        new_params = [tf.reshape(p_new, p.shape) for p, p_new in zip(params, splits)]
        return tf.group([tf.assign(p, p_new) for p, p_new in zip(params, new_params)])

    def train(self, lr, left_cliprange, right_cliprange, obs, returns, masks, actions, \
              values, neglogpacs, states=None):

        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advs = returns - values

        # Normalize the advantages
        if not self.sd_value:
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        td_map = {
            self.train_model.X : obs,
            self.A : actions,
            self.ADV : advs,
            self.R : returns,
            self.LR : lr,
            self.LEFT_CLIPRANGE : left_cliprange,
            self.RIGHT_CLIPRANGE : right_cliprange,
            self.OLDNEGLOGPAC : neglogpacs,
            self.OLDVPRED : values
        }
        if states is not None:
            td_map[self.train_model.S] = states
            td_map[self.train_model.M] = masks

        if self.use_3stage:
            td_map[self.condition2_ph] = np.ones((obs.shape[0],), dtype=np.float32)
            old_params = self.sess.run(self.flat_params)

        # stage1: one-step policy update
        if self.use_cd:
            condition2_tmp = self.sess.run(self.condition2, td_map)

        td_map[self.condition2_ph] = np.ones((obs.shape[0],), dtype=np.float32)
        stats = self.sess.run(
            self.stats_list + [self._train_op],
            td_map
        )[:-1]

        if self.use_3stage:
            # stage2: compute dropout indicator
            condition2 = self.sess.run(self.condition2, td_map)
            cond2mean_before = np.mean(condition2)
            # stage3: reset policy and train again with condition2
            self.sess.run(self.set_params, feed_dict={self.v_ph: old_params})
            td_map[self.condition2_ph] = condition2
            stats = self.sess.run(
                self.stats_list + [self._train_op],
                td_map
            )[:-1]
            # cond2mean_after = self.sess.run(tf.reduce_mean(self.condition2), td_map)
            # print(f'condition2: {cond2mean_before} -> {cond2mean_after}')

        if self.use_cd:
            td_map[self.condition2_ph] = condition2_tmp
            # td_map[self.condition2_ph] = np.ones((obs.shape[0],), dtype=np.float32)
            ratio_max, ratio_min, ratio_mean = self.sess.run([self.ratio_max, self.ratio_min, self.ratio_mean], td_map)
        else:
            ratio_max, ratio_min, ratio_mean = self.sess.run([self.ratio_max, self.ratio_min, self.ratio_mean], td_map)

        cond2sum = self.sess.run(self.condition2_sum, td_map)

        return stats, cond2sum, ratio_max, ratio_min, ratio_mean


class RegularizedModel(object):
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm, mpi_rank_weight=1, 
                reg_coef=1.0, margin_size=0.1, reg_type='average', 
                microbatch_size=None):
        self.sess = sess = get_session()

        if MPI is not None:
            comm = MPI.COMM_WORLD

        with tf.variable_scope('ppo2_model_regularized', reuse=tf.AUTO_REUSE):
            act_model = policy(nbatch_act, 1, sess)

            # Train model for training
            if microbatch_size is None:
                train_model = policy(nbatch_train, nsteps, sess)
            else:
                train_model = policy(microbatch_size, nsteps, sess)

        # CREATE THE PLACEHOLDERS
        self.A = A = train_model.pdtype.sample_placeholder([None])
        self.ADV = ADV = tf.placeholder(tf.float32, [None])
        self.R = R = tf.placeholder(tf.float32, [None])
        # Keep track of old actor
        self.OLDNEGLOGPAC = OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        # Keep track of old critic
        self.OLDVPRED = OLDVPRED = tf.placeholder(tf.float32, [None])
        self.LR = LR = tf.placeholder(tf.float32, [])

        neglogpac = train_model.pd.neglogp(A)

        # Calculate the entropy
        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # CALCULATE THE LOSS
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Clip the value to reduce variability during Critic training
        # Get the predicted value
        vpred = train_model.vf

        # Unclipped value
        vf_losses = tf.square(vpred - R)
        vf_loss = .5 * tf.reduce_mean(vf_losses)

        # Calculate ratio (pi current policy / pi old policy)
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        log_ratio = OLDNEGLOGPAC - neglogpac

        ratio_max = tf.reduce_max(ratio)
        ratio_min = tf.reduce_min(ratio)
        ratio_mean = tf.reduce_mean(ratio)

        # Defining Loss = - J is equivalent to max J
        pg_losses = -ADV * ratio

        max_adv = tf.reduce_max(ADV)
        tv_mean = tf.reduce_mean(tf.abs(ratio-1.0))
        kl_mean = tf.reduce_mean(tf.abs(log_ratio))

        if reg_type == 'average':
            ratio_reg = tf.reduce_mean(tf.abs(ratio - 1.0))
        elif reg_type == 'margin':
            abs_ratio = tf.abs(ratio - 1.0)
            ratio_reg = tf.reduce_mean(tf.clip_by_value(abs_ratio, margin_size, 1e6) - margin_size)
        elif reg_type == 'minmax':
            ratio_reg = ( (ratio_max - 1.0)**2 + ( 1.0 / ratio_min - 1.0)**2 )/2.0
        elif reg_type == 'minmax_min':
            ratio_reg = ( 1.0 / ratio_min - 1.0)**2 /2.0
        elif reg_type == 'minmax_max':
            ratio_reg = ( ratio_max - 1.0)**2 /2.0
        else:
            raise NotImplementedError

        # Final PG loss
        pg_loss = tf.reduce_mean(pg_losses)
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))

        # Total loss
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef + ratio_reg * reg_coef

        # UPDATE THE PARAMETERS USING LOSS
        # 1. Get the model parameters
        params = tf.trainable_variables('ppo2_model_reg')
        # 2. Build our trainer
        if comm is not None and comm.Get_size() > 1:
            self.trainer = MpiAdamOptimizer(comm, learning_rate=LR, mpi_rank_weight=mpi_rank_weight, epsilon=1e-5)
        else:
            self.trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        # 3. Calculate the gradients
        grads_and_var = self.trainer.compute_gradients(loss, params)
        grads, var = zip(*grads_and_var)

        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_var = list(zip(grads, var))
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

        self.grads = grads
        self.var = var
        self._train_op = self.trainer.apply_gradients(grads_and_var)
        self.loss_names = ['total_loss', 'policy_loss', 'value_loss', 'policy_entropy', 'approxkl', \
                           'reg_loss', 'max_adv', 'kl_mean', 'tv_mean', 'ratio_mean', 'ratio_max', 'ratio_min']
        self.stats_list = [loss, pg_loss, vf_loss, entropy, approxkl, \
                           ratio_reg, max_adv, kl_mean, tv_mean, ratio_mean, ratio_max, ratio_min] 

        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state

        self.save = functools.partial(save_variables, sess=sess)
        self.load = functools.partial(load_variables, sess=sess)

        initialize()
        global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
        if MPI is not None:
            sync_from_root(sess, global_variables, comm=comm) #pylint: disable=E1101

    def train(self, lr, obs, returns, masks, actions, \
              values, neglogpacs, states=None):

        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advs = returns - values

        # Normalize the advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        td_map = {
            self.train_model.X : obs,
            self.A : actions,
            self.ADV : advs,
            self.R : returns,
            self.LR : lr,
            self.OLDNEGLOGPAC : neglogpacs,
            self.OLDVPRED : values
        }
        if states is not None:
            td_map[self.train_model.S] = states
            td_map[self.train_model.M] = masks

        return self.sess.run(
            self.stats_list + [self._train_op],
            td_map
        )[:-1]


class DropoutModel(object):
    def __init__(self, *, policy, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm, mpi_rank_weight=1,cond1=False, cond2=False, sg=False, use_3stage=False, sd_delta=0.25, sd_value=False, no_abs=False, rminus=False, unbias=False, lclip=False):
        self.sess = sess = get_session()

        self.sd_value = sd_value

        if MPI is not None:
            comm = MPI.COMM_WORLD

        with tf.variable_scope('ppo2_model_dropout', reuse=tf.AUTO_REUSE):
            # CREATE OUR TWO MODELS
            # act_model that is used for sampling
            act_model = policy(nbatch_act, 1, sess)

            # Train model for training
            train_model = policy(nbatch_train, nsteps, sess)

        # CREATE THE PLACEHOLDERS
        self.A = A = train_model.pdtype.sample_placeholder([None])
        self.ADV = ADV = tf.placeholder(tf.float32, [None])
        self.R = R = tf.placeholder(tf.float32, [None])
        # Keep track of old actor
        self.OLDNEGLOGPAC = OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        # Keep track of old critic
        self.OLDVPRED = OLDVPRED = tf.placeholder(tf.float32, [None])
        self.LR = LR = tf.placeholder(tf.float32, [])
        # Cliprange
        self.CLIPRANGE = CLIPRANGE = tf.placeholder(tf.float32, [])
        neglogpac = train_model.pd.neglogp(A)
        self.use_3stage = use_3stage
        # if self.use_3stage:
        self.condition2_ph = tf.placeholder(tf.float32, [None])
        self.use_cd = cond2


        # Calculate the entropy
        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # CALCULATE THE LOSS
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Clip the value to reduce variability during Critic training
        # Get the predicted value
        vpred = train_model.vf


        # Calculate ratio (pi current policy / pi old policy)
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)

        # Surrogate objectives
        CLIPRANGE = 0.2

        # clip 太小的
        DELTA1 = 0.03
        condition = tf.cast(tf.greater(tf.abs(ratio - 1.0), DELTA1), tf.float32)
        a = tf.cast(tf.equal(tf.reduce_mean(condition), 0), tf.float32)
        condition1 = a * tf.ones_like(condition) + (1 - a) * condition
        print("condition1:", tf.reduce_mean(condition1))

        # cilp 太大的
        DELTA2 = sd_delta
        print ('sd_delta:', sd_delta)
        print ('no_abs:', no_abs)
        print ('lclip:', lclip)
        print ('rminus:', rminus)
        print ('unbias:', unbias)
        if no_abs: # r - 1 < 0.25
            condition = tf.cast(tf.less(ratio - 1.0, DELTA2), tf.float32)
        elif lclip: # 1 - r < 0.25
            condition = tf.cast(tf.less(1.0 - ratio, DELTA2), tf.float32)
        elif unbias:
            condition = tf.cast(tf.less(ratio, 1.0+DELTA2), tf.float32) * tf.cast(tf.greater(ratio, 1.0/(1.0+DELTA2)), tf.float32)
        else:
            condition = tf.cast(tf.less(tf.abs(ratio - 1.0), DELTA2), tf.float32)
        a = tf.cast(tf.equal(tf.reduce_mean(condition), 0), tf.float32)
        condition2 = a * tf.ones_like(condition) + (1 - a) * condition
        print("condition2:", tf.reduce_mean(condition2))

        self.condition2_sum = tf.reduce_sum(condition2)
        self.condition2 = condition2

        use_condition1 = cond1
        use_condition2 = cond2
        print('use_condition1:', use_condition1)
        print('use_condition2:', use_condition2)

        if use_condition2:
            # ratio_max = tf.reduce_max(ratio)
            # ratio_min = tf.reduce_min(ratio)
            # ratio_mean = tf.reduce_mean(ratio)
            self.ratio_max = ratio_max = tf.reduce_max(ratio * self.condition2_ph)
            self.ratio_min = ratio_min = tf.reduce_min(ratio * self.condition2_ph + 10.0 * (1.0 - self.condition2_ph))
            #self.ratio_mean = ratio_mean = tf.reduce_sum(ratio * self.condition2_ph) / (tf.reduce_sum(self.condition2_ph) + 1e-8)
            self.ratio_mean = ratio_mean = tf.reduce_mean(ratio)
            # self.ratio_max = ratio_max = tf.reduce_max(ratio * condition)
            # self.ratio_min = ratio_min = tf.reduce_min(ratio * condition + 1000.0 * (1.0 - condition))
            # self.ratio_mean = ratio_mean = tf.reduce_mean(ratio * condition)
        else:
            self.ratio_max = ratio_max = tf.reduce_max(ratio)
            self.ratio_min = ratio_min = tf.reduce_min(ratio)
            self.ratio_mean = ratio_mean = tf.reduce_mean(ratio)

        self.tv_mean = tv_mean = tf.reduce_mean(tf.abs(ratio - 1.0))

        if sd_value:
            adv_mean = tf.reduce_sum(ADV * condition2) / (tf.reduce_sum(condition2) + 1e-8)
            adv_std = tf.sqrt(tf.reduce_sum(ADV * ADV * condition2) / (tf.reduce_sum(condition2) + 1e-8) - adv_mean * adv_mean + 1e-8)
            ADV = (ADV - adv_mean) / (adv_std + 1e-8)

        # Defining Loss = - J is equivalent to max J
        if rminus:
            pg_losses = -ADV * (ratio - 1)
        else:
            pg_losses = -ADV * ratio
        ori_obj_mean = tf.reduce_mean(pg_losses)
        # E[x^2] - E[x]^2
        ori_obj_std = tf.reduce_mean(pg_losses * pg_losses) - ori_obj_mean * ori_obj_mean

        # Defining indicator
        self.adv_mean = adv_mean = tf.reduce_mean(tf.abs(pg_losses))

        # Final PG loss
        sd_pg_loss = tf.reduce_sum(pg_losses * condition2) / (tf.reduce_sum(condition2) + 1e-8)
        sd_obj_mean = sd_pg_loss
        # E[x^2] - E[x]^2
        sd_obj_std = tf.reduce_sum((pg_losses * condition2) * (pg_losses * condition2)) / (tf.reduce_sum(condition2) + 1e-8) - sd_obj_mean * sd_obj_mean
        if use_condition2:
            # espo+condition2
            if self.use_3stage:
                pg_loss = tf.reduce_sum(pg_losses * self.condition2_ph) / (tf.reduce_sum(self.condition2_ph) + 1e-8)
            else:
                pg_loss = tf.reduce_sum(pg_losses * condition2) / (tf.reduce_sum(condition2) + 1e-8)
                sd_obj_mean = pg_loss
                # E[x^2] - E[x]^2
                sd_obj_std = tf.reduce_sum((pg_losses * condition2) * (pg_losses * condition2)) / (tf.reduce_sum(condition2) + 1e-8) - sd_obj_mean * sd_obj_mean
                obj_mean = sd_obj_mean
                obj_std = sd_obj_std
        if (use_condition1 == False) and (use_condition2 == False):
            # espo
            pg_loss = tf.reduce_mean(pg_losses)
            obj_mean = tf.reduce_mean(pg_losses)
            obj_std = tf.reduce_mean(pg_losses * pg_losses) - obj_mean * obj_mean
        # pg_loss = tf.reduce_mean(pg_losses)

        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        self.spinup_approxkl = spinup_approxkl = tf.reduce_mean(neglogpac - OLDNEGLOGPAC)

        # Unclipped value
        if sd_value:
            vf_losses = tf.square(vpred - R)
            vf_loss = .5 * tf.reduce_sum(vf_losses * condition2) / (tf.reduce_sum(condition2) + 1e-8)
        else:
            vf_losses = tf.square(vpred - R)
            vf_loss = .5 * tf.reduce_mean(vf_losses)

        # Total loss
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        # UPDATE THE PARAMETERS USING LOSS
        # 1. Get the model parameters
        params = tf.trainable_variables('ppo2_model_dropout')
        self.params = params
        # 2. Build our trainer
        if comm is not None and comm.Get_size() > 1:
            self.trainer = MpiAdamOptimizer(comm, learning_rate=LR, mpi_rank_weight=mpi_rank_weight, epsilon=1e-5)
        else:
            self.trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        # 3. Calculate the gradients
        grads_and_var = self.trainer.compute_gradients(loss, params)
        grads, var = zip(*grads_and_var)

        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_var = list(zip(grads, var))
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

        self.grads = grads
        self.var = var
        self._train_op = self.trainer.apply_gradients(grads_and_var)
        self.loss_names = ['adv_mean', 'tv_mean', 'policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'spinup_approxkl', \
                           'ori_obj_mean', 'ori_obj_std', 'sd_obj_mean', 'sd_obj_std', 'obj_mean', 'obj_std', 'ratio_mean', 'ratio_max', 'ratio_min'] #最后两个必须是ratio_max和ratio_min
        self.stats_list = [adv_mean, tv_mean, pg_loss, vf_loss, entropy, approxkl, spinup_approxkl, \
                           ori_obj_mean, ori_obj_std, sd_obj_mean, sd_obj_std, obj_mean, obj_std, ratio_mean, ratio_max, ratio_min] #最后两个必须是ratio_max和ratio_min

        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state

        self.save = functools.partial(save_variables, sess=sess)
        self.load = functools.partial(load_variables, sess=sess)

        initialize()
        global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
        if MPI is not None:
            sync_from_root(sess, global_variables, comm=comm) #pylint: disable=E1101

        if self.use_3stage:
            self.flat_params = flat_concat(self.params)
            self.v_ph = tf.placeholder(tf.float32, shape=self.flat_params.shape)
            self.set_params = self.assign_params_from_flat(self.v_ph, params)

    def assign_params_from_flat(self, x, params):
        flat_size = lambda p: int(np.prod(p.shape.as_list()))
        splits = tf.split(x, [flat_size(p) for p in params])
        new_params = [tf.reshape(p_new, p.shape) for p, p_new in zip(params, splits)]
        return tf.group([tf.assign(p, p_new) for p, p_new in zip(params, new_params)])

    def train(self, lr, obs, returns, masks, actions, \
              values, neglogpacs, states=None):

        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advs = returns - values

        # Normalize the advantages
        if not self.sd_value:
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        td_map = {
            self.train_model.X : obs,
            self.A : actions,
            self.ADV : advs,
            self.R : returns,
            self.LR : lr,
            self.OLDNEGLOGPAC : neglogpacs,
            self.OLDVPRED : values
        }
        if states is not None:
            td_map[self.train_model.S] = states
            td_map[self.train_model.M] = masks

        if self.use_3stage:
            td_map[self.condition2_ph] = np.ones((obs.shape[0],), dtype=np.float32)
            old_params = self.sess.run(self.flat_params)

        # stage1: one-step policy update
        if self.use_cd:
            condition2_tmp = self.sess.run(self.condition2, td_map)

        td_map[self.condition2_ph] = np.ones((obs.shape[0],), dtype=np.float32)
        stats = self.sess.run(
            self.stats_list + [self._train_op],
            td_map
        )[:-1]

        if self.use_3stage:
            # stage2: compute dropout indicator
            condition2 = self.sess.run(self.condition2, td_map)
            cond2mean_before = np.mean(condition2)
            # stage3: reset policy and train again with condition2
            self.sess.run(self.set_params, feed_dict={self.v_ph: old_params})
            td_map[self.condition2_ph] = condition2
            stats = self.sess.run(
                self.stats_list + [self._train_op],
                td_map
            )[:-1]
            # cond2mean_after = self.sess.run(tf.reduce_mean(self.condition2), td_map)
            # print(f'condition2: {cond2mean_before} -> {cond2mean_after}')

        if self.use_cd:
            td_map[self.condition2_ph] = condition2_tmp
            # td_map[self.condition2_ph] = np.ones((obs.shape[0],), dtype=np.float32)
            ratio_max, ratio_min, ratio_mean = self.sess.run([self.ratio_max, self.ratio_min, self.ratio_mean], td_map)
        else:
            ratio_max, ratio_min, ratio_mean = self.sess.run([self.ratio_max, self.ratio_min, self.ratio_mean], td_map)

        curr_tv_mean, curr_adv_mean ,condition2_sum = self.sess.run([self.tv_mean, self.adv_mean,self.condition2_sum], td_map)
        return stats, curr_tv_mean, curr_adv_mean, condition2_sum, ratio_max, ratio_min, ratio_mean

    def full_batch_estimate(self, lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advs = returns - values

        # Normalize the advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        td_map = {
            self.train_model.X: obs,
            self.A: actions,
            self.ADV: advs,
            self.R: returns,
            self.LR: lr,
            self.CLIPRANGE: cliprange,
            self.OLDNEGLOGPAC: neglogpacs,
            self.OLDVPRED: values
        }
        if states is not None:
            td_map[self.train_model.S] = states
            td_map[self.train_model.M] = masks

        curr_tv_mean, condition2 = self.sess.run([self.tv_mean, self.condition2],
                                                                    td_map)
        return curr_tv_mean, condition2
