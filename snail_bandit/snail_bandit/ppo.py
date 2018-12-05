import numpy as np
import tensorflow as tf
import gym
import time
import core
from logx import EpochLogger
from mpi_tf import MpiAdamOptimizer, sync_all_params
from mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from bandit import BernoulliBanditEnv, GaussianBanditEnv
from collections import defaultdict
from collections import deque


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, act_dim, size, gamma=0.99, lam=0.95):
        # self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.act_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.act_buf, self.adv_buf, self.ret_buf, self.logp_buf, self.rew_buf]
        # return [self.act_buf.reshape(trials_per_epoch, steps_per_trial), 
        # self.adv_buf.reshape(trials_per_epoch, steps_per_trial), 
        # self.ret_buf.reshape(trials_per_epoch, steps_per_trial), 
        # self.logp_buf.reshape(trials_per_epoch, steps_per_trial), 
        # self.rew_buf.reshape((trials_per_epoch, steps_per_trial, 1))]


"""

Proximal Policy Optimization (by clipping), 

with early stopping based on approximate KL

"""
def ppo(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0,
        batch_size=250000, n = 100, epochs=100, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=1000, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, logger_kwargs=dict(), save_freq=10):
    """

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in placeholder symbols 
            for state, ``x_ph``, and action, ``a_ph``, and returns the main 
            outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       (batch, act_dim)  | Samples actions from policy given 
                                           | states.
            ``logp``     (batch,)          | Gives log probability, according to
                                           | the policy, of taking actions ``a_ph``
                                           | in states ``x_ph``.
            ``logp_pi``  (batch,)          | Gives log probability, according to
                                           | the policy, of the action sampled by
                                           | ``pi``.
            ``v``        (batch,)          | Gives the value estimate for states
                                           | in ``x_ph``. (Critical: make sure 
                                           | to flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
            function you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while 
            still profiting (improving the objective function)? The new policy 
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.)

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take 
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    tf.set_random_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    
    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # Inputs to computation graph
    # x_ph, a_ph = core.placeholders_from_spaces(env.observation_space, env.action_space)
    # rew_ph, adv_ph, ret_ph, logp_old_ph = core.placeholders(1, None, None, None)
    a_ph = tf.placeholder(dtype=tf.int32, shape=(None, n), name='a_ph')
    rew_ph = tf.placeholder(dtype=tf.float32, shape=(None, n), name='rew_ph')
#    input_ph = tf.placeholder(dtype=tf.float32, shape=(None, None, n, None), name='rew_ph')
    adv_ph = tf.placeholder(dtype=tf.float32, shape=(None), name='adv_ph')
    ret_ph = tf.placeholder(dtype=tf.float32, shape=(None), name='ret_ph')
    logp_old_ph = tf.placeholder(dtype=tf.float32, shape=(None), name='logp_old_ph')
    # Main outputs from computation graph
    pi, logp, logp_pi, v = actor_critic(a_ph, rew_ph, n, action_space=env.action_space)

    # Need all placeholders in *this* order later (to zip with data from buffer)
    all_phs = [a_ph, adv_ph, ret_ph, logp_old_ph, rew_ph]
#    for ph in all_phs:
#        print(ph.shape)

    # Every step, get: action, value, and logprob
    get_action_ops = [pi, v, logp_pi]

    # Experience buffer
    trials = batch_size // n
    buf = PPOBuffer(act_dim, batch_size, gamma, lam)

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in ['pi', 'v'])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # PPO objectives
    ratio = tf.exp(logp - logp_old_ph)          # pi(a|s) / pi_old(a|s)
    min_adv = tf.where(adv_ph>0, (1+clip_ratio)*adv_ph, (1-clip_ratio)*adv_ph)
    pi_loss = -tf.reduce_mean(tf.minimum(ratio * adv_ph, min_adv))
    v_loss = tf.reduce_mean((ret_ph - v)**2)

    # Info (useful to watch during learning)
    approx_kl = tf.reduce_mean(logp_old_ph - logp)      # a sample estimate for KL-divergence, easy to compute
    approx_ent = tf.reduce_mean(-logp)                  # a sample estimate for entropy, also easy to compute
    clipped = tf.logical_or(ratio > (1+clip_ratio), ratio < (1-clip_ratio))
    clipfrac = tf.reduce_mean(tf.cast(clipped, tf.float32))

    # Optimizers
    train_pi = MpiAdamOptimizer(learning_rate=pi_lr).minimize(pi_loss)
    train_v = MpiAdamOptimizer(learning_rate=vf_lr).minimize(v_loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Sync params across processes
    sess.run(sync_all_params())

    # Setup model saving
#    logger.setup_tf_saver(sess, inputs={'x': x_ph}, outputs={'pi': pi, 'v': v})

    def update():
        inputs = {k:v for k,v in zip(all_phs, buf.get())}
#        inputs[a_ph] = np.tril(np.transpose(np.repeat(inputs[a_ph], n).reshape(trials, n, n), [0, 2, 1]))
#        inputs[rew_ph] = np.tril(np.transpose(np.repeat(inputs[rew_ph], n).reshape(trials, n, n), [0, 2, 1]))
        inputs[a_ph] = inputs[a_ph].reshape(trials, n)
        inputs[rew_ph] = inputs[rew_ph].reshape(trials, n)
        pi_l_old, v_l_old, ent = sess.run([pi_loss, v_loss, approx_ent], feed_dict=inputs)
        
        # Training
        for i in range(train_pi_iters):
            _, kl = sess.run([train_pi, approx_kl], feed_dict=inputs)
#            kl = mpi_avg(kl)
#            if kl > 1.5 * target_kl:
#                logger.log('Early stopping at step %d due to reaching max kl.'%i)
#                break
        logger.store(StopIter=i)
        for _ in range(train_v_iters):
            sess.run(train_v, feed_dict=inputs)

        # Log changes from update
        pi_l_new, v_l_new, kl, cf = sess.run(
                [pi_loss, v_loss, approx_kl, clipfrac], feed_dict=inputs)
        logger.store(LossPi=pi_l_old, LossV=v_l_old, 
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(pi_l_new - pi_l_old),
                     DeltaLossV=(v_l_new - v_l_old))

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), np.zeros(1), False, 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for trail in range(trials):
            print(trail)
#            last_a = np.zeros(n).reshape(1, n)
#            last_r = np.zeros(n).reshape(1, n)
            last_a = deque(n * [0], n)
            last_r = deque(n * [0], n)
            means = env.sample_tasks(1)[0]
            print('task means:', means)
            action_dict = defaultdict(int)
            env.reset_task(means)
            for episode in range(n):
                a, v_t, logp_t = sess.run(
                        get_action_ops, feed_dict={
                                a_ph: np.array(last_a).reshape(1, n), 
                                rew_ph: np.array(last_r).reshape(1, n)})
#                print("a shape:", a.shape)
#                print("v_t shape:", v_t.shape)
#                print("logp_t shape:", logp_t.shape)
#                choosen_a = a[episode, 0]
#                choosen_v_t = v_t[0, episode]
#                choosen_logp_t = logp_t[episode]
                print('a:', a)
                choosen_a = a[-1]
                choosen_v_t = v_t[-1]
                choosen_logp_t = logp_t[-1]
                action_dict[choosen_a] += 1
                o, r, d, _ = env.step(choosen_a)
                # save and log
                buf.store(choosen_a, r, choosen_v_t, choosen_logp_t)
                logger.store(VVals=v_t)
                ep_ret += r
                ep_len += 1
                
                last_a.append(choosen_a)
                last_r.append(r)
    
                terminal = d or (ep_len == max_ep_len)
                if terminal or (episode==n-1):
                    if not(terminal):
                        print('Warning: trajectory cut off by epoch at %d steps.'%ep_len)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    last_val = r
                    buf.finish_path(last_val)
                    if terminal:
                        # only save EpRet / EpLen if trajectory finished
                        logger.store(EpRet=ep_ret, EpLen=ep_len)
                    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            print(action_dict)
        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)
        # Perform PPO update!
        update()

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*batch_size)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

if __name__ == '__main__':
    tf.reset_default_graph()
    exp_name = 'ppo'
    num_arms = 5
    env = GaussianBanditEnv
    env_fn = env(num_arms)
    actor_critic = core.snail_actor_critic
    ac_kwargs=dict()
    seed = 0
    batch_size = 500
    n = 10
    epochs=100
    gamma=0.99
    clip_ratio=0.2
    pi_lr=3e-4
    vf_lr=1e-3
    train_pi_iters=100
    train_v_iters=100
    lam=0.3
    max_ep_len=1000
    target_kl=0.01
    logger_kwargs=dict()
    save_freq=10
    from run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(exp_name, seed)
    ppo(lambda: env_fn, actor_critic, ac_kwargs, seed,
        batch_size, n, epochs, gamma, clip_ratio, pi_lr,
        vf_lr, train_pi_iters, train_v_iters, lam, max_ep_len,
        target_kl, logger_kwargs, save_freq)