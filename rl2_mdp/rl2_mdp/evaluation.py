import numpy as np
import tensorflow as tf
from mdp import TabularMDPEnv
import logx
from collections import defaultdict

tf.set_random_seed(0)
np.random.seed(0)

tf.reset_default_graph()
sess = tf.Session()

model = logx.restore_tf_graph(sess, "../data/ppo/ppo_s0/simple_save0")

x_in = model['x']
t_in = model['t']
a_in = model['a']
r_in = model['r']
pi_rnn_state_in = model['pi_rnn_state_in']
v_rnn_state_in = model['v_rnn_state_in']
pi_out = model['pi']
pi_rnn_state_out = model['pi_rnn_state_out']
v_rnn_state_out = model['v_rnn_state_out']

num_states = 10
num_actions = 5
env = TabularMDPEnv(num_states, num_actions)
mean = env.sample_tasks(1)[0]
#print('task means:', mean)
action_dict = defaultdict(int)
env.reset_task(mean)

gru_units = 256
n = 100
max_ep_len = 10
sequence_length = n * max_ep_len

last_a = np.array(0)
last_r = np.array(0)
last_pi_rnn_state = np.zeros((1, gru_units), np.float32)
last_v_rnn_state = np.zeros((1, gru_units), np.float32)

o, r, d, ep_ret, ep_len = env.reset(), np.zeros(1), False, 0, 0
total_reward = 0

for episode in range(sequence_length):
    a, pi_rnn_state_t, v_rnn_state_t = sess.run(
                        [pi_out, pi_rnn_state_out, v_rnn_state_out], feed_dict={
                                x_in: o.reshape(1, 1, 1),
                                t_in: np.array(int(d)).reshape(1, 1, 1), 
                                a_in: last_a.reshape(1, 1, 1), 
                                r_in: last_r.reshape(1, 1, 1), 
                                pi_rnn_state_in: last_pi_rnn_state, 
                                v_rnn_state_in: last_v_rnn_state})
    choosen_a = a[0]
    action_dict[choosen_a] += 1
    o, r, d, _ = env.step(choosen_a)  
    total_reward += r
    
    last_a = np.array(choosen_a)
    last_r = np.array(r)
    last_pi_rnn_state = pi_rnn_state_t
    last_v_rnn_state = v_rnn_state_t
    
    
    ep_ret += r
    ep_len += 1
    t = ep_len == max_ep_len
    
    terminal = d or t
    if terminal or (episode == sequence_length - 1):
        o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
        last_a = np.array(0)
        last_r = np.array(0)
    
print(action_dict)
print('average reward:', total_reward / sequence_length)
    
    