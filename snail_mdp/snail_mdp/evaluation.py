import numpy as np
import tensorflow as tf
from mdp import TabularMDPEnv
import logx
from collections import defaultdict, deque

tf.set_random_seed(0)
np.random.seed(0)

tf.reset_default_graph()
sess = tf.Session()

model = logx.restore_tf_graph(sess, "../data/ppo/ppo_s0/simple_save7")

x_in = model['x']
t_in = model['t']
a_in = model['a']
r_in = model['r']
pi_out = model['pi']

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

o_deque = deque(sequence_length * [0], sequence_length)
t_deque = deque(sequence_length * [0], sequence_length)
last_a = deque(sequence_length * [0], sequence_length)
last_r = deque(sequence_length * [0], sequence_length)

o, r, d, _, ep_len = env.reset(), np.zeros(1), False, 0, 0
total_reward = 0

for episode in range(sequence_length):
    a = sess.run([pi_out], feed_dict={x_in: np.array(o_deque).reshape(1, sequence_length), 
                                t_in: np.array(t_deque).reshape(1, sequence_length), 
                                a_in: np.array(last_a).reshape(1, sequence_length),
                                r_in: np.array(last_r).reshape(1, sequence_length)})
    choosen_a = a[0][-1]
    action_dict[choosen_a] += 1
    o, r, d, _ = env.step(choosen_a)  
    total_reward += r
    
    o_deque.append(o)
    t_deque.append(int(d))
    last_a.append(choosen_a)
    last_r.append(r)
    
    ep_len += 1
    t = ep_len == max_ep_len
    
    terminal = d or t
    if terminal or (episode == sequence_length - 1):
        o, r, d, _, ep_len = env.reset(), 0, False, 0, 0
        o_deque[-1] = 0
        t_deque[-1] = 0
        last_a[-1] = 0
        last_r[-1] = 0
    
print(action_dict)
print('average reward:', total_reward / sequence_length)
    
    