import numpy as np
import tensorflow as tf
from bandit import BernoulliBanditEnv, GaussianBanditEnv
import logx
from collections import defaultdict, deque

tf.set_random_seed(0)
np.random.seed(0)

tf.reset_default_graph()
sess = tf.Session()

model = logx.restore_tf_graph(sess, "../data/ppo/ppo_s0/simple_save11")

a_in = model['a']
r_in = model['r']
pi_out = model['pi']

num_arms = 10
n = 100
env = BernoulliBanditEnv(num_arms)
mean = env.sample_tasks(1)[0]
print('task means:', mean)
action_dict = defaultdict(int)
env.reset_task(mean)

last_a = deque(n * [0], n)
last_r = deque(n * [0], n)

o, r, d, ep_ret, ep_len = env.reset(), np.zeros(1), False, 0, 0
total_reward = 0

for _ in range(n):
    a = sess.run([pi_out], feed_dict={
                            a_in: np.array(last_a).reshape(1, n), 
                            r_in: np.array(last_r).reshape(1, n)})
    # print(a[0][-1])
    choosen_a = a[0][-1]
    action_dict[choosen_a] += 1
    o, r, d, _ = env.step(choosen_a)  
    total_reward += r
    
    last_a.append(choosen_a)
    last_r.append(r)
    
print(action_dict)
print('average reward:', total_reward / n)  