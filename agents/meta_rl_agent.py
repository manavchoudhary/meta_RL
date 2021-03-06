import tensorflow as tf
import yaml
import numpy as np
import gym
import matplotlib.pyplot as plt
import pickle
import copy
import sys
import bandit_env

class agent():
    def __init__(self, state_size, action_size, reward_size, time_size):
        self.state_size = state_size
        self.action_size = action_size
        self.reward_size = reward_size
        self.time_size = time_size
        self.input_size = self.state_size + self.action_size + self.reward_size
        # self.input_size = self.time_size + self.action_size + self.reward_size
        self.lstm_neurons = 48
        self.lstm_layers = 1

        self.architecture()

    def single_cell(self, shape):
        return tf.nn.rnn_cell.LSTMCell(shape, dtype=tf.float32, forget_bias=1.0, initializer=tf.contrib.layers.xavier_initializer(seed=2349), name='basic_lstm_cell')

    def rnn_net(self):
        lstm_layers = tf.nn.rnn_cell.MultiRNNCell([self.single_cell(self.lstm_neurons) for _ in range(self.lstm_layers)])
        batch_size = tf.shape(self.input)[0]
        self.init_state = lstm_layers.zero_state(batch_size=batch_size, dtype=tf.float32)
        lstm_out, self.rnn_hidden_state = tf.nn.dynamic_rnn(cell=lstm_layers, inputs = self.input, initial_state=self.init_state)
        hidden1 = tf.reshape(lstm_out, [-1, self.lstm_neurons])
        return hidden1

    def architecture(self):
        #input shape : (Batch size, sequence_length (num of tasks*num of episodes), input_size), Input==== [state, prev_action, prev_reward, time_step]
        self.input = tf.placeholder(shape=[None, None, self.input_size], dtype=tf.float32)
        self.target_v = tf.placeholder(shape=[None, None, 1], dtype=tf.float32)
        self.advantage = tf.placeholder(shape=[None, None, 1], dtype=tf.float32)
        self.action_taken = tf.placeholder(shape=[None, None, self.action_size], dtype=tf.float32)
        self.lr = tf.placeholder(tf.float32)

        self.hidden_1 = self.rnn_net()
        self.value_fn = tf.contrib.layers.fully_connected(self.hidden_1, 1, activation_fn=None,
                                                     weights_initializer = tf.contrib.layers.xavier_initializer(seed=2345),
                                                     biases_initializer= tf.contrib.layers.xavier_initializer(seed=2346), trainable=True)
        self.policy = tf.nn.softmax(tf.contrib.layers.fully_connected(self.hidden_1, self.action_size, activation_fn=None,
                                                                 weights_initializer = tf.contrib.layers.xavier_initializer(seed=2347),
                                                                 biases_initializer=tf.contrib.layers.xavier_initializer(seed=2348), trainable=True))

        self.target_v_flat = tf.reshape(self.target_v, [-1, 1],)
        self.advantage_flat = tf.reshape(self.advantage, [-1,1])
        self.action_taken_flat = tf.reshape(self.action_taken, [-1, self.action_size])
        value_loss = 0.5*tf.reduce_mean(tf.square(self.target_v_flat - self.value_fn))
        policy_loss = -tf.reduce_mean(tf.log(tf.reduce_sum(self.policy*self.action_taken_flat)+1e-9)*self.advantage_flat)
        entropy = -tf.reduce_mean(self.policy*tf.log(self.policy+1e-9))
        loss = policy_loss + value_loss - 0.05*entropy
        self.loss_summary = tf.summary.scalar('train_loss', loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, name='adam_opt')
        self.train_op = optimizer.minimize(loss)

def to_one_hot(val, vector_len):
    one_hot = np.zeros(vector_len)
    one_hot[val] = 1.0
    return one_hot

def train(agent, env):
    num_tasks = 100000
    num_episodes_per_task = 1
    # num_episodes_per_task = 2
    # max_len_episode = 100
    max_len_episode = 200
    discount_factor = 0.9
    # discount_factor = 0.8
    learning_rate = 0.001

    sess = tf.Session()
    saver = tf.train.Saver(max_to_keep=None)
    init_vars = tf.global_variables_initializer()
    sess.run(init_vars)
    results_dir = '.'
    writer_op = tf.summary.FileWriter(results_dir + '/tf_graphs', sess.graph)

    rewards_plot = []

    for task in range(num_tasks):
        # env = gym.make('correlatedbandit-v0', prob=0.1)
        env = gym.make('CartPole-v0')
        state_size = env.observation_space.shape[0]
        reward_size = 1
        num_actions = env.action_space.n
        env.reset()
        # input_size = 1+num_actions+1
        input_size = state_size+num_actions+reward_size

        hidden_state_val = None
        task_states = []; task_time_steps = []; task_rewards = []; task_actions = []; task_state_values = []; task_advantages = []; task_target_v = []

        for episode in range(num_episodes_per_task):
            env.reset()
            time_step = 1
            done = False
            # state = [0]
            state = env.state
            reward = 0.0
            action = 0
            episode_states = []; episode_time_steps = []; episode_rewards = []; episode_actions = []; episode_state_values = []; episode_advantages = []; episode_target_v = []

            while(done==False and time_step<=max_len_episode):
                #reshape to size : (Batch_size, sequence_length(num_tasks*num_episodes_per_task), (state_size + action_size + reward_size + time_step_size))
                a2c_input = np.concatenate((np.array(state), to_one_hot(action, num_actions), np.array([reward]))).reshape((1, -1, input_size))
                # a2c_input = np.concatenate((np.array([time_step]), to_one_hot(action, num_actions), np.array([reward]))).reshape((1, -1, input_size))
                if(hidden_state_val==None):
                    feed = {agent.input:a2c_input}
                else:
                    feed = {agent.input: a2c_input, agent.init_state:hidden_state_val}
                policy, state_v_val, hidden_state_val = sess.run([agent.policy, agent.value_fn, agent.rnn_hidden_state], feed_dict = feed)
                state_v_val = state_v_val[0][0]
                episode_state_values.append(state_v_val)
                policy = policy[0]
                action = np.random.choice(len(policy), p = policy)
                episode_states.append(state)
                episode_time_steps.append(time_step)
                episode_actions.append(to_one_hot(action, num_actions))
                state, reward, done, info = env.step(action)
                episode_rewards.append(reward)
                time_step+=1

            R = 0.0
            if(done==True):
                R = 0.0
            else:
                # final_time_step = time_step
                final_state = state
                a2c_input = np.concatenate((np.array([final_state]), to_one_hot(action, num_actions), np.array([reward]))).reshape((1, -1, input_size))
                # a2c_input = np.concatenate((np.array([final_time_step]), to_one_hot(action, num_actions), np.array([reward]))).reshape((1, -1, input_size))
                feed = {agent.input: a2c_input, agent.init_state:hidden_state_val}
                state_v_val = sess.run([agent.value_fn], feed_dict=feed)
                state_v_val = state_v_val[0][0]
                R = state_v_val
            #Generalized Advantage Estimate ::: A = Sum((gamma^t)*delta), delta = r + gamma*V(s(t+1)) - V(s(t))
            episode_general_adv = np.array(episode_rewards) + discount_factor*np.array(episode_state_values[1:]+[0.0]) - np.array(episode_state_values)
            adv = 0.0
            for i in reversed(range(len(episode_rewards))):
                R = episode_rewards[i] + discount_factor*R
                adv = episode_general_adv[i] + discount_factor*adv
                #standard advantage estimate ::: A = R - V(s)
                # adv = R - episode_state_values[i]
                episode_target_v.append(R)
                episode_advantages.append(adv)
            episode_target_v = np.flip(episode_target_v)
            episode_advantages = np.flip(episode_advantages)

            task_states.append(episode_states); task_time_steps.append(episode_time_steps); task_rewards.append(episode_rewards); task_actions.append(episode_actions); task_state_values.append(episode_state_values); task_advantages.append(episode_advantages); task_target_v.append(episode_target_v)

            rewards_plot.append(np.sum(episode_rewards))

        a2c_input = np.concatenate((np.vstack(task_states).reshape((1, -1, state_size)), np.concatenate((np.array([[[1,0]]]), np.vstack(task_actions).reshape((1, -1, num_actions))[:, :-1, :]), axis=1), np.concatenate((np.array([[[0.0]]]), np.hstack(task_rewards).reshape((1, -1, reward_size))[:, :-1, :]), axis=1)), axis=2)
        # a2c_input = np.concatenate((np.vstack(task_time_steps).reshape((1, -1, 1)), np.vstack(task_actions).reshape((1, -1, num_actions)), np.hstack(task_rewards).reshape((1, -1, reward_size))), axis=2)
        task_advantages = np.hstack(task_advantages).reshape((1, -1, 1))
        task_target_v = np.hstack(task_target_v).reshape((1, -1, 1))
        task_actions = np.vstack(task_actions).reshape((1, -1, num_actions))
        feed = {agent.input: a2c_input, agent.action_taken: task_actions, agent.advantage: task_advantages, agent.target_v: task_target_v, agent.lr: learning_rate}
        _, loss_summary_val = sess.run([agent.train_op, agent.loss_summary], feed_dict=feed)
        writer_op.add_summary(loss_summary_val, task)

        if ((task + 1) % 1000 == 0):
            print(task)
            cumsum_rewards = np.cumsum(np.insert(rewards_plot, 0, 0))
            smoothed_rewards = (cumsum_rewards[50:] - cumsum_rewards[:-50])/50.0
            plt.plot(smoothed_rewards)
            plt.savefig('graph_task_'+str(task+1)+'.png')
            saver.save(sess, results_dir + '/models/itr_{0}.ckpt'.format(task+1))

    cumsum_rewards = np.cumsum(np.insert(rewards_plot, 0, 0))
    smoothed_rewards = (cumsum_rewards[50:] - cumsum_rewards[:-50]) / 50.0
    fp = open('baseline_A2C_bandit_results.pickle', 'wb')
    pickle.dump(smoothed_rewards, fp)
    fp.close()
    saver.save(sess, results_dir + '/models/final.ckpt')
    sess.close()
    plt.plot(smoothed_rewards)
    plt.show()
    return


def test(agent, env):
    max_len_episode = 1000
    num_episodes_per_task = 100
    sess = tf.Session()
    saver = tf.train.Saver(max_to_keep=None)
    results_dir = './exp_results/cart_pole/generalized_Adv/meta_RL/2_episode'
    saver.restore(sess, results_dir+'/models/final.ckpt')
    rewards_plot = []
    env = gym.make('CartPole-v0')
    state_size = env.observation_space.shape[0]
    reward_size = 1
    num_actions = env.action_space.n
    input_size = state_size + num_actions + reward_size
    env.reset()

    hidden_state_val = None
    for episode in range(num_episodes_per_task):
        env.reset()
        time_step = 1
        done = False
        # state = [0]
        state = env.state
        reward = 0.0
        action = 0

        episode_rewards = []
        while (done == False and time_step <= max_len_episode):
            # reshape to size : (Batch_size, sequence_length(num_tasks*num_episodes_per_task), (state_size + action_size + reward_size + time_step_size))
            a2c_input = np.concatenate((np.array(state), to_one_hot(action, num_actions), np.array([reward]))).reshape((1, -1, input_size))
            # a2c_input = np.concatenate((np.array([time_step]), to_one_hot(action, num_actions), np.array([reward]))).reshape((1, -1, input_size))
            if (hidden_state_val == None):
                feed = {agent.input: a2c_input}
            else:
                feed = {agent.input: a2c_input, agent.init_state: hidden_state_val}
            policy, state_v_val, hidden_state_val = sess.run([agent.policy, agent.value_fn, agent.rnn_hidden_state], feed_dict=feed)
            policy = policy[0]
            action = np.random.choice(len(policy), p=policy)
            print(state)
            state, reward, done, info = env.step(action)
            episode_rewards.append(reward)
            time_step += 1

        rewards_plot.append(np.sum(episode_rewards))

    print(rewards_plot)
    plt.plot(rewards_plot)
    plt.show()

def main():
    # env = gym.make('correlatedbandit-v0')
    env = gym.make('CartPole-v0')
    env.reset()
    num_actions = env.action_space.n
    state_size = env.observation_space.shape[0]
    reward_size = 1
    time_size = 1

    tf.reset_default_graph()
    meta_rl_agent = agent(state_size, num_actions, reward_size, time_size)
    train(meta_rl_agent, env)
    tf.reset_default_graph()
    meta_rl_agent = agent(state_size, num_actions, reward_size, time_size)
    test(meta_rl_agent, env)

    return

if __name__=='__main__':
    main()