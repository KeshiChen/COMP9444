import gym
import tensorflow as tf
import numpy as np
import random

# General Parameters
# -- DO NOT MODIFY --
ENV_NAME = 'CartPole-v0'
EPISODE = 200000  # Episode limitation
STEP = 200  # Step limitation in an episode
TEST = 10  # The number of tests to run every TEST_FREQUENCY episodes
TEST_FREQUENCY = 100  # Num episodes to run before visualizing test accuracy

# TODO: HyperParameters
GAMMA =  0.9# discount factor
INITIAL_EPSILON =  0.6# starting value of epsilon
FINAL_EPSILON =  0.1# final value of epsilon
EPSILON_DECAY_STEPS =  100# decay period
HIDDEN_DIM = 32
BATCH_SIZE = 128
REPLAY_BUFFER_SIZE = 1000
REPLAY = []
REPLACE_ITER = 100
LEARN_STEP_COUNTER = 0
# Create environment
# -- DO NOT MODIFY --
env = gym.make(ENV_NAME)
epsilon = INITIAL_EPSILON
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n
#print(STATE_DIM)
# Placeholders
# -- DO NOT MODIFY --
state_in = tf.placeholder("float", [None, STATE_DIM])
action_in = tf.placeholder("float", [None, ACTION_DIM])
target_in = tf.placeholder("float", [None])
#print(np.shape(state_in))
# TODO: Define Network Graph
# eval network
with tf.variable_scope('eval_network'):
    collection_names, w_initializer, b_initializer = \
        ['eval_params', tf.GraphKeys.GLOBAL_VARIABLES], \
        tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers
    with tf.variable_scope('layer1'):
        # Input layer
        weightIn = tf.get_variable(name="weightIn", shape=[STATE_DIM, HIDDEN_DIM], initializer=w_initializer, collections=collection_names)
        biasIn = tf.get_variable(name="biasIn", shape=[1, HIDDEN_DIM], initializer=b_initializer, collections=collection_names)
        layer1 = tf.tanh(tf.matmul(state_in, weightIn) + biasIn) # layer 1
    with tf.variable_scope('layer2'):
        # Output layer
        weightOut = tf.get_variable(name="weightOut", shape=[HIDDEN_DIM, ACTION_DIM], initializer=w_initializer, collections=collection_names)
        biasOut = tf.get_variable(name="biasOut", shape=[1, ACTION_DIM], initializer=b_initializer, collections=collection_names)
# TODO: Network outputs
        q_values = tf.matmul(layer1, weightOut) + biasOut
q_action = tf.reduce_sum(tf.multiply(q_values, action_in), reduction_indices=1)

# TODO: Loss/Optimizer Definition
with tf.variable_scope('loss'):
    loss = tf.reduce_sum(tf.square(target_in - q_action))
with tf.variable_scope('train'):
    optimizer = tf.train.AdamOptimizer().minimize(loss)

# target network
state_next = tf.placeholder(tf.float32, [None, STATE_DIM], name='state_next')

with tf.variable_scope('target_network'):
    collection_names = ['target_params', tf.GraphKeys.GLOBAL_VARIABLES]
    with tf.variable_scope('layer1'):
        weightIn = tf.get_variable(name="weightIn", shape=[STATE_DIM, HIDDEN_DIM])
        biasIn = tf.get_variable(name="biasIn", shape=[1, HIDDEN_DIM], initializer=tf.constant_initializer(0.0))
        layer1 = tf.tanh(tf.matmul(state_in, weightIn) + biasIn) # layer 1
    with tf.variable_scope('layer2'):
        # Output layer
        weightOut = tf.get_variable(name="weightOut", shape=[HIDDEN_DIM, ACTION_DIM])
        biasOut = tf.get_variable(name="biasOut", shape=[1, ACTION_DIM], initializer=tf.constant_initializer(0.0))
        q_values_next = tf.matmul(layer1, weightOut) + biasOut

eval_params = tf.get_collection("eval_params")
target_params = tf.get_collection("target_params")
replace_target_params = [tf.assign(t, e) for t,e in zip(target_params, eval_params)]
# Start session - Tensorflow housekeeping
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())

def updateReplay(state, one_hot_action, reward, next_state, done):
    REPLAY.append((state, one_hot_action, reward, next_state, done))
    if len(REPLAY) > REPLAY_BUFFER_SIZE:
        REPLAY.pop(0)

def batchGeneration(q_values, action_in, state_in, mini_batch):
    target = []
    print(mini_batch[0])
    batch_state = [row[0] for row in mini_batch]
    batch_action = [row[1] for row in mini_batch]
    batch_reward = [row[2] for row in mini_batch]
    next_state_batch = [row[3] for row in mini_batch]
    # batch_q = q_values.eval(feed_dict={
    #     state_in: next_state_batch,
    #     action_in: batch_action
    # })
    # for i in range(0, BATCH_SIZE):
    #     isDone = mini_batch[i][4]
    #     if isDone:
    #         target.append(batch_reward[i])
    #     else:
    #         action = np.max(batch_q[i])
    #         target_val = batch_reward[i] + GAMMA * action
    #         target.append(target_val)
    return target, batch_state, batch_action, batch_reward, next_state_batch

def oneTrainStep(state_in, action_in, q_values, q_values_next): # learn
    global LEARN_STEP_COUNTER
    if checkIfReplace(LEARN_STEP_COUNTER, REPLACE_ITER):
        session.run(replace_target_params)
    minibatch = random.sample(REPLAY, BATCH_SIZE)
    # action_batch: (128,1) reward_batch: (128,1)
    # q_eval: (128,2)
    target_batch, state_batch, action_batch, reward_batch, next_state_batch = batchGeneration(q_values, action_in, state_in, minibatch)
    #print(action_batch)
    q_values_next, q_values = session.run([q_values_next, q_values], feed_dict={state_next: next_state_batch, state_in: state_batch})
    q_target = q_values.copy()
    reward = np.asarray(reward_batch)
    q_target[:,action_batch] = reward + GAMMA * np.max(q_values_next, axis=1)
    #print(q_target)
    _, cost = session.run([optimizer, loss],
                                 feed_dict={state_in: state_batch,
                                            target_in: q_target})
    LEARN_STEP_COUNTER += 1
    return target_batch, state_batch, action_batch

def checkIfReplace(step_counter, replace_iter):
    if step_counter % replace_iter == 0:
        return True

def my_explore(state, epsilon):
    state = state[np.newaxis,:]
    if np.random.uniform() <= epsilon:
        action = np.random.randint(0, ACTION_DIM - 1)  # 随机选择
    else:
        actions_value = session.run(q_values, feed_dict={state_in: state})
        action = np.argmax(actions_value)
    #one_hot_action = np.zeros(ACTION_DIM)
    #one_hot_action[action] = 1
    #return one_hot_action
    return action

# -- DO NOT MODIFY ---
def explore(state, epsilon):
    """
    Exploration function: given a state and an epsilon value,
    and assuming the network has already been defined, decide which action to
    take using e-greedy exploration based on the current q-value estimates.
    """
    Q_estimates = q_values.eval(feed_dict={
        state_in: [state]
    })
    if random.random() <= epsilon:
        action = random.randint(0, ACTION_DIM - 1)
    else:
        action = np.argmax(Q_estimates)
    #one_hot_action = np.zeros(ACTION_DIM)
    #one_hot_action[action] = 1
    #return one_hot_action
    return action


# Main learning loop
for episode in range(EPISODE):

    # initialize task
    state = env.reset()

    # Update epsilon once per episode
    epsilon -= epsilon / EPSILON_DECAY_STEPS

    # Move through env according to e-greedy policy
    for step in range(STEP):
        action = explore(state, epsilon) # get one hot action
        next_state, reward, done, _ = env.step(np.argmax(action))
        if done:
            reward = -1
        updateReplay(state, action, reward, next_state, done) # update replay buffer
        #print(REPLAY)
        nextstate_q_values = q_values.eval(feed_dict={
            state_in: [next_state]
        })
        # TODO: Calculate the target q-value.
        # hint1: Bellman
        # hint2: consider if the episode has terminated
        target = []

        # Do one training step
        if len(REPLAY) > BATCH_SIZE:
            target, state_batch, action_batch = oneTrainStep(state_in, action_in, q_values, q_values_next)
            #print(np.shape(state_batch))
            #print(np.shape([state_batch]))
            #target, state, action = batchGeneration(q_values, action_in, state_in, mini_batch)
            session.run([optimizer, loss], feed_dict={
                target_in: target,
                action_in: action_batch,
                state_in: state_batch
            })

        # Update
        state = next_state
        if done:
            break

    # Test and view sample runs - can disable render to save time
    # -- DO NOT MODIFY --
    if (episode % TEST_FREQUENCY == 0 and episode != 0):
        total_reward = 0
        for i in range(TEST):
            state = env.reset()
            for j in range(STEP):
                #env.render()
                action = np.argmax(q_values.eval(feed_dict={
                    state_in: [state]
                }))
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break
        ave_reward = total_reward / TEST
        print('episode:', episode, 'epsilon:', epsilon, 'Evaluation '
                                                        'Average Reward:', ave_reward)

env.close()
