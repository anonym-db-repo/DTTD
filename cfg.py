ENV_ID = 'Drone'

RENDER = True  # render while training

# RL training
ALG_NAME = 'SAC_2'
TRAIN_EPISODES = 100000  # total number of episodes for training
TEST_EPISODES = 1  # total number of episodes for training
MAX_STEPS = 1000  # total number of steps for each episode
EXPLORE_STEPS = 500  # 500 for random action sampling in the beginning of training

BATCH_SIZE = 256  # update batch size
HIDDEN_DIM = 64  # size of hidden lay ers for ne

UPDATE_ITR = 3  # repeated updates for single step
SOFT_Q_LR = 3e-4  # q_net learning rate
POLICY_LR = 3e-4  # policy_net learning rate
ALPHA_LR = 3e-4  # alpha learning rate
POLICY_TARGET_UPDATE_INTERVAL = 3  # delayed update for the policy net and target networks
REWARD_SCALE = 1.  # value range of reward
REPLAY_BUFFER_SIZE = 5e4  # size of the replay buffer
OPT_REPLAY_BUFFER_SIZE = 1e4

AUTO_ENTROPY = False  # automatically updating variable alpha for entropy
# DEVICE = "cuda:0"
DEVICE = 'cpu'

IMAGE_VIEW = False

# dynamics
Ix = 0.0820  # kg*m^2
Iy = 0.0845  # kg*m^2
Iz = 0.1377  # kg*m^2
m = 4.34  # kg
g = 9.82  # kg*s^s

# track
test_data_id = 0  # ID of test data
