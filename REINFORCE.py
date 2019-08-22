import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.misc import imresize
from Model import ModelCatchGame, ModelSnakeGame

IMAGE_SIZE = 84
LEARNING_RATE = 1e-3
GAMMA = 0.9 # discount factor
INPUT_SIZE = IMAGE_SIZE ** 2
EPISODES_TO_TRAIN = 32

class ReinforceAgent():
  
    def __init__(self, env, model, num_actions):
        self.env = env
        self.name_model = model
        self.num_actions = num_actions
        self.model = self.build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

    def build_model(self):
        if self.name_model == "Catch":
            return ModelCatchGame(INPUT_SIZE, self.num_actions)
        if self.name_model == "Snake":
            return ModelSnakeGame(self.num_actions)
        return None

    # Preprocessing original image (400,400) to (84,84)
    def preprocess_image_catch(self, image):
        
        # single image
        x_t = image
        x_t = imresize(x_t, (IMAGE_SIZE, IMAGE_SIZE, 1)) 
        x_t = x_t.astype("float")
        x_t /= 255.0

        x_t = np.expand_dims(x_t, axis=0)

        return np.reshape(x_t, (1, IMAGE_SIZE*IMAGE_SIZE*1)).squeeze()

    # Preprocessing original image to (84,84,3)
    def preprocess_image_snake(self, image):
        
        # single image
        x_t = image
        x_t = imresize(x_t, (IMAGE_SIZE, IMAGE_SIZE, 3)) 
        x_t = np.transpose(x_t,(2,0,1))
        x_t = x_t.astype("float")
        x_t /= 255.0

        x_t = np.expand_dims(x_t, axis=0)

        return x_t
        
    def calc_qvals(self, rewards):
        res = []
        sum_r = 0.0
        for r in reversed(rewards):
            sum_r *= GAMMA
            sum_r += r
            res.append(sum_r)
        return list(reversed(res))
        
    # Picking an action stochastically
    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        logits = self.model(state)
        actions_probabilities = F.softmax(logits).cpu().detach().numpy().squeeze()
        action = np.random.choice(self.num_actions, 1, p=actions_probabilities)[0]

        return action
      
    def generate_experience(self):
        experience = self.execute_one_episode()
        steps = 0
        states = []
        actions = []
        rewards = []
        
        for exp in experience:
            states.append(exp[0])
            actions.append(int(exp[1]))
            rewards.append(exp[2])
            steps += 1
          
        return [states, actions, rewards], steps
        
    def execute_one_episode(self):
        self.env.reset()
        state = self.env.get_current_frame()
        game_over = False
        experiences = []
        
        while not game_over:
            if self.name_model == 'Catch':
                state = self.preprocess_image_catch(state)
            if self.name_model == 'Snake':
                state = self.preprocess_image_snake(state)
            action = self.get_action(state)
            new_state, reward, game_over = self.env.step(action)
            if game_over:
                new_state = None
            experience = tuple([state, action, reward])
            experiences.append(experience)
            state = new_state
            
        return experiences
        
    def train(self):
      
        total_rewards = []
        step_idx = 0
        done_episodes = 0

        batch_episodes = 0
        batch_states, batch_actions, batch_qvals = [], [], []
        cur_rewards = []

        while True:
            
            # Generates a new episode
            exp_source, num_steps = self.generate_experience()
            
            batch_states.extend(exp_source[0])
            batch_actions.extend(exp_source[1])
            cur_rewards.extend(exp_source[2])
            step_idx += num_steps

            batch_qvals.extend(self.calc_qvals(cur_rewards))
            batch_episodes += 1
                
            reward = np.sum(cur_rewards)
            cur_rewards.clear()

            done_episodes += 1
            total_rewards.append(reward)
            mean_rewards = float(np.mean(total_rewards[-100:]))
            print("%d: reward: %6.2f, mean_100: %6.2f, episodes: %d" % (
                step_idx, reward, mean_rewards, done_episodes))
            if mean_rewards > 0.9:
                print("Solved in %d steps and %d episodes!" % (step_idx, done_episodes))
                break
                
            if batch_episodes < EPISODES_TO_TRAIN:
                continue

            self.optimizer.zero_grad()
            states_v = torch.FloatTensor(batch_states).squeeze()
            batch_actions_t = torch.LongTensor(batch_actions)
            batch_qvals_v = torch.FloatTensor(batch_qvals)
            print(states_v.shape)
            logits_v = self.model(states_v)
            log_prob_v = F.log_softmax(logits_v, dim=1)
            log_prob_actions_v = batch_qvals_v * log_prob_v[range(len(batch_states)), batch_actions_t]
            loss_v = -log_prob_actions_v.mean()

            loss_v.backward()
            self.optimizer.step()

            batch_episodes = 0
            batch_states.clear()
            batch_actions.clear()
            batch_qvals.clear()