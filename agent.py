import utils
import math
import numpy as np

class Agent:
    def __init__(self, actions, two_sided = False): 
        self.two_sided = two_sided
        self._actions = actions
        self._train = True
        self.paddle_height = 0.2
        self._x_bins = utils.X_BINS
        self._y_bins = utils.Y_BINS
        self._v_x = utils.V_X
        self._v_y = utils.V_Y
        self._paddle_locations = utils.PADDLE_LOCATIONS
        self._num_actions = utils.NUM_ACTIONS
        # Create the Q Table to work with
        self.Q = utils.create_q_table()
        # Q-Learning with TD constants
        self.learning_rate = 1.0
        self.gamma = .99 # discounted factor, how much to weight future rewards hold
        self.min_tries = 3 # tries an action at least this many times for each state before repeating
        self.max_reward = 10 # max reward for any given state
        self.decay_factor = 10
        # book keeping
        self.current_bounces = 0
        self.current_state = None
        self.current_action = 0
        self.state_action_pair_counts = {}
    ###
    # a state has the following:
    # self.ball_x,
    # self.ball_y,
    # self.velocity_x,
    # self.velocity_y,
    # self.paddle_y,
    # self.opponent_y,
    # done = game over
    # won = flag, true is you won
    # bounces, how many times the ball has bounced from your paddle
    
    def act(self, next_state, bounces, done, won):
        # if the program just began, it will choose to remain still
        if self.current_state is None and self._train:
            self.current_state = self.get_discrete_state(next_state)
            self.current_action = 0
            return self._actions[self.current_action]

        if self._train:
            # current state 
            ball_x1, ball_y1, velocity_x1, velocity_y1, paddle_y1 = self.current_state
            # next state
            ball_x2, ball_y2, velocity_x2, velocity_y2, paddle_y2 = self.get_discrete_state(next_state)

            
            index = (ball_x1, ball_y1, velocity_x1, velocity_y1, paddle_y1, self.current_action)
            # increment count of the current state seen by 1
            self.state_action_pair_counts[index] = self.state_action_pair_counts.setdefault(index, 0) + 1

            #Q(s,a)
            q_current = self.Q[ball_x1][ball_y1][velocity_x1][velocity_y1][paddle_y1][self.current_action]

            # a′Q(s′,a′)
            q_next = self.Q[ball_x2][ball_y2][velocity_x2][velocity_y2][paddle_y2]
            # γmaxa′Q(s′,a′)
            q_next_max = self.gamma * np.max(q_next)
            
            # the reward seen from commanding an action, a, from state s.
            reward = self.get_reward(bounces, done, won)

            # calculate learning rate with decay, as more states are seen, the smaller the learning rate is
            alpha = self.learning_rate * (self.decay_factor / (self.decay_factor + self.state_action_pair_counts[index]))
            # Q(s,a)=Q(s,a)+α[R(s)−Q(s,a)+γmaxa′Q(s′,a′)]
            q_updated = q_current + (alpha * (reward - q_current + q_next_max))
            self.Q[ball_x1][ball_y1][velocity_x1][velocity_y1][paddle_y1][self.current_action] = q_updated

            
            # update next state
            self.current_state = (ball_x2, ball_y2, velocity_x2, velocity_y2, paddle_y2)
            
           
            # # actions with exploration
            # if np.random.uniform(0, 1) < 0.01: # will randomly explore 5 percent of the time
            #     action_index = np.random.choice(range(len(self._actions)))
            # else: # explore using Boltzmann Technique
            #     next_actions = self.Q[ball_x2][ball_y2][velocity_x2][velocity_y2][paddle_y2]
            #     next_actions_exp = np.exp(next_actions)
            #     probs = next_actions_exp / np.sum(next_actions_exp)
            #     action_index = np.random.choice(range(len(self._actions)), p=probs)

            next_actions = self.Q[ball_x2][ball_y2][velocity_x2][velocity_y2][paddle_y2]
            exploration =  self.calculate_exploration(next_actions)

            best_action = np.argmax(exploration)
            self.current_action = best_action
            #self.current_action = action_index
            
            return self._actions[best_action]
        else: # evaluation
            ball_x2, ball_y2, velocity_x2, velocity_y2, paddle_y2 = self.get_discrete_state(next_state)
            next_actions = self.Q[ball_x2][ball_y2][velocity_x2][velocity_y2][paddle_y2]
            best_action = np.argmax(next_actions)
            return self._actions[best_action]

    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    def save_model(self,model_path):
        # At the end of training save the trained model
        print("discount {}, decay {}, min_tries {}, max_reward {}".format(self.gamma, self.decay_factor, self.min_tries, self.max_reward))
        utils.save(model_path,self.Q)

    def load_model(self,model_path):
        # Load the trained model for evaluation
        self.Q = utils.load(model_path)

    def get_discrete_state(self, state):
        ball_x, ball_y, velocity_x, velocity_y, paddle_y = state

        ball_x = math.floor(ball_x * self._x_bins)
        # removes the extra state
        if ball_x >= self._x_bins:
            ball_x = self._x_bins - 1

        ball_y = math.floor(ball_y * self._y_bins)
        if ball_y >= self._y_bins:
            ball_y = self._y_bins - 1

        paddle_y = math.floor(self._paddle_locations * paddle_y / (1 - self.paddle_height))
        if paddle_y >= self._paddle_locations:
            paddle_y = self._paddle_locations - 1

        # Discretize the X-velocity of the ball to have only 2 (V_X) 
        # possible values: +1 or -1 (the exact value does not matter, only the sign).
        if velocity_x > 0:
            velocity_x = 1
        else:
            velocity_x = -1
        
        # Discretize the Y-velocity of the ball to have only 3 (V_Y) 
        # possible values: +1, 0, or -1. It should map to Zero if |velocity_y| < 0.015.
        if abs(velocity_y) < 0.015:
            velocity_y = 0
        elif velocity_y > 0:
            velocity_y = 1
        else:
            velocity_y = -1
        
        return ball_x, ball_y, velocity_x, velocity_y, paddle_y
        
    def get_reward(self, bounces, done, won):
        # checks if a bounce was made
        if bounces > self.current_bounces:
            self.current_bounces += 1
            return 5
        # if the game is over, just for part 1
        # part 2 can adjust the reward by checking if won
        if done:
            self.current_bounces = 0
            return -10
        return 0
    
    def calculate_exploration(self, next_actions):
        ball_x, ball_y, velocity_x, velocity_y, paddle_y = self.current_state
        for i in range(len(self._actions)):
            index = (ball_x, ball_y, velocity_x, velocity_y, paddle_y, i)
            pair_count = self.state_action_pair_counts.setdefault(index, 0)
            if pair_count < self.min_tries:
                next_actions[i] = self.max_reward
        return next_actions