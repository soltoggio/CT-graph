"""CT-graph environments

 Copyright (C) 2019-2021 Andrea Soltoggio, Pawel Ladosz, Eseoghene Ben-Iwhiwhu, Jeff Dick.
"""
import gymnasium as gym
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding
import numpy as np
import logging
import os
import csv
from .CTgraph_images import CTgraph_images

class CTgraphEnv(gym.Env):
    """Main CT-graph class
The graph uses 5 different state types:\n
    # stateTypes:
    # 0: root/home
    # 1: wait state
    # 2: decision state
    # 3: graph end
    # 4: crash

The graph can be configured to be MDP or POMDP. The subsets of observations can be specified in graph.json. Other parameters that can be specified can be found in graph.json
    """
    '''def __init__(self):
        '''

    def __init__(self, conf_data, images):
        print("---------------------------------------------------")
        print("             The CT-graph environments             ")
        print("---------------------------------------------------")

        self.DEPTH = conf_data['graph_shape']['d']
        if self.DEPTH < 1:
            print('Depth must be >= 1, setting it to 1')
            self.DEPTH = 1
        self.BRANCH = conf_data['graph_shape']['b']
        if self.BRANCH < 2:
            print('Branching factor must be at least 2, setting it to 2')
            self.BRANCH = 2
        self.P = conf_data['graph_shape']['p']
        if (self.P < 0) or (self.P) >= 1:
            print('The probability of wait p must be in the range [0,1). Setting parameter to 0')
            self.P = 0

        self.HIGH_REWARD_VALUE = conf_data['reward']['high_r']
        self.CRASH_REWARD_VALUE = conf_data['reward']['fail_r']
        self.REWARD_DISTRIBUTION = conf_data['reward']['reward_distribution']
        #self.STOCHASTIC_SAMPLING = conf_data['reward']['stochastic_d']
        self.REWARD_STD = conf_data['reward']['std_r']
        self.MIN_STATIC_REWARD_EPISODES =  conf_data['reward']['min_static_reward_episodes']
        self.MAX_STATIC_REWARD_EPISODES = conf_data['reward']['max_static_reward_episodes']
        self.oneD = conf_data['image_dataset']['1D']
        self.NR_OF_IMAGES = conf_data['image_dataset']['nr_of_images']

        # observation subsets: there are five subsets
        self.OBS = np.zeros((5,2))
        self.OBS[0] = [0,0] #only one observation for state type home
        self.OBS[1] = np.array(conf_data['observations']['W_IDs'])
        self.OBS[2] = np.array(conf_data['observations']['D_IDs'])
        self.OBS[3] = np.array(conf_data['observations']['graph_ends'])
        self.OBS[4] = [1,1] #only one observation for state type crash
        self.setSizes = np.zeros((5,1))
        for i in range(0,5):
            self.setSizes[i] = self.OBS[i,1] - self.OBS[i,0] + 1

        for i in range(1,4):
            assert (self.OBS[i,0] >= 2), "ERROR: Observations 0 and 1 are reserved for home and crash states. Change graph.json."
            #print('self.OBS[i,1], self.OBS[i+1,0]',self.OBS[i,1], self.OBS[i+1,0])
        '''for i in range(1,3):
            assert self.OBS[i,1] < self.OBS[i+1,0], "ERROR: overlapping observations for different state types. Change graph.json"'''

        assert self.OBS[3,1] < self.NR_OF_IMAGES, "ERROR: Check consistency in json to have a dataset with a suffcent number of images to satisfy settings of subsets."

        self.rnd = np.random.RandomState()
        self.set_seed(conf_data['image_dataset']['seed'])

        self.images = images

        self.MDP_waits = conf_data['observations']['MDP_W']
        self.MDP_decisions = conf_data['observations']['MDP_D']

        self.MDPsize = self.computeMDPsize()
        if self.MDP_waits:
            assert (self.computeMDPsize()[1] <= self.setSizes[1]), "ERROR: There are no enough images in the subset for wait states to be used as states in the MDP. Modify graph.json to increase the subset size."
        if self.MDP_decisions:
            assert (self.computeMDPsize()[2] <= self.setSizes[2]), "ERROR: There are no enough images in the subset for decision states to be used as states in the MDP. Modify graph.json to increase the subset size."


        self.set_seed(conf_data['general_seed'])
        self.set_observation_images(images)
        
        # the number of the decision actions plus one (a0) that is the wait action
        self.action_space = spaces.Discrete(self.BRANCH + 1)
        #Introduce the Observation space of the environment
        '''self.observation_space = gym.spaces.Box(low=0, high=255,
                                   shape=(12, 12),
                                   dtype=np.uint8)'''
        oneDsize = self.computeMDPsize()[0] + 1

        if self.oneD :
            self.observation_space = gym.spaces.Box(low=0, high=1,
                                   shape= (oneDsize, 0),
                                   dtype=np.uint8)
        else:
            self.observation_space = gym.spaces.Box(low=0, high=255,
                                   shape=(12, 12),
                                   dtype=np.uint8)

        self.static_reward_episodes = None # will be set after complete_reset(...) call
        
        self.complete_reset()

        # Call this function to create the CSV file with headers
        self.create_csv_with_headers('high_reward_path_log_b.csv')

        logging.basicConfig(filename='high_reward_path.log', level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        print("---------------------------------------------------")
        print("This instance of CT-graph has\n- %d" % self.DEPTH, "sequential decision state(s)\n- %d" % (self.DEPTH+1), "sequential wait states")
        print("- %d" % pow(self.BRANCH,self.DEPTH), "leaf nodes (ends)")
        print("- %d" % self.computeMDPsize()[0], "total states")
        print("- %d" % self.computeMDPsize()[1], "total wait states")
        print("- %d" % self.computeMDPsize()[2], "total decision points")
        print("---------------------------------------------------")
        # stateTypes:
        # 0: root/home
        # 1: wait state
        # 2: decision state
        # 3: graph end
        # 4: crash
        #return self.images.getNoisyImage(self.X()), 0.0, False, "Root"
        #return self.images.getNoisyImage(self.X())


    def set_observation_images(self, images):
        self.images = images
    
    
    def get_observation_images(self):
        return self.images.getNoisyImage(self.X())

    def computeMDPsize(self):
        """Compute size of the minimal MDP according to equation XX and returns total MDP states, total DP and total DS."""
        size = 0
        for i in range(0,self.DEPTH+1):
            size = size + np.power(self.BRANCH,i)
        MDPsize = (2 * size) + 2
        WaitStatesNr = size

        size = 0
        for i in range(0,self.DEPTH):
            size = size + np.power(self.BRANCH,i)
        DecisionStatesNr = size
        # returning
        return MDPsize, WaitStatesNr, DecisionStatesNr

    def X(self):
        """Stochastic process X that selects one observation from the subsets 0,1,2,3,4 according to stateType"""

        if self.MDP_decisions or self.MDP_waits:
            '''Reconstruct MDP with nr states equal to Eq.(7)
            '''
            if self.stateType == 0:
                return 0 # image 0 is reserved for home
            if self.stateType == 4:
                return 1 # image 1 is researved for crash
            lastBitIdx = self.DEPTH + 1 # this makes an array of size self.BRANCH plus 2, one bit for the stateType and one bit for the one in front
            identifier = np.zeros(lastBitIdx+1).astype(int)
            if self.decision_point_action_counter > 0:
                # setting a one in front of the path bit identifier
                positionOfOneIdx = self.DEPTH -self.decision_point_action_counter
                # if all decisions are taken, positionOfOne is 0
                # one decision is taken, positionOfOne will be
                identifier[positionOfOneIdx] = 1
                for i in range(0,self.decision_point_action_counter):
                    identifier[positionOfOneIdx+i+1] = self.recorded_path[i]
            if self.stateType == 2 or self.stateType == 3:
                identifier[lastBitIdx] = 1
            idxDec = 0
            # converting to decimal: taking all bits except right-most bit that is used instead to select the subset. This way I have 0 to setSize for both sets, as opposed to 0 to sum(setSizes)
            for i in range(0,lastBitIdx):
                idxDec += identifier[i] * np.power(self.BRANCH,lastBitIdx-1-i)
            #    print('identifier[i]', identifier[i])
            #    print('np.power(self.BRANCH,lastBitIdx-1-i)', (np.power(self.BRANCH,lastBitIdx-1-i)))
            #print('idxDec from binary conv ', idxDec)
            # these following lines are insanely complicated: trust them. They return a sequential number 0 to setSizes for stateTypes 1 and 2, then offset them accordingly to fetch the right images in the sets.
            idxDec = max((idxDec-1), 0) % self.setSizes[self.stateType]

            #print("self.setSizes[self.stateType])", ((self.setSizes[self.stateType])))
            #print('idxDec before sum ', idxDec)
            idxDec += self.OBS[self.stateType,0]

            if self.stateType == 1 and not self.MDP_waits:
                idxDec = self.rnd.randint(self.OBS[self.stateType,0],self.OBS[self.stateType,1]+1)
            if self.stateType == 2 and not self.MDP_decisions:
                idxDec = self.rnd.randint(self.OBS[self.stateType,0],self.OBS[self.stateType,1]+1)


            #print('identifier ', identifier)
            #print("Returning observation nr %d" % idxDec)

            return int(idxDec)
            # non-MDP case
        else:
            observation = self.rnd.randint(self.OBS[self.stateType,0],self.OBS[self.stateType,1]+1)
            #print("Returning observation:", observation)
            return observation

    def info(self):
        #return "State: " + str(self.stateType)
        return {"state": str(self.stateType)}

    def reset(self, seed=None, options=None):
        """Set the CT-graph at the root node for a new episode"""
        self.step_counter = 0
        self.stateType = 0
        self.decision_point_action_counter = 0
        self.recorded_path = -np.ones((self.DEPTH,), dtype=int)
        return self.images.getNoisyImage(self.X()), self.info()

    def complete_reset(self):
        """Set the CT-graph at the root node for a newe episode and reset all data from the previous episodes: rwd_accumulator, reward location, and episode_counter"""
        self.rwd_accumulator = 0
        self.reward_static_location_counter = 0
        self.episode_counter = 0
        self.reset_static_reward()
        return self.reset()

    def step(self, action):
        self.step_counter = self.step_counter + 1
        if self.step_counter == 1: # new episode
            self.episode_counter = self.episode_counter + 1
            self.reward_static_location_counter = self.reward_static_location_counter + 1
            if self.reward_static_location_counter == self.static_reward_episodes:
                self.reset_static_reward()

        if self.stateType == 0:
            self.stateType = 1
            #print('>>st:1, wait, img:', self.X())
            return self.images.getNoisyImage(self.X()), 0.0, False, False, self.info()

        if (self.stateType == 1): # wait state
            if action == 0:
                randomNumber = self.rnd.rand()
                if randomNumber < self.P: #remain in wait state
                    #print('>>st:1, wait, img:', self.X())
                    return self.images.getNoisyImage(self.X()), 0.0, False, False, self.info()
                else: # move to decision state or graph end
                    if self.decision_point_action_counter  == self.DEPTH:
                        self.stateType = 3
                        reward = self.calculate_reward()
                        reward_image = self.images.add_reward_cue(self.images.getNoisyImage(self.X()),reward/self.HIGH_REWARD_VALUE)
                        #print('>>st:3, move to END, img:', self.X())
                        return reward_image, reward, False, False, self.info()
                    else:
                        #decision point
                        self.stateType = 2
                        #print('>>st:2, move to DP, img:', self.X())
                        return self.images.getNoisyImage(self.X()), 0.0, False, False, self.info()
            else: # crashing from wait
                self.stateType = 4
                #print('>>st:4, img:', self.X())
                return self.images.getNoisyImage(self.X()), self.CRASH_REWARD_VALUE, True, False, self.info()

        if self.stateType == 2: # decision state
            if action > 0:
                # the path recorded is action - 1 to convert a [1,b] range to a [0,b-1] range that is a more suitable code
                self.recorded_path[self.decision_point_action_counter] = action - 1
                self.decision_point_action_counter += 1
                self.stateType = 1 # going to wait state
                #print('>>st:1, to wait, img:', self.X())
                return self.images.getNoisyImage(self.X()), 0.0, False, False, self.info()
            else:
                self.stateType = 4 # going to crash state
                #print('>>st:1, to crash, img:', self.X())
                return self.images.getNoisyImage(self.X()), self.CRASH_REWARD_VALUE, True, False, self.info()
        if self.stateType == 3:
            # any action gives reward and return home
            self.reset()
            return self.images.getNoisyImage(self.X()), 0.0, True, False, self.info()

        if self.stateType == 4: # at a crash state
            self.stateType = 0 # home state
            return self.images.getNoisyImage(self.X()), 0, True, False, self.info()

    def render(self, mode='human', close=False):
        print('dynamic maze render')

    def set_seed(self, seed):
        self.rnd.seed(seed)

    def set_high_reward_path(self, path):
        """Set the reward at the location specified in the argument."""
        assert len(path) == self.DEPTH, "length of maze array (%d) must be equal to the depth of maze (%d)"%(len(path), self.DEPTH)
        for idx, num in enumerate(path):
            assert num < self.BRANCH, "the numbers in graph array represent the route to the largest reward at each decision point; they must be lower than the number of branches; however the element %d in the graph array (%d) is larger than the number of branches (%d)"%(idx, num, self.BRANCH)
        self.high_reward_path = path


    def create_csv_with_headers(self, filename):
        headers = ['Episode'] + [f"Node {i}" for i in range(1, self.DEPTH + 1)]
        with open(filename, mode='w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(headers)


    def log_to_csv(self, filename, log_data):
        with open(filename, mode='a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(log_data)


    def reset_static_reward(self):
        """Updates reward location and decides when the reward location will change again"""
        if self.MIN_STATIC_REWARD_EPISODES <= 0 or self.MAX_STATIC_REWARD_EPISODES <= 0:
            self.static_reward_episodes = -1 # this disables any task/reward change
        else:
            assert(self.MIN_STATIC_REWARD_EPISODES < self.MAX_STATIC_REWARD_EPISODES)
            self.static_reward_episodes = np.random.randint(self.MIN_STATIC_REWARD_EPISODES, self.MAX_STATIC_REWARD_EPISODES)
        self.reward_static_location_counter = 0
        self.set_high_reward_path(self.get_random_path())
        log_data = [self.episode_counter] + list(self.high_reward_path)
        #print(f"Episode {self.episode_counter}: High reward path changed to {self.high_reward_path}")
        self.log_to_csv('high_reward_path_log_b.csv', log_data)

    def get_random_path(self):
        """Create and return a random location (graph-end)."""
        return self.rnd.randint(0, self.BRANCH, self.DEPTH)

    def get_high_reward_path(self):
        return self.high_reward_path

    def calculate_reward(self):
        """Compute the reward by matching the current state with the location of the high reward."""
        if self.REWARD_DISTRIBUTION == 'needle_in_haystack':
            reward = self.HIGH_REWARD_VALUE * np.floor(1 - np.mean(np.absolute((self.high_reward_path - self.recorded_path)/(self.BRANCH-1))))
        elif self.REWARD_DISTRIBUTION == 'linear':
            weighted_score = np.arange(self.DEPTH,0,-1) * (1 - np.absolute(self.high_reward_path - self.recorded_path))
            print(weighted_score)
            reward = np.sum(weighted_score)/sum(np.arange(self.DEPTH,0,-1)) * self.HIGH_REWARD_VALUE
        if self.REWARD_STD > 0:
            reward = reward + (reward * np.random.normal(0,self.REWARD_STD))
        return reward
    
    def close(self):
        '''if self.window is not None:
            pygame.display.quit()
            pygame.quit()'''
        pass
