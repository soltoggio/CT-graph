"""
CT-graph.
 Copyright (C) 2019-2021 Andrea Soltoggio, Pawel Ladosz, Eseoghene Ben-Iwhiwhu, Jeff Dick.

Launch script to test one single navigation episode in automatic mode and manual mode"""
from ast import Or
import numpy as np
import gym
from gym_CTgraph import CTgraph_env
from gym_CTgraph.CTgraph_plot import CTgraph_plot
from gym_CTgraph.CTgraph_conf import CTgraph_conf
from gym_CTgraph.CTgraph_images import CTgraph_images
import argparse
import json
import random
import matplotlib.pyplot as plt
import timeit
from keras.utils.np_utils import to_categorical
from itertools import permutations

def printout(p_obs, p_reward, p_act, p_done, p_info, p_counter):
    """Print out the navigation variable at each step"""
    print("Feeding action: ", p_act)
    print("Step:", p_counter)
    print("Observation: ", p_obs)
    print("Obs Type", type(p_obs))
    print("Reward: ", p_reward)
    print("Done: ", p_done)
    print("--\nInfo: ", p_info)

#what is this parser?
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-c', '--case', default=1, dest="CASE",
                    help='exectution mode')

args = parser.parse_args()

# fetch the parameters from the json file
configuration = CTgraph_conf("graphTT2.json") #graphT2.json
conf_data = configuration.getParameters()
# print configration data
print(json.dumps(conf_data, indent=3))

# instantiate the maze
start = timeit.timeit()
env = gym.make('CTgraph-v0')
end = timeit.timeit()
#print(end - start)

imageDataset = CTgraph_images(conf_data)

# initialise and get initial observation and info
'''ERROR: to many values to unpuck.'''
#observation, reward, done, info = env.init(conf_data, imageDataset)

#print(np.shape(imageDataset))
print(imageDataset.image)

'''imageDatasetList = []

for image_nr in range(imageDataset.nrOfImages()):
            imageDatasetList.append(imageDataset.getNoisyImage(image_nr))

obsTemp0 = imageDatasetList[8]
obsTemp = imageDatasetList[6]
imageDatasetList[6] = imageDatasetList[4]
imageDatasetList[4] = obsTemp
imageDataset.image = imageDatasetList
imageDatasetList[8] = imageDatasetList[2]
imageDatasetList[2] = obsTemp0'''

#print("Permuted image dataset:\n", imageDataset.image)
#print("HIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII:", imageDataset)
print()
print()
#image = list(permutations(imageDatasetList))
print()
print()
#print(permutations_imgs_pos)

print()
print()
observation = env.init(conf_data, imageDataset)
reward = 0.
done = False
info = {}
#print("SHAPE OBS:", np.shape(observation))
#print(observation)

#plotting: uncomment the following line to plot the observations
#CTgraph_plot.plotImages(imageDataset, False)

# get a random path from the maze
#high_reward_path = env.get_random_path()
# use this random path to set the path to the high reward. Note that the maze would have already a high_reward_path from the initialisation
env.set_high_reward_path(np.array([0, 0]))

print("*--- Testing script ----*")

action = 0
counter = 0
print_results = True

print("Type of Obs Space:", type(env.observation_space))

action_space = np.array([0, 1, 2])
action_space_onehot = to_categorical(action_space, num_classes = (len(action_space)))
action_space_onehot = action_space_onehot.astype('int')
action_space_onehot = action_space_onehot.tolist()
print("Action Space:", action_space)
print("One-Hot Action Space:", action_space_onehot)

env.reset()
CASE = int(args.CASE)
#interactive case: step-by-step with operator inputs
if CASE == 0:
    observation = env.complete_reset()
    print("OOOOOOOOOOOOOOOOOOOOBS TYPE:", type(observation))
    print("The test script sets the high reward path to: ", env.get_high_reward_path())
    printout(observation, reward, action, done, info, counter)

    start = timeit.timeit()
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(3, 5))
    #plt.figure(figsize=(2,2))
    axs.imshow(observation)
    plt.show(block=False)
    while not done:
        action = int(input("Action: "))
        observation, reward, done, info = env.step(action)
        counter = counter  + 1
        if print_results:
            printout(observation, reward, action, done, info, counter)
            axs.imshow(observation)
            plt.draw()
            plt.show(block=False)
    print("close images to end")
    plt.show(block=True)

#automated: for testing many episodes
if CASE == 1:
    #tesing high rewards
    total_reward = 0
    total_time_steps = 0
    nr_episodes = 17000
    probDelayCrash = 0.0
    probDecisionPointCrash = 0.0
    probWrongDecision = 0.0
    tranjectory = []

    for test in range(0,nr_episodes):
        done = False
        #observation, reward, done, info = env.complete_reset()
        observation = env.complete_reset()
        #high_reward_path = env.get_random_path()
        #env.set_high_reward_path(high_reward_path) #get_high_reward_path()
        high_reward_path = env.get_high_reward_path()
        index_decision_point_actions = 0
        print("E:%d" % test, end='')
        print(" testing path:", env.get_high_reward_path(), end='\n')
        print(env.info())
        print("HELLO THERE!!!")
        while not done:
            total_time_steps = total_time_steps + 1
            print("Hello THERE!!!!!")
            print(env.info())
            # check if I'm in a delay or root stateType
            if "1" in env.info().values() or "0" in env.info().values():
                action = 0
                if random.random() < probDelayCrash:
                    action = np.random.randint(1,env.BRANCH+1)
                print('x%d' % env.step_counter, end='')
                observation, reward, done, info = env.step(action)
                print(observation)
                print(type(observation))
                print("'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''")
                flat_obs = [pixel for row in observation for pixel in row]
                a = int(action)
                oh_a = action_space_onehot[a]
                r = [int(reward)]
                sar = flat_obs + oh_a + r
                print("Action:", a)
                print(sar)
                print()
                if len(tranjectory) < 50000:
                    tranjectory.append(sar)
                total_reward = total_reward + reward

            if "2" in env.info().values():
                # correct action
                action = high_reward_path[index_decision_point_actions] + 1
                print("Action:", action)
                if random.random() < probDecisionPointCrash: #do something wrong with a small prob
                    action = 0

                if random.random() < probWrongDecision: #do something wrong with a small prob, cycling through actions
                    action = action % (env.BRANCH) + 1
                print('(a:%d)' % action, end='')
                observation, reward, done, info = env.step(action)
                print(observation)
                print(type(observation))
                print("'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''")
                flat_obs = [pixel for row in observation for pixel in row]
                #flat_obs = [pixel for row in observation for pixel in row]
                a = int(action)
                oh_a = action_space_onehot[a]
                r = [int(reward)]
                sar = flat_obs + oh_a + r
                print("Action:", a)
                print(sar)
                #print([pixel for row in observation for pixel in row], action, reward,)
                if len(tranjectory) < 50000:
                    tranjectory.append(sar)
                index_decision_point_actions = index_decision_point_actions + 1
                total_reward = total_reward + reward

            if "3" in env.info().values():
                print("-E, R:%0.1f" % reward ," in %d" % env.step_counter, "steps")
                observation, reward, done, info = env.step(0)
                print(observation)
                print(type(observation))
                print("'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''")
                flat_obs = [pixel for row in observation for pixel in row]
                a = int(action)
                oh_a = action_space_onehot[a]
                r = [int(reward)]
                sar = flat_obs + oh_a + r
                print("Action:", a)
                print(sar)
                if len(tranjectory) < 50000:
                    tranjectory.append(sar)
                total_reward = total_reward + reward

            if "4" in env.info().values():
                print("Crash at step", env.step_counter, end='\n')
    print("total reward: ", total_reward, "total timesteps:", total_time_steps)
    
    import pandas as pd
    df = pd.DataFrame(tranjectory)
    print()
    #print(tranjectory)
    print("---------------------------------------------------------")
    print(df)
    np.save('CT-I2R1P2b.npy',tranjectory)
    print("The test script sets the high reward path to: ", env.get_high_reward_path())
    #print(df[144].unique())
    df.to_csv('CT-I2R1P2b.csv')
    np.savetxt("CT-I2R1P2b.txt", 
           df.values,
           fmt='%d')
    
