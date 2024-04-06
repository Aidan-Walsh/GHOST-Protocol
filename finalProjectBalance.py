import numpy as np
import math
import matplotlib.pyplot as plt
from collections import namedtuple
import random

# in context of ethereum

# partitioning power of adversary
# assume that only 2 partitions exist - more can exist
# with partitioning, paritioned parties cannot see each other 
# adversaries are not mining in this because this is a function of partitioning power 
# can add modification where adversaries can also add blocks of their own


class Node:
    def __init__(self, parent, children, weight, partition):
        self.parent = parent
        self.children = children
        self.weight = weight
        self.partition = partition


class Worker:
    def __init__(self, partition, probability, mining_node):
        self.partition = partition # either 0 or 1
        self.probability = probability
        self.mining_node = mining_node


Genesis0 = Node(parent=None, children=[], weight=1, partition = 0)
Genesis1 = Node(parent=None, children=[], weight=1, partition = 1)


def update_weights(node):
    curr_node = node
    while True: 
      if (curr_node.parent == None): 
          break
      curr_node.parent.weight += 1
      curr_node = curr_node.parent



def update_tip(node): # fix this
    
    curr_node = node 
    while True: 
        if len(curr_node.children) == 0: 
            return curr_node
        max_weight = curr_node.children[0].weight
        max_node = curr_node.children[0]
        for i in range(1, len(curr_node.children)):
          if max_weight < curr_node.children[i].weight:
            max_weight = curr_node.children[i].weight
            max_node = curr_node.children[i]
        curr_node = max_node
            




def chain_quality(node):
    curr_node = node
    good_count = 0.0
    total_count = 0.0
    while True:
        if curr_node.parent == None:
            break
        if (curr_node.advers):
            total_count += 1.0
        else:
            total_count += 1.0
            good_count += 1.0
        curr_node = curr_node.parent
    print(total_count)
    return good_count/total_count
        



lam = 1

total_miners = 100





for p in [0,50,100]:  # probability that miner changes over to new block
    x = []
    y = []
    for partition_power in range(0, 101):
        x_val = partition_power
        miners = []
        
        Genesis0 = Node(parent=None, children=[], weight=1, partition = 0)
        Genesis1 = Node(parent=None, children=[], weight=1, partition = 1)      

        block_count = 1
        true_tip = 0 # 0 means tip belongs to Genesis0
        partition0_tip = Genesis0
        partition1_tip = Genesis1
        hash_power = 1 / total_miners
        switches = 0
        # create miners in different partitions
        for i in range(0, total_miners):
            if i < (partition_power/100) * total_miners:
                miners.append(
                    Worker(partition = 0, probability=0.0, mining_node=Genesis0)
                )
            else:
                miners.append(
                    Worker(partition = 1, probability=0.0, mining_node=Genesis1)
                )
            miners[i].probability = np.random.exponential(1 / ((hash_power) * lam))

        while block_count < 1000:
            miners.sort(key=lambda x: x.probability)
            min_prob = 0
            tip0_updated = False
            tip_updated = False
            tip1_updated = False
            for i in range(0, len(miners)):
                if (miners[i].partition == 0):
                    if i == 0:
                        min_prob = miners[i].probability
                        miners[i].probability = np.random.exponential(
                            1 / ((hash_power) * lam)
                        )
                        new_node = Node(
                            parent=miners[i].mining_node,
                            weight=1,
                            partition =0,
                            children=[],
                        )

                        miners[i].mining_node.children.append(
                            new_node
                        )  # create new block
                        update_weights(new_node)
                        old0_tip = partition0_tip
                        partition0_tip = update_tip(Genesis0)
                        if old0_tip != partition0_tip:
                            tip0_updated = True
                            if ((Genesis0.weight > Genesis1.weight and true_tip == 1) or (Genesis1.weight == Genesis0.weight)):
                                true_tip = 0
                                switches += 1
                            else: 
                                if (Genesis0.weight > Genesis1.weight):
                                    true_tip = 0
                    


                        
                        block_count += 1
                    else:
                        # now go through and use probability that miners moves over to new block
                        random_value = random.randint(1, 100)
                        if tip0_updated and random_value <= p: 
                            partition0_tip = update_tip(Genesis0)
                            miners[i].mining_node = partition0_tip

                            miners[i].probability = np.random.exponential(
                                1 / ((hash_power) * lam)
                            )
                        else:
                            miners[i].probability -= min_prob
                else:  # miner in other partition, essentially same but they mine off each other
                   if i == 0:
                        min_prob = miners[i].probability
                        miners[i].probability = np.random.exponential(
                            1 / ((hash_power) * lam)
                        )
                        new_node = Node(
                            parent=miners[i].mining_node,
                            weight=1,
                            partition =1,
                            children=[],
                        )

                        miners[i].mining_node.children.append(
                            new_node
                        )  # create new block
                        update_weights(new_node)
                        old1_tip = partition1_tip
                        partition1_tip = update_tip(Genesis1)
                        if old1_tip != partition1_tip:
                            tip1_updated = True
                            if ((Genesis1.weight > Genesis0.weight and true_tip == 0) or Genesis1.weight == Genesis0.weight):
                             
                                true_tip = 1
                                switches += 1
                            else: 
                                if (Genesis1.weight > Genesis0.weight):
                                    true_tip = 1
                         


                        
                        block_count += 1
                   else:
                        # now go through and use probability that miners moves over to new block
                        random_value = random.randint(1, 100)
                        if random_value <= p and tip1_updated:
                            partition1_tip = update_tip(Genesis1)
                            miners[i].mining_node = partition1_tip

                            miners[i].probability = np.random.exponential(
                                1 / ((hash_power) * lam)
                            )
                        else:
                            miners[i].probability -= min_prob

        x.append(x_val)
        print(switches/block_count)
        y.append(switches/block_count)

    plt.plot(x, y, label="P = " + str(p / 100.0))

    # adversary starts off not working
    # pass units of time
plt.title("Tip Conflicts versus Adversary Partition power in GHOST")
plt.xlabel("Adversary Partition Power (%)")
plt.ylabel("Tip Conflicts (switches/epochs)")
plt.grid(axis = 'y')
plt.legend()
plt.show()


# update weights


# adv_prob = np.random.exponential(1 / ((beta) * lam))
# miner_prob = np.random.exponential(1 / ((1 - beta) * lam)) #lesser one typically has higher probability


# for balance can do throughput as adversarial hash power changes
# can also do number of swings per epoch as adversarial hash power changes or as adverarial partition power changes
