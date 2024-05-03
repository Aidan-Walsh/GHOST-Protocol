# GHOST-Protocol-and-Vulnerabilities
Welcome to my repo on research that I conducted on Ethereum's GHOST protocol. Note that Ethereum's current protocol uses Proof of Stake (POS), and this research does not properly consider this - we just focus on the notion of GHOST. This research attempted to test the capabilities and effectiveness of the Private attack and the Balance attack on the GHOST protocol. Refer to the writeup for more details!

## Private Attack
FinalProjectPrivate.py simulates the private attack on Ethereum's GHOST protocol. The probability that the honest miner mines on the heaviest chain can be tailored, and so can the probability that the adversary mines on its own, private chain (compared to mining on the actual heaviest chain) can be adjusted. 

## Balance Attack
FinalProjectBalance2.py simulates the balance attack on Ethereum's GHOST protocol where the adversary is able to partition the network into two partitions. The adversary's mining power can be adjusted, and the honest miner's probability that it mines on its own chain can be tailored. 
