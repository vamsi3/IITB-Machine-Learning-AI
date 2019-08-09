import argparse
from lib import MDP


parser = argparse.ArgumentParser()
parser.add_argument('--mdp_file', type=str)

if __name__ == '__main__':
    args = parser.parse_args()
    mdp = MDP()
    mdp.load(args.mdp_file)
    policy, iterations = mdp.ValueIteration(eps=1e-16)
    print(policy)
    print("iterations {}".format(iterations))
