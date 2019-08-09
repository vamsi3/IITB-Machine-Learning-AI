import argparse
from lib import Maze, Policy
import random


random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--encoder', action='store_true')
parser.add_argument('--decoder', action='store_true')
parser.add_argument('--grid_file', type=str)
parser.add_argument('--value_and_policy_file', type=str)
parser.add_argument('--probability', type=float, default=1.0)

if __name__ == '__main__':
    args = parser.parse_args()

    maze = Maze()
    maze.load(args.grid_file)

    if args.encoder:
        mdp = maze.encode_as_MDP(args.probability)
        print(mdp)

    if args.decoder:
        policy = Policy()
        policy.load(args.value_and_policy_file)
        solution = maze.inference(policy, args.probability)
        print(' '.join(solution))
