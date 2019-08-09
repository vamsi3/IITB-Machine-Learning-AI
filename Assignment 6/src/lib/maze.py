from .mdp import MDP
import random


class Maze:
    def __init__(self):
        self.end = []

    def load(self, file):
        with open(file, 'r') as f:
            self.grid = list(map(lambda line: list(map(int, line.split())), f.readlines()))
        self.height, self.width = len(self.grid), len(self.grid[0])
        for i in range(self.height):
            for j in range(self.width):
                if self.grid[i][j] == 2:
                    self.start = (i, j)
                elif self.grid[i][j] == 3:
                    self.end.append((i, j))

    def index_1d(self, coordinates):
        i, j = coordinates
        return self.height * i + j

    def next_state(self, state, action, index_1d=False):
        i, j = state
        if action == 0:
            i -= 1
        elif action == 1:
            j += 1
        elif action == 2:
            i += 1
        elif action == 3:
            j -= 1

        if i < 0 or i >= self.height or j < 0 or j >= self.width or self.grid[i][j] == 1:
            next_state = state
        else:
            next_state = i, j
        return self.index_1d(next_state) if index_1d else next_state

    def encode_as_MDP(self, p=1):
        mdp = MDP(num_states=self.height * self.width, num_actions=4,
                  discount=1, start=self.index_1d(self.start),
                  end=list(map(lambda state: self.index_1d(state), self.end)))
        for i in range(self.height):
            for j in range(self.width):
                state = (i, j)
                if self.grid[i][j] == 1 or state in self.end:
                    continue
                state_1d = self.index_1d(state)
                next_states_1d = []
                for action in range(4):
                    next_states_1d.append(self.next_state(state, action, index_1d=True))

                random_prob = [1-p if next_states_1d[i] != state_1d else 0 for i in range(4)]
                valid_moves_count = sum(random_prob)
                if valid_moves_count:
                    random_prob = [p / valid_moves_count for p in random_prob]
                for current_action in range(4):
                    for actual_action in range(4):
                        prob = random_prob[actual_action]
                        if actual_action == current_action:
                            prob += p
                        if prob:
                            mdp.transitions[state_1d][current_action][next_states_1d[actual_action]] = (-1, prob)
        return mdp

    def inference(self, policy, p=1):
        state = self.start
        solution = []
        while state not in self.end:
            if random.random() < p:
                action = policy.policy[self.index_1d(state)]
            else:
                valid_actions = []
                for action in range(4):
                    if state != self.next_state(state, action):
                        valid_actions.append(action)
                action = random.choice(valid_actions)
            solution.append(action)
            state = self.next_state(state, action)
        action_code = ['N', 'E', 'S', 'W']
        return list(map(action_code.__getitem__, solution))
