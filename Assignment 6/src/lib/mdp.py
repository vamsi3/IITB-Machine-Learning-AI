from collections import defaultdict
from .policy import Policy


class MDP:
    def __init__(self, num_states=1, num_actions=1, discount=1, start=1, end=[]):
        self.num_states = num_states
        self.num_actions = num_actions
        self.discount = discount
        self.start = start
        self.end = end
        self.transitions = defaultdict(lambda: defaultdict(dict))

    def load(self, file):
        with open(file, 'r') as f:
            for line in f:
                line = line.split()
                key, value = line[0], line[1:]
                if key == 'numStates':
                    self.num_states = int(value[0])
                elif key == 'numActions':
                    self.num_actions = int(value[0])
                elif key == 'start':
                    self.start = int(value[0])
                elif key == 'end':
                    self.end = list(map(int, value))
                elif key == 'transition':
                    s1, ac, s2, r, p = line[1:]
                    self.transitions[int(s1)][int(ac)][int(s2)] = (float(r), float(p))
                elif key == 'discount':
                    self.discount = float(value[0])
                else:
                    raise AssertionError("File contains key '{}' which is not expected.".format(key))

    def __str__(self):
        string = ""
        string += "numStates {}\n".format(self.num_states)
        string += "numActions {}\n".format(self.num_actions)
        string += "start {}\n".format(self.start)
        string += "end {}\n".format(' '.join(list(map(str, self.end))))
        for s1 in self.transitions:
            for ac in self.transitions[s1]:
                for s2 in self.transitions[s1][ac]:
                    r, p = self.transitions[s1][ac][s2]
                    string += "transition {} {} {} {} {}\n".format(s1, ac, s2, r, p)
        string += "discount {}".format(self.discount)
        return string

    def ValueIteration(self, eps):
        policy = Policy(self.num_states)
        iteration = 0

        while True:
            iteration += 1
            converged = True
            values_prev = policy.values[:]
            for s in self.transitions:
                if s in self.end:
                    continue
                policy.values[s] = float('-inf')
                for a in self.transitions[s]:
                    value = 0
                    for s_prime in self.transitions[s][a]:
                        r, p = self.transitions[s][a][s_prime]
                        value += p * (r + self.discount * values_prev[s_prime])
                    if value > policy.values[s]:
                        policy.values[s] = value
                        policy.policy[s] = a
                if abs(policy.values[s] - values_prev[s]) > eps:
                    converged = False
            if converged:
                break
        return policy, iteration
