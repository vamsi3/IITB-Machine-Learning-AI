class Policy:
    def __init__(self, num_states=1):
        self.num_states = num_states
        self.init_values_policy()

    def init_values_policy(self):
        self.values = [0 for _ in range(self.num_states)]
        self.policy = [-1 for _ in range(self.num_states)]

    def load(self, file):
        with open(file, 'r') as f:
            value_policy_list = f.readlines()[:-1]
        self.num_states = len(value_policy_list)
        self.init_values_policy()
        for state, value_policy in enumerate(value_policy_list):
            self.values[state], self.policy[state] = value_policy.split()
        self.values, self.policy = list(map(float, self.values)), list(map(int, self.policy))

    def __str__(self):
        string = ""
        for state in range(self.num_states):
            string += "{} {}\n".format(self.values[state], self.policy[state])
        return string[:-1]  # remvove the last '\n' character
