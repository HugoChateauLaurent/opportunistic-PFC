import numpy as np
import string

class Environment:
    def __init__(self, n_contexts=4, n_tasks=8, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng
        self.n_tasks = n_tasks
        self.W, self.X, self.Y, self.Z = create_vocab(self.n_tasks, 4, rng)
        self.contexts = np.eye(n_contexts)
        self.contexts_dict = {c:self.contexts[i] for i,c in enumerate(list(string.ascii_uppercase)[:n_contexts])}

        self.create_lists()

    def prepare_data(self, context_keys, context_info='auto'):
        if context_info == 'auto':
            context_info = {k:k for k in context_keys}
        parameters = {k: np.tile(self.contexts_dict[context_info[k]], (self.lists[k].shape[0],1)) for k in context_keys}
        context_data = [np.concatenate((self.lists[k], parameters[k]), axis=1) for k in context_keys]
        return np.concatenate(tuple(context_data), axis=0)

    @property
    def n_contexts(self):
        return self.contexts.shape[0]


class NavawongseEnvironment(Environment):

    def create_lists(self):
        self.lists = {}

        self.lists['A'] = []
        self.lists['B'] = []
        self.lists['C'] = []
        self.lists['D'] = []

        for i in range(self.n_tasks):

            correct_idx = self.rng.integers(2)

            # create list for context A
            self.lists['A'].append([self.X[i], self.Y[i], self.X[i] if correct_idx else self.Y[i]])

            # create list for context B
            self.lists['B'].append([self.X[i], self.Y[i], self.X[i] if 1-correct_idx else self.Y[i]])

            # create list for contexts C and D
            for which_list in range(2):
                self.lists['D' if which_list else 'C'].append([None, None, None])
                self.lists['D' if which_list else 'C'][i][which_list] = self.Y[i] if which_list else self.X[i]
                self.lists['D' if which_list else 'C'][i][1-which_list] = self.Z[i]
                self.lists['D' if which_list else 'C'][i][2] = self.lists['D' if which_list else 'C'][i][self.rng.integers(2)]

        for k,v in self.lists.items():
            self.lists[k] = np.array(v)
            self.lists[k] = self.lists[k].reshape((self.lists[k].shape[0],-1))

    @property
    def input_size(self):
        return self.lists['A'].shape[1]





class PetersEnvironment(Environment):

    def create_lists(self):
        self.lists = {}

        self.lists['A'] = []
        self.lists['B'] = []
        self.lists['C'] = []
        self.lists['D'] = []

        for i in range(self.n_tasks):

            correct_idx = self.rng.integers(2)

            # create list for context A
            self.lists['A'].append([self.X[i], self.Y[i], self.X[i] if correct_idx else self.Y[i]])

            # create list for context B
            task = [None, None, None]
            invariant_idx = self.rng.integers(2)
            task[invariant_idx] = self.lists['A'][i][invariant_idx]
            task[1-invariant_idx] = self.W[i]
            task[2] = task[1-invariant_idx] if invariant_idx==correct_idx else task[invariant_idx]
            self.lists['B'].append(task)

            # create list for contexts C and D
            for which_list in range(2):
                self.lists['D' if which_list else 'C'].append([None, None, None])
                self.lists['D' if which_list else 'C'][i][which_list] = self.Y[i] if which_list else self.X[i]
                self.lists['D' if which_list else 'C'][i][1-which_list] = self.Z[i]
                self.lists['D' if which_list else 'C'][i][2] = self.lists['D' if which_list else 'C'][i][self.rng.integers(2)]

        for k,v in self.lists.items():
            self.lists[k] = np.array(v)
            self.lists[k] = self.lists[k].reshape((self.lists[k].shape[0],-1))

    # def check_perseverative(self, target):
    #     if target in self.X or target in self.Y:
    #         return False
    #     elif target in self.W:
    #         return True




        # return target in
        # for task in self.lists['A']:
        #     for i in range(2):
        #         if np.allclose(choice_1,task[i]) or np.allclose(choice_2,task[i]):
        #             return np.allclose(task[i], task[2])
        # print('Error: the choices do not correspond to any List 1 task')

def create_pattern(size, n_active, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    data = np.zeros(size)
    data[rng.choice(size, n_active, replace=False)] = 1
    return data

def create_vocab(n_patterns=8, n_lists=4, rng=None):
    return tuple([[create_pattern(6*4, 6, rng) for i in range(n_patterns)] for j in range(n_lists)])
