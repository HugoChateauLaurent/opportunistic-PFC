import numpy as np

import torch
from torch.nn import Linear,ReLU,Sequential,MSELoss,Module
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.optim import RMSprop
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split

from . import utils

class Modulation(object):
    """docstring for Modulation"""
    """0: weight average"""
    """1: weight sum"""
    """2: bias average"""
    """3: bias sum"""
    """4: activity average"""
    """5: activity sum"""
    """6: input"""
    def __init__(self, target_label, mechanism, activation_func=torch.nn.Identity()):
        super(Modulation, self).__init__()
        self.target_label = target_label
        self.mechanism = mechanism
        self.activation_func = activation_func

    @property
    def targets_weights(self):
        return self.mechanism in [0,1]

    @property
    def shape_out(self):
        return self.target.weight.shape if self.targets_weights else (self.target.out_features,)

    @property
    def size_out(self):
        shape_out = self.shape_out
        return shape_out[0] * shape_out[1] if self.targets_weights else shape_out[0]

    def connect_to_target(self, n_contexts, target=None, populations=None):
        if target is None:
            self.target = populations[self.target_label]
        else:
            self.target = target
        self.target.modulations[self.mechanism] = self
        self.module = Population(
            n_contexts,
            self.size_out,
            self.activation_func
        )

    def forward(self, inp):
        return self.module.forward(inp, None, False).reshape(self.shape_out)

class Population(Linear):
    """docstring for Population"""
    def __init__(self, in_features, out_features, activation_func=F.relu, use_bias=True):
        super(Population, self).__init__(in_features, out_features, bias=use_bias)
        self.out_features = out_features
        self.activation_func = activation_func
        self.modulations = {}

    def forward(self, inp, context, muscimol):
        weight, bias = self.modulate_weight_and_bias(context, muscimol)
        x = self.activation_func(F.linear(inp, weight, bias) + self.input_modulation(context, muscimol))
        return self.modulate_activity(x, context, muscimol)

    def modulate_activity(self, activity, context, muscimol):
        if 4 in self.modulations.keys() and not muscimol:
            activity = 1/2 * activity + 1/2 * self.modulations[4].forward(context)

        if 5 in self.modulations.keys() and not muscimol:
            activity = activity.clone()
            activity += self.modulations[5].forward(context)

        return activity

    def modulate_weight_and_bias(self, context, muscimol, use_bias=True):

        weight = self.weight.clone() # We use clone to keep gradient without affecting data
        if 0 in self.modulations.keys() and not muscimol:
            weight = 1/2 * weight + 1/2 * self.modulations[0].forward(context)
        if 1 in self.modulations.keys() and not muscimol:
            weight += self.modulations[1].forward(context)

        if use_bias:
            bias = self.bias.clone() # We use clone to keep gradient without affecting data
            if 2 in self.modulations.keys() and not muscimol:
                bias = 1/2 * bias + 1/2 * self.modulations[2].forward(context)
            if 3 in self.modulations.keys() and not muscimol:
                bias += self.modulations[3].forward(context)

            return weight, bias

        return weight

    def input_modulation(self, context, muscimol):
        if 6 in self.modulations.keys() and not muscimol:
            return self.modulations[6].forward(context)
        else:
            return 0

class Hebbian(Population):
    """docstring for Hebbian"""
    def __init__(self, kappa, lamb, eta, n_hebb, in_features, out_features, activation_func=F.relu):
        super(Hebbian, self).__init__(in_features, out_features, activation_func, use_bias=False)
        self.kappa = kappa
        self.lamb = lamb
        self.eta = eta

        self.n_hebb = n_hebb
        self.weight.data.fill_(0.0)

    def forward(self, inp, context, muscimol):
        weight = self.modulate_weight_and_bias(context, muscimol, use_bias=False)
        h = inp

        for i in range(self.n_hebb):
            h = self.kappa*h + F.linear(h, weight) + self.input_modulation(context, muscimol)
            h = self.activation_func(h)

        return self.modulate_activity(h, context, muscimol)

    def update(self, x):
        self.weight.data = torch.clamp(self.lamb*self.weight.data + self.eta*torch.outer(torch.squeeze(x), torch.squeeze(x)), min=-1, max=1)

class DL(Module):

    def __init__(self, N, n_contexts, modulations, hebbian=None):

        super().__init__()

        self.modulations = modulations

        print("N SI:", N["SI"])

        self.populations = torch.nn.ModuleDict([
            ["ECin",     Population(N["SI"], N["ECin"])],
            ["DG",       Population(N["ECin"], N["DG"])],
            ["CA3",      Population(N["DG"], N["CA3"])],
            ["CA1",      Population(N["CA3"], N["CA1"])],
            ["ECout",    Population(N["CA1"], N["ECout"])],
            ["SIout",    Population(N["ECout"], N["SI"], torch.sigmoid)],
        ])

        hebbian_modulations = []
        for modulation in modulations:
            print("modulation", modulation)
            modulation.connect_to_target(n_contexts, populations=self.populations)
            if modulation.target_label == "CA3" and hebbian is not None:
                hebbian_modulations.append(Modulation("hebbian", modulation.mechanism, modulation.activation_func))
                hebbian_modulations[-1].connect_to_target(n_contexts, target=hebbian)
        modulations.extend(hebbian_modulations)

        self.modulations_modules = torch.nn.ModuleDict([
            [modulation.target_label+" "+str(modulation.mechanism), modulation.module]
        for modulation in modulations])

    def forward(self, SIin, context, hebbian=None, muscimol=False, record_activity=False):

        x = SIin
        activity = {'SIin':x.detach().clone().data[0]} if record_activity else {}

        for label, pop in self.populations.items():

            if label=="CA3" and hebbian is not None:

                # CA3
                x = pop.forward(x, context, muscimol)

                # Always track CA3 output for hebbian learning
                activity[label] = x.detach().clone().data[0]

                # Recurrent CA3
                x = hebbian.forward(x, context, muscimol)
                if record_activity:
                    activity["hebbian"] = x.detach().clone().data[0]

            else:
                x = pop.forward(x, context, muscimol)
                if record_activity:
                    activity[label] = x.detach().clone().data[0]

        return x, activity

class HPC(object):

    def __init__(self, input_size, N_scale, n_contexts, n_hebb, kappa, lamb, eta, lr=.001, summary=True, modulations=None, rng=None):

        self.N = utils.N_scale_to_N(N_scale)
        self.N["SI"] = input_size

        if modulations is None or modulations==0:
            modulations = []
        self.modulations = modulations

        self.kappa = kappa
        self.lamb = lamb
        self.eta = eta

        torch.manual_seed(rng.integers(2**32))
        np.random.seed(rng.integers(2**32))

        print("cuda available:", torch.cuda.is_available())
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng

        self.n_contexts = n_contexts
        self.lr = lr

        self.create_model(summary, n_hebb)

    def create_model(self, summary, n_hebb):

        if n_hebb <= 0:
            self.hebbian = None
        else:
            self.hebbian = Hebbian(self.kappa, self.lamb, self.eta, n_hebb, self.N["CA3"], self.N["CA3"]).to(self.device)


        self.DLnet = DL(self.N, self.n_contexts, self.modulations, self.hebbian).to(self.device)
        self.loss_fn = MSELoss(reduction='mean').to(self.device)


        if summary:
            print(self.DLnet)

    def reset_optimizer(self):
        self.optimizer = RMSprop(self.DLnet.parameters(), lr=self.lr, weight_decay=0)

    def train_init(self, data):

        if self.device != "cpu":
            torch.cuda.empty_cache()

        self.reset_optimizer()
        training_ingredients = [] # Container for all the elements necessary for training initialization

        data = torch.from_numpy(data).float().to(self.device)
        training_ingredients.append(data)

        mask = torch.ones((data.shape[1])).to(self.device)
        mask[24*2:24*3] *= 0 # mask target pattern
        nomask = torch.ones((data.shape[1])).to(self.device)

        training_ingredients.append(mask)
        training_ingredients.append(nomask)

        return tuple(training_ingredients)

    def step(self, data, mask, muscimol, training=True, record_activity=False, shuffle_input_patterns=True):

        self.DLnet.train(mode=training)

        data = data.detach().clone()

        self.optimizer.zero_grad()

        if shuffle_input_patterns and self.rng.random() > .5:
            tmp = data[:,:24].detach().clone()
            data[:,:24] = data[:,24:2*24].detach().clone()
            data[:,24:2*24] = tmp.detach().clone()


        prediction, activity = self.DLnet((data*mask.unsqueeze(0).repeat((data.shape[0],1)))[:,:self.N["SI"]], data[:,self.N["SI"]:], self.hebbian, muscimol, record_activity)
        loss = self.loss_fn(prediction, data[:,:self.N["SI"]])
        R_loss = self.loss_fn(prediction[:,24*2:self.N["SI"]], data[:,24*2:self.N["SI"]])


        if training:
            loss.backward()
            self.optimizer.step()
            if self.hebbian is not None:
                self.hebbian.update(activity["CA3"])

        return loss, R_loss, prediction, activity


    def concurrent_acquisition(
        self,
        data,
        muscimol,
        n_sessions=100,
        criterion=1,
        n_consecutive=2,
        training=True,
        test=True,
        env=None,
        record_training_activity=False,
        record_test_activity=False,
        record_last_only=False,
        max_sessions=np.inf,
        training_as_test="non hebbian"
    ):

        if training_as_test == "non hebbian":
            training_as_test = self.hebbian is None

        data, mask, nomask = self.train_init(data)
        corrects = {"training":[], "test":[]}
        perseveratives = {"training":[], "test":[]}
        R_losses = {"training":[], "test":[]}
        losses = {"training":[], "test":[]}

        activity = {}
        if record_training_activity:
            activity["training"] = {}
        if record_test_activity:
            activity["test"] = {}

        s, criterion_consec = 0, 0
        while s < max_sessions and s < n_sessions: #and  and not criterion_consec >= n_consecutive:

            if s==max_sessions-20:
                print("Limit reached:", s, "sessions")

            s_corrects = {"training":[], "test":[]}
            s_R_losses = {"training":[], "test":[]}
            s_losses = {"training":[], "test":[]}
            s_perseveratives = {"training":[], "test":[]}

            s_activity = {}
            if record_training_activity:
                s_activity["training"] = {"Task":[]}
            if record_test_activity:
                s_activity["test"] = {"Task":[]}

            modes = []
            if training:
                modes.append("training")
            if test and not training_as_test:
                modes.append("test")

            tasks = torch.clone(data)
            shuf_order = list(range(len(tasks)))
            self.rng.shuffle(shuf_order)
            tasks = tasks[shuf_order]

            for task in tasks:
                task = task.unsqueeze(0)
                for mode in modes:
                    record_activity = (mode=="training" and record_training_activity) or (mode=="test" and record_test_activity)
                    loss, R_loss, prediction, step_activity = self.step(task, mask if self.hebbian is None or mode=="test" else nomask, muscimol, mode=="training", record_activity)

                    correct, perseverative = utils.evaluate_performance(prediction, task, self.loss_fn, self.rng, env)

                    for effective_mode in [mode] + (["test"] if mode=="train" and training_as_test else []):
                        if record_activity:
                            s_activity[effective_mode]['Task'].append(task.detach().cpu().numpy()[0])

                            for pop in step_activity.keys():
                                if pop not in s_activity[effective_mode]:
                                    s_activity[effective_mode][pop] = []
                                s_activity[effective_mode][pop].append(step_activity[pop].detach().cpu().numpy())

                        s_corrects[effective_mode].append(correct)
                        s_R_losses[effective_mode].append(R_loss.detach().cpu().numpy())
                        s_losses[effective_mode].append(loss.detach().cpu().numpy())
                        s_perseveratives[effective_mode].append(perseverative)



            # if np.mean(s_corrects) >= criterion:
            #     criterion_consec += 1
            # else:
            #     criterion_consec = 0

            unshuf_order = np.zeros(len(tasks), dtype=int)
            unshuf_order[shuf_order] = np.array(list(range(len(tasks))), dtype=int)

            for mode in ['training','test']:
                corrects[mode].append(np.array(s_corrects[mode])[unshuf_order] if len(s_corrects[mode])>0 else np.array(s_corrects[mode]))
                R_losses[mode].append(np.array(s_R_losses[mode])[unshuf_order] if len(s_R_losses[mode])>0 else np.array(s_R_losses[mode]))
                losses[mode].append(np.array(s_losses[mode])[unshuf_order] if len(s_losses[mode])>0 else np.array(s_losses[mode]))
                perseveratives[mode].append(np.array(s_perseveratives[mode])[unshuf_order] if len(s_perseveratives[mode])>0 else np.array(s_perseveratives[mode]))

                if mode in s_activity.keys():
                    for pop in s_activity[mode].keys():
                        if pop not in activity[mode]:
                            activity[mode][pop] = []
                        activity[mode][pop].append(np.array(s_activity[mode][pop])[unshuf_order])

            s += 1


        return {
            "corrects": {k:np.array(v) for k,v in corrects.items()},
            "perseveratives": {k:np.array(v) for k,v in perseveratives.items()},
            "R_losses": {k:np.array(v) for k,v in R_losses.items()},
            "losses": {k:np.array(v) for k,v in losses.items()},
            "activity": {k:{k2:np.array(v2)[-1 if record_last_only else slice(None)] for k2,v2 in v.items()} for k,v in activity.items()},
        }

    # def blocked_acquisition(self, data, muscimol, n_sessions=100, criterion=np.inf, n_consecutive=10, mode="training"):

    #     corrects, data, mask, loader = self.train_init(data, use_mask=self.hebbian is None or mode=="test")

    #     max_sessions = np.inf # 500

    #     for task in loader:
    #         task_corrects = []
    #         s = 0
    #         while s < n_sessions and s < max_sessions and np.sum(task_corrects[-10:]) < criterion*10:

    #             if s==max_sessions-20:
    #                 print("Limit reached:", s, "sessions")

    #             loss, R_loss, prediction, activity = self.step(task, mask, muscimol, mode)

    #             correct, perseverative = utils.evaluate_performance(prediction, task[0], self.loss_fn, self.rng)
    #             task_corrects.append(correct)

    #         corrects.append(np.array(task_corrects))
    #         s += 1

    #     return np.array(corrects)
