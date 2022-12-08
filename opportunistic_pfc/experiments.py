import pickle
import os
from . import model, utils, environments, parameters
import numpy as np
import shutil
from itertools import chain, combinations

# Avoids problem in PlaFRIM
if "beegfs" not in os.getcwd():
    from tqdm import tqdm

class AbstractExperiment(object):

    def __init__(self, args):
        super(AbstractExperiment, self).__init__()

        self.params = dict({})
        self.exploration = dict({})
        self.name = type(self).__name__+'_'+utils.formatted_time() if args.experiment_id is None else args.experiment_id
        print(self.name)

        self.set_default_parameters()
        self.set_args_parameters(args)

        # For details: https://www.plafrim.fr/
        if self.params["use_plafrim"].default_value:
            from . import plafrim_utils
            self.job_dispatcher = plafrim_utils.JobDispatcher(self.name)

    def add_parameter(self, name, default_value, comment=None):
        assert name not in self.params.keys()
        self.params[name] = parameters.Parameter(name, default_value, comment)

    def set_default_parameters(self):
        self.add_parameter('seed', 1, comment='Seed for the random number generators')
        self.add_parameter('lr', 10**-4, comment='Learning rate')
        self.add_parameter('modulations', None, comment='Contextual modulations')
        self.add_parameter('muscimols', [0,1], comment='Doses of muscimol (0 or 1)')
        self.add_parameter('n_test_sessions', 1, comment='Number of test sessions')
        self.add_parameter('N_scale', 1e-3, comment='Scaling factor for the number of neurons')
        self.add_parameter('msp', False, comment='Whether to use a monosynaptic pathway (i.e. ECin-to-CA1 connection)')
        self.add_parameter('n_hebb', 1, comment='Number of iterations of hebbian recall')
        self.add_parameter('kappa', .5, comment='Hebbian retrieval decay term')
        self.add_parameter('lamb', .5, comment='Hebbian rate of forgetting')
        self.add_parameter('eta', .5, comment='Hebbian rate of remembering')
        self.add_parameter('env', lambda rng: environments.NavawongseEnvironment(n_contexts=2, rng=rng))

    def set_args_parameters(self, args):
        self.add_parameter('args', args, comment='Keep track of all arguments that were passed for reproducibility purposes')
        self.add_parameter('n_train_sessions', args.nsess, comment='Number of training sessions')
        self.add_parameter('use_plafrim', args.plafrim, comment='Whether to use the PlaFRIM computing platform')
        self.add_parameter('record_training_activity', args.record_training_activity, comment='keep track of neural activations during training')
        self.add_parameter('record_test_activity', args.record_test_activity, comment='keep track of neural activations during test')
        self.add_parameter('record_last_only', args.reclast, comment='If any, recording last session only')

    def __call__(self):

        path = 'logs/'+self.name
        os.makedirs(path+"/code")
        shutil.copy("main.py", path+"/code")
        shutil.copytree("opportunistic_pfc", path+"/code"+"/opportunistic_pfc")

        self.setup_exploration()
        self.launch_exploration()

    def setup_exploration(self):
        raise NotImplementedError("Subclasses should implement this!")

    def launch_exploration(self):
        default_params = {p for p in self.params.values() if p.default} # params that will not change
        explorer_params = set(self.params.values()) - default_params # params that will change

        # Compute the number of simulations in the exploration
        if len(explorer_params) > 0:
            exploration_length = [len(p.exploration_values) for p in explorer_params]
            assert np.all(np.array(exploration_length) == exploration_length[0]) # check if all explorer parameters have the same number of exploration values
            exploration_length = exploration_length[0]
        else:
            exploration_length = 1

        iterator = range(exploration_length)
        if not self.params["use_plafrim"].default_value:
            iterator = tqdm(iterator)
        for i in iterator:
            sim_params = dict({}) # Now get parameters for this simulation
            for p in explorer_params:
                sim_params[p.name] = p.exploration_values[i]
            for p in default_params:
                sim_params[p.name] = p.default_value

            # run this simulation
            self.launch_simulation(sim_params, i)

        # For details: https://www.plafrim.fr/
        if self.params["use_plafrim"].default_value:
            self.job_dispatcher.launch_jobs()


    def launch_simulation(self, sim_params, sim_id):

        sim_params["rng"] = np.random.default_rng(sim_params["seed"])
        sim_params["xp_cls"] = self.__class__
        if callable(sim_params["env"]):
            sim_params["env"] = sim_params["env"](sim_params["rng"])

        path = 'logs/'+self.name+'/simulations/sim_'+'{0:07d}'.format(sim_id)+'_' + utils.formatted_time()
        os.makedirs(path)
        pickle.dump(sim_params, open(path+"/params.pickle", "wb"))

        # For details: https://www.plafrim.fr/
        if sim_params["use_plafrim"]:
            from . import plafrim_utils
            self.job_dispatcher.tasks.append(plafrim_utils.Task('logs/'+self.name+'/code', path))
        else:
            self.simulate(path, sim_params)

    @staticmethod
    def simulate(path, sim_params):

        HPC = model.HPC(
            input_size=sim_params["env"].input_size,
            N_scale=sim_params["N_scale"],
            msp=sim_params["msp"],
            n_contexts=sim_params["env"].n_contexts,
            n_hebb=sim_params["n_hebb"],
            kappa=sim_params["kappa"],
            lamb=sim_params["lamb"],
            eta=sim_params["eta"],
            lr=sim_params["lr"],
            summary=False,
            modulations=sim_params["modulations"],
            rng=sim_params["rng"]
        )

        data = sim_params["env"].prepare_data(context_keys=['A','B'])

        results = HPC.concurrent_acquisition(
            data,
            muscimol=0,
            n_sessions=sim_params["n_train_sessions"],
            criterion=np.inf,
            training=True,
            test=True,
            record_training_activity=sim_params["record_training_activity"],
            record_test_activity=sim_params["record_test_activity"],
            record_last_only=sim_params["record_last_only"],
        )

        utils.save_results(path, results)


class DefaultExperiment(AbstractExperiment):
    def __init__(self, args):
        super(DefaultExperiment, self).__init__(args)
    def setup_exploration(self):
        pass

class DebugExperiment(AbstractExperiment):
    def __init__(self, args):
        super(DebugExperiment, self).__init__(args)
    def setup_exploration(self):
        # self.params["env"].default_value = lambda rng: environments.NavawongseEnvironment(n_contexts=2, n_tasks=100, rng=rng)
        # self.params["N_scale"].default_value = .001
        for m in [None]:
            for n_hebb in [0]:#,1]:
                for seed in range(50,55):
                    self.params["modulations"].exploration_values.append(m)
                    self.params["n_hebb"].exploration_values.append(n_hebb)
                    self.params["seed"].exploration_values.append(seed)

class RandomParametersExperiment(AbstractExperiment):
    def setup_exploration(self):
        rng = np.random.default_rng(42) # rng for generating reproducible series of random seeds and parameters
        target_labels = ["ECin", "DG", "CA3", "CA1", "ECout"]
        modulation_combinations = [None]
        for mechanism in [6]:
            target_combinations = chain.from_iterable(combinations(target_labels, n+1) for n in range(len(target_labels)))
            modulation_combinations += [[model.Modulation(target, mechanism) for target in target_group] for target_group in target_combinations]



        for i in range(250):#250):
            s = int(rng.integers(99999999))
            ns = rng.uniform(1e-3, 5e-3)
            lr = 10**(rng.uniform(-5.5, -3))
            k = 1-1*10**(rng.uniform(-3, 0))
            l = 1-1*10**(rng.uniform(-6, 0))
            e = rng.uniform()
            for msp in [False,True]:
                for nh in [0,1]:#[0,1]:
                    for m in modulation_combinations:
                        self.params["msp"].exploration_values.append(msp)
                        self.params["seed"].exploration_values.append(s)
                        self.params["N_scale"].exploration_values.append(ns)
                        self.params["n_hebb"].exploration_values.append(nh)
                        self.params["lr"].exploration_values.append(lr)
                        self.params["kappa"].exploration_values.append(k)
                        self.params["lamb"].exploration_values.append(l)
                        self.params["eta"].exploration_values.append(e)
                        self.params["modulations"].exploration_values.append(m)

class MSPRandomParametersExperiment(RandomParametersExperiment):
    def setup_exploration(self):
        self.params["msp"].default_value = True
        super().setup_exploration()
