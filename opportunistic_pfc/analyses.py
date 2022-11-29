import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from . import utils, constants

import os

# Avoids problem in PlaFRIM
if "beegfs" not in os.getcwd():
    import seaborn as sns

from scipy.stats import sem, gaussian_kde, mannwhitneyu, wilcoxon, spearmanr, pearsonr, normaltest, ttest_ind

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class AbstractAnalysis(object):

    def __init__(self, args):
        super(AbstractAnalysis, self).__init__()
        self.xp_id = args.experiment_id

    @property
    def xp_path(self):
        return "./logs/"+self.xp_id

    @property
    def processed_results_path(self):
        return self.xp_path+"/processed_results.df"

    @property
    def figures_path(self):
        return self.xp_path+"/figures"

    @property
    def simulations_path(self):
        return self.xp_path+"/simulations"

    def __call__(self):
        os.makedirs(self.figures_path, exist_ok=True)
        self.prepare_df()
        self.analyse()

    def prepare_df(self):

        # If dataframe file already created, load it
        if os.path.isfile(self.processed_results_path):
            self.results = pickle.load(open(self.processed_results_path, 'rb'))
            print("loading")

        else:
            simulation_dirs = [self.simulations_path+'/'+simulation_dir for simulation_dir in os.listdir(self.simulations_path)]
            self.results = [] # temporary list to store the data to be stored in the dataframe

            for d in simulation_dirs:

                params = pickle.load(open(d+"/params.pickle",'rb'))
                data = pickle.load(open(d+"/results.pickle",'rb'))

                data = utils.flatten_dict(data)

                for k in list(data.keys()):

                    # data[k] = np.array(data[k])

                    # Should be removed for new simulations with bug fixed
                    if 'test' in k and params['n_hebb']==0:
                        data[k] = data[k.replace('test','training')]

                    if 'corrects' in k:
                        # print(data[k].shape)
                        data['_'.join(['mean',k])] = np.mean(data[k][:100]) if len(data[k])>0 else None

                # Add params to data
                for k,v in params.items():
                    data[k] = v


                if params["modulations"] is None:
                    data["mechanisms"] = -1
                    data["target_labels"] = (-1,)
                    data["n_targets"] = 0
                    for target_label in constants.MODULAR_LAYERS:
                        data[target_label] = 0
                else:
                    assert (np.array([params["modulations"][i].mechanism for i in range(len(params["modulations"]))])==params["modulations"][0].mechanism).all() # For now, only one mechanism is used at a time
                    data["mechanisms"] = params["modulations"][0].mechanism # For now, only one mechanism is used at a time
                    data["target_labels"] = tuple([params["modulations"][i].target_label for i in range(len(params["modulations"]))])
                    data["n_targets"] = len(data["target_labels"])

                    for target_label in constants.MODULAR_LAYERS:
                        data[target_label] = int(target_label in data["target_labels"])

                self.results.append(data)

            self.results = pd.DataFrame(self.results)

            # Save
            pickle.dump(self.results, open(self.processed_results_path, "wb"))

    def analyse(self):
        raise NotImplementedError("Subclasses should implement this!")



class Analysis(AbstractAnalysis):
    def analyse(self):
        for n_hebb in [1,0]:

            ##############################
            # n_targets plot
            ##############################

            plot_df = self.results[(self.results['n_hebb']==n_hebb) & (self.results['mechanisms']==6)]
            plot_df["n_targets"] = plot_df["n_targets"].astype(int).astype(str)
            order = [str(i) for i in range(1,int(plot_df["n_targets"].max()) + 1)]
            x='n_targets'
            y='mean_test_corrects'
            plt.figure(figsize=(3,3))
            flierprops = dict(marker='o', markersize=2)
            ax = sns.boxplot(
                data=plot_df,
                x=x,
                y=y,
                showfliers=True,
                order=order,
                flierprops=flierprops)
            xlim = ax.get_xlim()
            plt.plot([-10,10], [50 for i in range(2)], linestyle='--', color='k', alpha=.2, zorder=-1)
            ax.set_xlim(xlim)

            # Annotations for statistical significance

            spearman_df = plot_df
            spearmanres = spearmanr(spearman_df['n_targets'].astype(str).astype(int), spearman_df[y], alternative="greater")
            print("N:", len(spearman_df.index), "|", spearmanres)

            pairs = [("1","2"), ("2","5")]#[("0","1")]
            if len(pairs)>0:
                # print("0 better than chance:", wilcoxon(
                #     plot_df[plot_df[x]=="0"][y].to_numpy()-50,
                #     alternative="greater"
                # ))
                stattest = {
                    p:mannwhitneyu(
                        plot_df[plot_df[x]==p[0]][y],
                        plot_df[plot_df[x]==p[1]][y],
                        alternative="less"
                    ) for p in pairs}

                for p,v in stattest.items():
                    for i in range(2):
                        print(p[i], "N:", len(plot_df[plot_df[x]==p[i]][y].index), "Median:", plot_df[plot_df[x]==p[i]][y].median())
                    print(v)
                    print("\n\n")
                # annotator = Annotator(ax, pairs, data=plot_df, x=x, y=y, order=order)
                # annotator.configure(text_format="star", loc="inside")
                # annotator.set_pvalues_and_annotate([v.pvalue for v in stattest.values()])

            ax.set_xlim(xlim)
            ax.set_xlabel("Number of modulated layers")
            ax.set_ylabel("Performance (% correct)")
            fig = ax.get_figure()
            fig.tight_layout()
            utils.make_fig(fig, ax, self.figures_path, "ntargets"+"_nhebb_"+str(n_hebb))

            # # fig, ax = plt.subplots()
            # plot_df = df[(df['n_hebb']==n_hebb) & (df['mechanisms']==6) & (df['n_targets']>0)]
            # ax = sns.regplot(data=plot_df, x=x, y=y, x_estimator=np.mean)
            #
            # # corr_dfs = [df[(df['n_hebb']==n_hebb) & (df['mechanisms']==6) & (df['n_targets']>min) & (df['n_targets']<max)] for min,max in [(0,4), (2,6)]]
            # # print(spearmanr(corr_dfs[0][x], corr_dfs[0][y], alternative='greater'))
            # # print(spearmanr(corr_dfs[1][x], corr_dfs[1][y], alternative='less'))
            # print(spearmanr(plot_df[x], plot_df[y], alternative='greater'))
            #
            # reg = LinearRegression().fit(plot_df[x].to_numpy()[:,None], plot_df[y])
            # print(reg.score(plot_df[x].to_numpy()[:,None], plot_df[y]))
            #
            # mod = sm.OLS(plot_df[y], plot_df[x].to_numpy()[:,None])
            # fii = mod.fit()
            # print(fii.summary2())
            # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            # ax.set_xlabel("Number of modulated layers")
            # ax.set_ylabel("Performance (% correct)")
            # fig = ax.get_figure()

            # # Annotations for statistical significance
            # plot_df = df[(df['n_hebb']==n_hebb) & ((df['mechanisms']==6) | (df['mechanisms']==-1))]
            # plot_df["n_targets"] = plot_df["n_targets"].astype(int).astype(str)
            # pairs = [("1","2"),("2","3"),("4","3"),("5","4"),("5","3")]
            # p_values = {
            #     p:mannwhitneyu(
            #         plot_df[plot_df[x]==p[0]][y],
            #         plot_df[plot_df[x]==p[1]][y],
            #         alternative="less"
            #     ).pvalue for p in pairs}
            # annotator = Annotator(ax, pairs, data=plot_df, x=x, y=y, order=[str(i) for i in range(int(plot_df["n_targets"].max()) + 1)])
            # annotator.configure(text_format="star", loc="inside")
            # annotator.set_pvalues_and_annotate(list(p_values.values()))
            # utils.make_fig(fig, ax, "fig/seaborn")

            ##############################
            # target_labels plot
            ##############################


            fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(4,3), sharey=True, gridspec_kw={'width_ratios':[5,1]})
            plot_df = self.results[((self.results['n_targets']<2) | (self.results['target_labels']==('CA1', 'ECout'))) & (self.results['n_hebb']==n_hebb) & (self.results['mechanisms']==6)]
            plot_df["target_labels"] = plot_df["target_labels"].astype(str)
            order = ["('ECin',)", "('DG',)", "('CA3',)", "('CA1',)", "('ECout',)"]#, "('CA1', 'ECout')"]
            x='target_labels'
            y='mean_test_corrects'

            sns.boxplot(
                data=plot_df[plot_df['n_targets']<2],
                ax=ax[0],
                x=x,
                y=y,
                showfliers = True,
                flierprops=flierprops,
                order=order)


            sns.boxplot(
                data=plot_df[(self.results['target_labels']==('CA1', 'ECout'))],
                ax=ax[1],
                x=x,
                y=y,
                flierprops=flierprops,
                showfliers = True)

            # Annotations for statistical significance
            # pairs = list(combinations(order, 2))

            spearman_df = plot_df[plot_df['n_targets']<2]
            spearmanres = spearmanr([order.index(target) for target in spearman_df['target_labels']], spearman_df[y], alternative="greater")
            print("N:", len(spearman_df.index), "|", spearmanres)

            pairs = [("('ECin',)","('DG',)"), ("('ECin',)","('CA3',)"), ("('ECin',)","('CA1',)"), ("('ECin',)","('ECout',)"), ("('CA1',)","('ECout',)"), ("('CA1',)","('CA1', 'ECout')"), ("('ECout',)","('CA1', 'ECout')")]
            stattest = {}
            for p in pairs:
                alternative = "two-sided" if p==("('CA1',)","('ECout',)") else "less"
                print(p, alternative)
                stattest[p] = wilcoxon(
                    plot_df[plot_df[x]==p[0]][y],
                    plot_df[plot_df[x]==p[1]][y],
                    alternative=alternative,
                    method='approx'
                )
            for p,v in stattest.items():
                for i in range(2):
                    print(p[i], "N:", len(plot_df[plot_df[x]==p[i]][y].index), "Median:", plot_df[plot_df[x]==p[i]][y].median())
                print("pvalue", v.pvalue)
                print("statistic", v.statistic)
                print("z statistic", v.zstatistic)
                print("\n\n")
            # # annotator = Annotator(ax, pairs, data=plot_df, x=x, y=y, order=order)
            # # annotator.configure(text_format="star", loc="outside")
            # # annotator.set_pvalues_and_annotate([v.pvalue for v in stattest.values()])


            for i in range(2):
                xlim = ax[i].get_xlim()
                ax[i].plot([-10,10], [50 for i in range(2)], linestyle='--', color='k', alpha=.2, zorder=-1)
                ax[i].set_xlim(xlim)
                ax[i].set_xlabel("")
            ax[1].set_ylabel("")
            # ax[1].set_ylim(ax[0].get_ylim())
            # add a big axis, hide frame
            fig.add_subplot(111, frameon=False)
            # hide tick and tick label of the big axis
            plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
            plt.xlabel("Modulated layers")
            ax[0].set_ylabel("Performance (% correct)")
            ax[0].set_xticklabels([t.get_text()[2:-3] for t in ax[0].get_xticklabels()])
            ax[1].set_xticklabels(["CA1 & ECout"])
            fig.tight_layout()
            utils.make_fig(fig, ax, self.figures_path, "target_labels"+"_nhebb_"+str(n_hebb))


            plot_df = self.results[(self.results['n_hebb']==n_hebb) & (self.results['mechanisms']==6) &
                (
                    ((self.results['n_targets']==2) & (self.results['CA1']==True) & (self.results['ECout']==True)) |
                    ((self.results['n_targets']==1) & (self.results['CA1']==True)) |
                    ((self.results['n_targets']==1) & (self.results['ECout']==True))
                )
            ]
            plot_df["target_labels"] = plot_df["target_labels"].astype(str)
            print(plot_df['target_labels'])
            order = ["('CA1',)", "('ECout',)", "('CA1', 'ECout')"]
            pairs = [
                # ("('CA1',)", "('ECout',)"),
                ("('CA1',)", "('CA1', 'ECout')"),
                ("('ECout',)", "('CA1', 'ECout')")
            ]
            x='target_labels'
            y='mean_test_corrects'
            ax = sns.boxplot(
                data=plot_df,
                x=x,
                y=y,
                showfliers = True,
                flierprops=flierprops,
                order=order)
            xlim = ax.get_xlim()
            plt.plot([-10,10], [50 for i in range(2)], linestyle='--', color='k', alpha=.2, zorder=-1)

            # Annotations for statistical significance
            # pairs = list(combinations(order, 2))
            stattest = {
                p:mannwhitneyu(
                    plot_df[plot_df[x]==p[0]][y],
                    plot_df[plot_df[x]==p[1]][y],
                    alternative="two-sided"
                ) for p in pairs}
            print("pairs",pairs)
            # annotator = Annotator(ax, pairs, data=plot_df, x=x, y=y, order=order)
            # annotator.configure(text_format="star", loc="inside")
            # annotator.set_pvalues_and_annotate([v.pvalue for v in stattest.values()])

            ax.set_xlim(xlim)
            ax.set_xlabel("Contextual input location")
            ax.set_ylabel("Performance (% correct)")
            ax.set_xticks(range(3), ["CA1","ECout","CA1 & ECout"])
            fig = ax.get_figure()
            fig.tight_layout()
            utils.make_fig(fig, ax, self.figures_path)

            ##############################
            # All target_labels plot
            ##############################

            plot_df = self.results[(self.results['n_hebb']==n_hebb) & (self.results['mechanisms']==6) &
                (
                    (self.results['n_targets']>0)
                )
            ]
            plot_df["target_labels"] = plot_df["target_labels"].astype(str)
            order = list(plot_df.groupby(by=["target_labels"])["mean_test_corrects"].quantile(.25).sort_values(ascending=False).index)
            print(order)
            x='target_labels'
            y='mean_test_corrects'

            fig,ax = plt.subplots(nrows=2, ncols=1, figsize=(5,3.5), sharex=True, gridspec_kw={'height_ratios':[3,2]})

            sns.boxplot(
                data=plot_df,
                x=x,
                y=y,
                ax=ax[0],
                showfliers = True,
                flierprops=flierprops,
                order=order)
            xlim = ax[0].get_xlim()
            ax[0].plot([-10,10000], [50 for i in range(2)], linestyle='--', color='k', alpha=.2, zorder=-1)

            print(order)
            for i,c in enumerate(order):
                for j,l in enumerate(constants.MODULAR_LAYERS[::-1]):
                    if l in c:
                        ax[1].plot([i], [j], marker="x", color='black')


            # # Annotations for statistical significance
            # # pairs = list(combinations(order, 2))
            # p_values = {
            #     p:mannwhitneyu(
            #         plot_df[plot_df[x]==p[0]][y],
            #         plot_df[plot_df[x]==p[1]][y],
            #         alternative="less"
            #     ).pvalue for p in pairs}
            # print("pairs",pairs)
            # annotator = Annotator(ax, pairs, data=plot_df, x=x, y=y, order=order)
            # annotator.configure(text_format="star", loc="inside")
            # annotator.set_pvalues_and_annotate(list(p_values.values()))

            ax[0].set_xlim(xlim)
            ax[0].set_xlabel("")
            ax[0].set_ylabel("Performance (% correct)")
            ax[0].set_xticks([])
            ax[1].set_yticks(range(len(constants.MODULAR_LAYERS)), constants.MODULAR_LAYERS[::-1], rotation=20)
            ax[1].set_ylim(-1, len(constants.MODULAR_LAYERS)-.5)
            ax[1].set_aspect('auto')
            ax[1].set_xlabel("Combinations of modulated layers")
            ax[1].spines.right.set_visible(False)
            ax[1].spines.left.set_visible(False)
            ax[1].spines.top.set_visible(False)
            ax[1].spines.bottom.set_visible(False)
            ax[1].xaxis.set_ticks_position('none')
            ax[1].yaxis.set_ticks_position('none')
            fig.tight_layout()
            plt.subplots_adjust(hspace=.01)
            utils.make_fig(fig, ax, self.figures_path, "alltargets"+"_nhebb_"+str(n_hebb))



def analyse_splitters(xp_df, start_session=50, end_session=None, n_contexts=2, n_tasks=8, target_labels_list=None):

    organized_activity = {target_labels: {} for target_labels in xp_df['target_labels'].unique()}
    splitters = {target_labels: {} for target_labels in xp_df['target_labels'].unique()}

    if end_session is None:
        end_session = xp_df.iloc[0]['n_train_sessions']

    sessions = np.arange(start_session, end_session, dtype=int)
    tasks = np.arange(n_tasks, dtype=int)
    contexts = np.arange(n_contexts, dtype=int)

    proportion_dicts = []

    for i,target_labels in enumerate(target_labels_list):

        print(target_labels)

        tmp_df = xp_df[(xp_df['target_labels']==target_labels)]
        assert len(tmp_df.index)==1

        proportion_dict = {}

        for layer in constants.MODULAR_LAYERS:

            print(layer)

            activity = tmp_df.iloc[0]['test_activity'][layer]

            n_neurons = activity.shape[2]


            organized_activity[target_labels][layer] = organize_activity(
                activity=activity,
                tasks=tasks,
                contexts=contexts,
                sessions=sessions,
                n_neurons=n_neurons
            )

            splitters[target_labels][layer] = find_splitters(
                activity_df=organized_activity[target_labels][layer],
                tasks=tasks,
                contexts=contexts,
                n_neurons=n_neurons
            )


    org_act_without_splitters, splitter_count = remove_splitters(organized_activity, splitters, n_tasks)
    org_act_without_random = remove_random(organized_activity, splitter_count)

    for plot_func in [pca_hist, pca_scatter]:
        splitter_pca(
            organized_activity=organized_activity,
            target_labels_list=target_labels_list,
            tasks=tasks,
            contexts=contexts,
            plot_func=plot_func,
            name="pca",
            in_3d=False
        )

        splitter_pca(
            organized_activity=org_act_without_splitters,
            target_labels_list=target_labels_list,
            tasks=tasks,
            contexts=contexts,
            plot_func=plot_func,
            name="pca_without_splitters",
            in_3d=False
        )

        splitter_pca(
            organized_activity=org_act_without_random,
            target_labels_list=target_labels_list,
            tasks=tasks,
            contexts=contexts,
            plot_func=plot_func,
            name="pca_without_random",
            in_3d=False
        )

    proportion_plot(fig, ax, splitters)

    n_tasks_plot(
        splitter_df=pd.concat(splitter_dfs),
        tasks=tasks
    )

def remove_splitters(organized_activity, splitters, n_tasks):

    org_act_without_splitters = {target_labels: {} for target_labels in organized_activity.keys()}
    splitter_count = {target_labels: {} for target_labels in organized_activity.keys()}

    for target_labels_k, target_labels_v in organized_activity.items():
        for layer_k, layer_v in target_labels_v.items():
            splitters_df = splitters[target_labels_k][layer_k]
            splitters_list = list(splitters_df[splitters_df["Total"]==n_tasks]["#"])

            org_act_without_splitters[target_labels_k][layer_k] = layer_v.drop(['#'+str(i) for i in splitters_list], axis=1)
            splitter_count[target_labels_k][layer_k] = len(splitters_list)

    return org_act_without_splitters, splitter_count

def remove_random(organized_activity, splitter_count, rng=None):

    if rng is None:
        rng = np.random.default_rng()

    org_act_without_random = {target_labels: {} for target_labels in organized_activity.keys()}
    for target_labels_k, target_labels_v in organized_activity.items():
        for layer_k, layer_v in target_labels_v.items():

            n_neurons = int(layer_v.columns[-1][1:]) + 1
            print(layer_k, n_neurons)
            to_remove = rng.choice(n_neurons, size=splitter_count[target_labels_k][layer_k], replace=False)
            org_act_without_random[target_labels_k][layer_k] = layer_v.drop(['#'+str(i) for i in to_remove], axis=1)

    return org_act_without_random

def organize_activity(activity, tasks, contexts, sessions, n_neurons):

    organized_activity = []

    for task in tasks:
        for context in contexts:

            task_x_context = task + len(tasks) * context

            for session in sessions:

                # Add information about session, task and context
                organized_activity.append({
                    'session': session,
                    'task': task,
                    'context': context,
                })

                # Add activity of all neurons
                for neuron in range(n_neurons):

                    organized_activity[-1]['#'+str(neuron)] = activity[session, task_x_context, neuron]

    return pd.DataFrame(organized_activity)

def find_splitters(activity_df, tasks, contexts, n_neurons, pvalue_threshold=.001):

    # Stores whether a cell is splitter in a task (1) or not (0)
    splitter = [{t:0 for t in tasks} for n in range(n_neurons)]

    splitter_proportion = {}
    for task in tasks:

        # group by context
        context_dfs = [activity_df[(activity_df['task']==task) & (activity_df['context']==context)] for context in contexts]

        for neuron in range(n_neurons):

            # check if any difference between contexts
            if ((context_dfs[0]['#'+str(neuron)].to_numpy() - context_dfs[1]['#'+str(neuron)].to_numpy())!=0).any():
                test = wilcoxon(
                    context_dfs[0]['#'+str(neuron)],
                    context_dfs[1]['#'+str(neuron)],
                    alternative="two-sided")

                if test.pvalue < pvalue_threshold:
                    splitter[neuron][task] = 1

    splitter = pd.DataFrame(splitter)
    splitter['Total'] = splitter.sum(axis=1)
    splitter['#'] = range(n_neurons)
    return splitter

def splitter_pca(organized_activity, target_labels_list, tasks, contexts, plot_func, name='pca', in_3d=False):

    subplot_kw = dict(projection='3d') if in_3d else None
    fig, ax = plt.subplots(len(target_labels_list), len(constants.MODULAR_LAYERS), sharex=True, figsize=(10,5),subplot_kw=subplot_kw)

    for i,target_labels in enumerate(target_labels_list):
        for j,layer in enumerate(constants.MODULAR_LAYERS):


            activity_df = organized_activity[target_labels][layer]
            training_data = activity_df[[col for col in activity_df.columns if '#' in col]]

            #Scale the data
            scaler = StandardScaler()
            scaler.fit(training_data)
            training_data = scaler.transform(training_data)

            #Obtain principal components
            pca = PCA().fit(training_data)

            plot_func(fig, ax[i,j], pca, activity_df, contexts, name, in_3d)
    plt.tight_layout()
    # utils.make_fig(fig, ax, "fig/seaborn/splitter", name)

def pca_hist(fig, ax, pca, activity_df, contexts, name, in_3d):
    pcs = pca.components_
    print(pcs.shape)

    ytrain = activity_df['context']
    Xtrain = activity_df[[col for col in activity_df.columns if '#' in col]]
    print(ytrain.shape, Xtrain.shape)
    Xtrain = pca.transform(Xtrain)#[:,:4]

    print(ytrain.shape, Xtrain.shape)

    for i in range(5):
        spearmanres = spearmanr(Xtrain[:,i], ytrain)
        print(spearmanres)

    ax.hist([pcs[i,:] for i in range(3)])


def pca_scatter(fig, ax, pca, activity_df, contexts, name, in_3d):
    for h,context in enumerate(contexts):
        context_data = activity_df[activity_df['context']==h]
        context_data = context_data[[col for col in activity_df.columns if '#' in col]]
        proj = pca.transform(context_data)

        if in_3d:
            ax.scatter(proj[:,0], proj[:,1], proj[:,2], s=2, c=["red","blue"][h], label=str(context), alpha=.7)
        else:
            ax.scatter(proj[:,0], proj[:,1], s=2, c=["red","blue"][h], label=str(context), alpha=.7)




def proportion_plot(fig, ax, proportion_dicts):
    fig, ax = plt.subplots(len(organized_activity), 1, sharex=True, figsize=(10,5))
    for i in range(len(proportion_dicts)):
        sns.barplot(data=proportion_dicts[i], x="Layer", y="Proportion", ax=ax[i])

    plt.tight_layout()
    utils.make_fig(fig, ax, "fig/seaborn/splitter", "proportion")

def n_tasks_plot(splitter_df, tasks):
    fig, ax = plt.subplots()
    splitter_df['Total'].hist(grid=False, bins=range(len(tasks)+2), ax=ax)
    ax.set_xlabel("Number of tasks")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    utils.make_fig(fig, ax, "fig/seaborn/splitter", "in_n_tasks")
