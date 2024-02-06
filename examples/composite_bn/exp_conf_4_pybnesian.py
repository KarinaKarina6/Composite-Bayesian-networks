from datetime import timedelta
import sys
import os
# parentdir = os.getcwd() 
# sys.path.insert(0, parentdir)
# sys.path.append('C:\\Users\\anaxa\\Documents\\Projects\\GOLEM\\examples\\bn')
import pathlib
current_path = pathlib.Path().resolve()
parentdir = os.getcwd() 
sys.path.insert(0, parentdir)
sys.path.append(str(current_path) + '\\examples\\bn')
import pandas as pd
from sklearn import preprocessing
import bamt.preprocessors as pp
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.adapter import DirectAdapter
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.genetic.operators.selection import SelectionTypesEnum
from golem.core.optimisers.genetic.operators.crossover import exchange_parents_one_crossover, exchange_parents_both_crossover, exchange_edges_crossover
from golem.core.optimisers.objective.objective import Objective
from golem.core.optimisers.optimizer import GraphGenerationParams
from examples.composite_bn.composite_model import CompositeModel
from examples.composite_bn.composite_node import CompositeNode
from sklearn.model_selection import train_test_split
from examples.composite_bn.composite_bn_genetic_operators import (
    custom_crossover_all_model as composite_crossover_all_model, 
    custom_mutation_add_structure as composite_mutation_add_structure, 
    custom_mutation_delete_structure as composite_mutation_delete_structure, 
    custom_mutation_reverse_structure as composite_mutation_reverse_structure,
    custom_mutation_add_model as composite_mutation_add_model,
)
from functools import partial

from comparison import Comparison
from fitness_function import FitnessFunction
from rule import Rule
from likelihood import Likelihood
from write_txt import Write
from examples.bn.bn_genetic_operators import (
    custom_mutation_add_structure as classical_mutation_add_structure,
    custom_mutation_delete_structure as classical_mutation_delete_structure,
    custom_mutation_reverse_structure as classical_mutation_reverse_structure
)
from pgmpy.estimators import K2Score
from bamt.networks.continuous_bn import ContinuousBN
from bamt.networks.hybrid_bn import HybridBN
from bamt.networks.discrete_bn import DiscreteBN
from statistics import mean

# data_train <- read.csv(paste('data/',data_name, '/' , data_name, '_train_fold_', as.character(fold), '.csv', sep = ""), check.names = F)[, c(-1)]
# data_test <- read.csv(paste('data/',data_name, '/' , data_name, '_test_fold_', as.character(fold), '.csv', sep = ""), check.names = F)[, c(-1)]
# data_edges <- read.csv(paste('data/',data_name, '/' , data_name, '_edges_fold_', as.character(fold), '.csv', sep = ""), check.names = F)
# edges = data_edges$edges

# data_train$class <- NULL
# data_test$class <- NULL

def run_example(file):
    if exist_true_str:
        with open('examples/data/1000/txt/'+(file)+'.txt') as f:
            lines = f.readlines()
        true_net = []
        for l in lines:
            e0 = l.split()[0]
            e1 = l.split()[1].split('\n')[0]
            true_net.append((e0, e1))    
    
    fitness_function = FitnessFunction()
    FF_classical = fitness_function.classical_K2 # classical_metric_2
    FF_composite = fitness_function.composite_metric

    if bn_type == 'classical':
        fitness_function_GA = FF_classical

        mutations = [
        classical_mutation_add_structure, 
        classical_mutation_delete_structure, 
        classical_mutation_reverse_structure
        ]

        crossovers = [
            exchange_parents_one_crossover,
            exchange_parents_both_crossover,
            exchange_edges_crossover
            ]
    elif bn_type == 'composite':

        composite_FF = FF_composite
        complexity_FF = FF_classical


        mutations = [
        # composite_mutation_add_structure, 
        # composite_mutation_delete_structure, 
        # composite_mutation_reverse_structure, 
        composite_mutation_add_model    
        ]    

        crossovers = [
            # exchange_parents_one_crossover,
            # exchange_parents_both_crossover,
            # exchange_edges_crossover,
            composite_crossover_all_model
            ]
    else:
        print('There is no such type of BN: "{}". You can only use "classical" or "composite".'.format(bn_type)) 
        return 


    # if file in ['abalone', 'adult', 'australian_statlog', 'liver_disorders']:
    #     data = pd.read_csv('examples/data/1000/UCI/' + file + '.data') 
    # else:
    #     data = pd.read_csv('examples/data/1000/csv/' + file + '.csv') 

    # if 'Unnamed: 0' in data.columns:
    #     data.drop(['Unnamed: 0'], axis=1, inplace=True)
    # data.dropna(inplace=True)
    # data.reset_index(inplace=True, drop=True)
    # vertices = list(data.columns)
    # if file == 'adult':
    #     for i in data.columns:
    #         if data[i].dtype == 'int64':
    #             data[i] = data[i].astype(float)


    encoder = preprocessing.LabelEncoder()
    discretizer = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')

    # if bn_type == 'classical':
    #     p = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)]) 
    #     discretized_data, _ = p.apply(data)
    #     data_train_test , data_val = train_test_split(discretized_data, test_size=0.2, shuffle = True, random_state=random_seed[number-1])
    #     data_train , data_test = train_test_split(data_train_test, test_size=0.2, shuffle = True, random_state=random_seed[number-1])
    # elif bn_type == 'composite':
    #     p = pp.Preprocessor([('encoder', encoder)])
    #     discretized_data, _ = p.apply(data)
    #     data_train_test , data_val = train_test_split(discretized_data, test_size=0.2, shuffle = True, random_state=random_seed[number-1])
    #     data_train , data_test = train_test_split(data_train_test, test_size=0.2, shuffle = True, random_state=random_seed[number-1])


    
    # data_train_composite , data_test_composite = train_test_split(discretized_data_composite, test_size=0.2, shuffle = True, random_state=random_seed[number-1])
    data_train_composite = pd.read_csv('examples/data/pybnesian_MoTBFs/' + file + '/' + file + '_train_fold_' + str(number) + '.csv') 
    data_test_composite = pd.read_csv('examples/data/pybnesian_MoTBFs/' + file + '/' + file + '_test_fold_' + str(number) + '.csv') 
    structure_init = pd.read_csv('examples/data/pybnesian_MoTBFs/' + file + '/' + file + '_edges_fold_' + str(number) + '.csv') 
    
    
    structure = list(zip(structure_init['edges'][0::2], structure_init['edges'][1::2]))
    
    # if 'Unnamed: 0' in data_train_composite.columns:
    #     data_train_composite.drop(['Unnamed: 0'], axis=1, inplace=True)
    # if 'class' in data_train_composite.columns:
    #     data_train_composite.drop(['class'], axis=1, inplace=True)

    # if 'Unnamed: 0' in data_test_composite.columns:
    #     data_test_composite.drop(['Unnamed: 0'], axis=1, inplace=True)
    # if 'class' in data_test_composite.columns:
    #     data_test_composite.drop(['class'], axis=1, inplace=True)

    def data_tranformation(data):
        if 'Unnamed: 0' in data.columns:
            data.drop(['Unnamed: 0'], axis=1, inplace=True)
        if 'class' in data.columns:
            data.drop(['class'], axis=1, inplace=True)    
        
        for i in data.columns:
            if data[i].dtype == 'int64':
                data[i] = data[i].astype(float)

        return data
    
    data_train_composite = data_tranformation(data_train_composite)
    data_test_composite = data_tranformation(data_test_composite)
    vertices = list(data_train_composite.columns)


    # p_for_K2 = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)]) 
    p_for_composite = pp.Preprocessor([('encoder', encoder)])
    # discretized_data_K2, _ = p_for_K2.apply(data)
    discretized_data_composite, _ = p_for_composite.apply(data_train_composite)

    # data_train = discretized_data
    # data_test = pd.DataFrame()

    # правила для байесовских сетей: нет петель, нет циклов, нет повторяющихся узлов
    rules = Rule().bn_rules()

    # инициализация начальной сети (пустая)
    initial = [CompositeModel(nodes=[CompositeNode(nodes_from=None,
                                                    content={'name': vertex,
                                                            'type': p_for_composite.nodes_types[vertex],
                                                            'parent_model': None}) 
                                                    for vertex in vertices])] 



    init = initial[0]
    
    # types=list(p_for_K2.info['types'].values())
    # if 'cont' in types and ('disc' in types or 'disc_num' in types):
    #     bn = HybridBN(has_logit=False, use_mixture=False)
    # elif 'disc' in types or 'disc_num' in types:
    #     bn = DiscreteBN()
    # elif 'cont' in types:
    #     bn = ContinuousBN(use_mixture=False)

    # bn.add_nodes(p_for_K2.info)
    # bn.add_edges(discretized_data_K2, scoring_function=('K2', K2Score))        

    # for node in init.nodes: 
    #     parents = []
    #     for n in bn.nodes:
    #         if str(node) == str(n):
    #             parents = n.cont_parents + n.disc_parents
    #             break
    #     for n2 in init.nodes:
    #         if str(n2) in parents:
    #             node.nodes_from.append(n2)

    def structure_to_opt_graph(fdt, structure):

        encoder = preprocessing.LabelEncoder()
        # discretizer = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
        p = pp.Preprocessor([('encoder', encoder)])
        discretized_data, est = p.apply(data_train_composite)

        bn = []
        if 'cont' in p.info['types'].values() and ('disc' in p.info['types'].values() or 'disc_num' in p.info['types'].values()):
            bn = HybridBN(has_logit=False, use_mixture=False)
        elif 'disc' in p.info['types'].values() or 'disc_num' in p.info['types'].values():
            bn = DiscreteBN()
        elif 'cont' in p.info['types'].values():
            bn = ContinuousBN(use_mixture=False)  

        bn.add_nodes(p.info)
        bn.set_structure(edges=structure)
        
        for node in fdt.nodes: 
            parents = []
            for n in bn.nodes:
                if str(node) == str(n):
                    parents = n.cont_parents + n.disc_parents
                    break
            for n2 in fdt.nodes:
                if str(n2) in parents:
                    node.nodes_from.append(n2)      

        for node in init.nodes:
            if not (node.nodes_from == None or node.nodes_from == []):
                if node.content['type'] == 'cont':
                    node.content['parent_model'] = 'LinearRegression'
                else:
                    node.content['parent_model'] = 'LogisticRegression'            
        
        return fdt    

    init = structure_to_opt_graph(init, structure)

    print(init.get_edges())

    # LLL = LL.likelihood_function_composite(init, data_train_composite, data_test_composite)

    # задаем для оптимизатора fitness-функции на качество и сложность структур
    objective = Objective(
            {'fitness function': partial(composite_FF, data_train = data_train_composite, data_test = data_test_composite)},
            is_multi_objective=False,
        )
    # objective = Objective(
    #         quality_metrics={'composite_FF': partial(composite_FF, data_train = data_train_composite, data_test = data_test_composite)},
    #         complexity_metrics={'complexity_FF': partial(complexity_FF, data_train = data_train_K2, data_test = data_test_K2)},
    #         is_multi_objective=True,
    #     )
    requirements = GraphRequirements(
        max_arity=100,
        max_depth=100, 
        num_of_generations=n_generation,
        timeout=timedelta(minutes=time_m),
        # history_dir = True,
        early_stopping_iterations = early_stopping_iterations,
        n_jobs=-1
        )

    optimiser_parameters = GPAlgorithmParameters(
        multi_objective=objective.is_multi_objective,
        pop_size=pop_size,
        max_pop_size = pop_size,
        crossover_prob=crossover_probability, 
        mutation_prob=mutation_probability,
        genetic_scheme_type = GeneticSchemeTypesEnum.steady_state,
        selection_types = [SelectionTypesEnum.tournament],
        mutation_types = mutations,
        crossover_types = crossovers,
    )

    graph_generation_params = GraphGenerationParams(
        adapter=DirectAdapter(base_graph_class=CompositeModel, base_node_class=CompositeNode),
        rules_for_constraint=rules,
        )

    optimiser = EvoGraphOptimizer(
        graph_generation_params=graph_generation_params,
        graph_optimizer_params=optimiser_parameters,
        requirements=requirements,
        initial_graphs=initial,
        objective=objective)

    # запуск оптимизатора
    # optimized_graphs содержит всех недоминируемых особей, которые когда-либо жили в популяции
    # в результате получаем несколько индивидов
    optimized_graphs = optimiser.optimise(objective)

    # optimiser.history.save('C:\\Users\\anaxa\\Documents\\Projects\\GOLEM_fork\\GOLEM\\examples\\results\\history_classical.json')

    vars_of_interest = {}
    comparison = Comparison()
    LL = Likelihood()    

    for optimized_graph in optimized_graphs:
        # if bn_type == 'classical':
        #     optimized_graph = fitness_function.edge_reduction(optimized_graph, data_train=data_train, data_test=data_val)
        optimized_structure = [(str(edge[0]), str(edge[1])) for edge in optimized_graph.get_edges()]
        # score = fitness_function_GA(optimized_graph, data_train = data_train, data_test = data_val)
        # score = fitness_function_GA(optimized_graph, data_train = data_train, data_test = data_test)
        # if bn_type == 'composite':
        #     score = - score
        # spent_time = optimiser.timer.minutes_from_start
        spent_time = optimiser.timer.seconds_from_start

        # if bn_type == 'composite':
        #     p = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)]) 
        #     discretized_data, _ = p.apply(data)
        #     data_train_test , data_val = train_test_split(discretized_data, test_size=0.2, shuffle = True, random_state=random_seed[number-1])
        #     data_train , data_test = train_test_split(data_train_test, test_size=0.2, shuffle = True, random_state=random_seed[number-1])

        # likelihood = LL.likelihood_function(optimized_graph, data_train=data_train, data_val=data_val)
        # likelihood = LL.likelihood_function(optimized_graph, data_train=data_train, data_val=data_train)
        likelihood = LL.likelihood_function_composite(optimized_graph, data_train_composite, data_test_composite)
        if exist_true_str:
            f1 = comparison.F1(optimized_structure, true_net)
            SHD = comparison.precision_recall(optimized_structure, true_net)['SHD']
        if bn_type == 'composite': 
            models = {node:node.content['parent_model'] for node in optimized_graph.nodes}


        # true_net_graph = CompositeModel(nodes=[CompositeNode(nodes_from=None,
        #                                             content={'name': vertex,
        #                                                     'type': p.nodes_types[vertex],
        #                                                     'parent_model': None}) 
        #                                             for vertex in vertices])
        # for parent,child in true_net:
        #     parent_node = true_net_graph.get_nodes_by_name(parent)[0]
        #     child_node = true_net_graph.get_nodes_by_name(child)[0]
        #     child_node.nodes_from.append(parent_node)   
        
        # score_true_net = fitness_function_GA(true_net_graph, data_train = data_train, data_test = data_val)
        # score_deviation = (score - score_true_net)*100 / score_true_net
        
        vars_of_interest['Structure'] = optimized_structure
        # vars_of_interest['Score'] = -score
        # vars_of_interest['Score_deviation'] = score_deviation
        vars_of_interest['Likelihood'] = likelihood
        vars_of_interest['Spent time'] = spent_time
        if exist_true_str:
            vars_of_interest['f1'] = f1
            vars_of_interest['SHD'] = SHD
        if bn_type == 'composite': 
            vars_of_interest['Models'] = models

        vars_of_interest['Generation number'] = optimiser.current_generation_num
        vars_of_interest['Population number'] = optimiser.graph_optimizer_params.pop_size

        print(LL_list.append(likelihood))
        print('LL_list = ', likelihood)
        print(vars_of_interest)
        write = Write()
        write.write_txt(vars_of_interest, path = os.path.join(parentdir, 'examples', 'results', 'results_for_pybnesian_MoTBFs'), file_name = 'CBN_for_pybnesian_MoTBFs_' + file + '_run_' + str(number) + '.txt')
        


if __name__ == '__main__':
    # ['asia', 'cancer', 'earthquake', 'sachs', 'survey', 'healthcare', 'sangiovese', 'alarm', 'barley', 'child', 'insurance', 'mildew', 'water', 'ecoli70', 'magic-niab', 'mehra-complete', 'hailfinder', ]
     
    #    
    # files = ['asia', 'cancer', 'earthquake', 'sachs', 'survey', 'healthcare', 'sangiovese', 'alarm', 'barley', 'child', 'insurance', 'mildew', 'water', 'ecoli70', 'magic-niab'] # , 'mehra-complete', 'hailfinder'
    # files = ['asia']  # 'asia' , 'cancer', 'earthquake', 'sachs', 'sangiovese', 'barley'
    # 'Iris','Yeast', 'QSAR fish toxicity', 'Liver', 'Glass', 'Ecoli', 'Balance', 'Breast_cancer', 'Parkinsons', 'Vowel', 'QSAR Aquatic', 'Wine'
    files = ['Iris','Yeast', 'QSAR fish toxicity', 'Liver', 'Glass', 'Ecoli', 'Balance', 'Breast_cancer', 'Parkinsons', 'Vowel','Block', 'Breast_tissue', 'CPU', 'Ionosphere', 'Sonar', 'Vehicle', 'Wdbc', 'Wpbc'] # ['Iris','Yeast', 'QSAR fish toxicity', 'Liver', 'Glass', 'Ecoli', 'Balance', 'Breast_cancer', 'Parkinsons', 'Vowel', 'QSAR Aquatic', 'Wine']
    exist_true_str = False
    # размер популяции     
    pop_size = 20 # 40
    # количество поколений
    n_generation = 1000
    # вероятность кроссовера
    crossover_probability = 0.8
    # вероятность мутации 
    mutation_probability = 0.9
    # stopping_after_n_generation
    early_stopping_iterations = 10 # 20
    time_m = 15
    # это нужно для того, чтобы одно и то же разделение выборки на train/test/val можно было применять для GA и для HC (для каждого прогона своё значение random_seed[i]) 
    random_seed = [87, 60, 37, 99, 42, 92, 48, 91, 86, 33]

    # количество прогонов
    n = 9
    for file in files:
        LL_list = []
        for bn_type in ['composite']: 
            number = 0
            while number <= n:
                try:
                    run_example(file) 
                except:
                    print('except')
                    number -= 1
                number += 1 
        print(file)
        print(LL_list)
        print(mean(LL_list))
        # except Exception as ax:
        #     print(ax)



