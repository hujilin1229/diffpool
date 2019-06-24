import networkx as nx
import numpy as np
import os
import re
import random
import torch
import torch_geometric as tg
from torch_geometric.data import Data
from graph_sampler import GraphSampler
import pickle
from util import precompute_dist_data

def read_graphfile(datadir, dataname, max_nodes=None):
    ''' Read data from https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
        graph index starts with 1 in file

    Returns:
        List of networkx objects with graph and node labels
    '''
    prefix = os.path.join(datadir, dataname, dataname)
    filename_graph_indic = prefix + '_graph_indicator.txt'
    # index of graphs that a given node belongs to
    graph_indic={}
    with open(filename_graph_indic) as f:
        i=1
        for line in f:
            line=line.strip("\n")
            graph_indic[i]=int(line)
            i+=1

    filename_nodes=prefix + '_node_labels.txt'
    node_labels=[]
    try:
        with open(filename_nodes) as f:
            for line in f:
                line=line.strip("\n")
                node_labels+=[int(line) - 1]
        num_unique_node_labels = max(node_labels) + 1
    except IOError:
        print('No node labels')
 
    filename_node_attrs=prefix + '_node_attributes.txt'
    node_attrs=[]
    try:
        with open(filename_node_attrs) as f:
            for line in f:
                line = line.strip("\s\n")
                attrs = [float(attr) for attr in re.split("[,\s]+", line) if not attr == '']
                node_attrs.append(np.array(attrs))
    except IOError:
        print('No node attributes')
       
    label_has_zero = False
    filename_graphs=prefix + '_graph_labels.txt'
    graph_labels=[]

    # assume that all graph labels appear in the dataset 
    #(set of labels don't have to be consecutive)
    label_vals = []
    with open(filename_graphs) as f:
        for line in f:
            line=line.strip("\n")
            val = int(line)
            #if val == 0:
            #    label_has_zero = True
            if val not in label_vals:
                label_vals.append(val)
            graph_labels.append(val)
    #graph_labels = np.array(graph_labels)
    label_map_to_int = {val: i for i, val in enumerate(label_vals)}
    graph_labels = np.array([label_map_to_int[l] for l in graph_labels])

    
    filename_adj=prefix + '_A.txt'
    adj_list={i:[] for i in range(1,len(graph_labels)+1)}    
    index_graph={i:[] for i in range(1,len(graph_labels)+1)}
    num_edges = 0
    with open(filename_adj) as f:
        for line in f:
            line=line.strip("\n").split(",")
            e0,e1=(int(line[0].strip(" ")),int(line[1].strip(" ")))
            adj_list[graph_indic[e0]].append((e0,e1))
            index_graph[graph_indic[e0]]+=[e0,e1]
            num_edges += 1
    for k in index_graph.keys():
        index_graph[k]=[u-1 for u in set(index_graph[k])]

    graphs=[]
    for i in range(1,1+len(adj_list)):
        # indexed from 1 here
        G=nx.from_edgelist(adj_list[i])
        if max_nodes is not None and G.number_of_nodes() > max_nodes:
            continue
      
        # add features and labels
        G.graph['label'] = graph_labels[i-1]
        for u in G.nodes():
            if len(node_labels) > 0:
                node_label_one_hot = [0] * num_unique_node_labels
                node_label = node_labels[u-1]
                node_label_one_hot[node_label] = 1
                G.node[u]['label'] = node_label_one_hot
            if len(node_attrs) > 0:
                G.node[u]['feat'] = node_attrs[u-1]
        if len(node_attrs) > 0:
            G.graph['feat_dim'] = node_attrs[0].shape[0]

        # relabeling
        mapping={}
        it=0
        if float(nx.__version__)<2.0:
            for n in G.nodes():
                mapping[n]=it
                it+=1
        else:
            for n in G.nodes:
                mapping[n]=it
                it+=1
            
        # indexed from 0
        graphs.append(nx.relabel_nodes(G, mapping))
    return graphs


def prepare_data(graphs, args, test_graphs=None, max_nodes=0):

    random.shuffle(graphs)
    if test_graphs is None:
        train_idx = int(len(graphs) * args.train_ratio)
        test_idx = int(len(graphs) * (1-args.test_ratio))
        train_graphs = graphs[:train_idx]
        val_graphs = graphs[train_idx: test_idx]
        test_graphs = graphs[test_idx:]
    else:
        train_idx = int(len(graphs) * args.train_ratio)
        train_graphs = graphs[:train_idx]
        val_graphs = graphs[train_idx:]
    print('Num training graphs: ', len(train_graphs),
          '; Num validation graphs: ', len(val_graphs),
          '; Num testing graphs: ', len(test_graphs))

    print('Number of graphs: ', len(graphs))
    print('Number of edges: ', sum([G.number_of_edges() for G in graphs]))
    print('Max, avg, std of graph size: ',
            max([G.number_of_nodes() for G in graphs]), ', '
            "{0:.2f}".format(np.mean([G.number_of_nodes() for G in graphs])), ', '
            "{0:.2f}".format(np.std([G.number_of_nodes() for G in graphs])))

    # minibatch
    dataset_sampler = GraphSampler(train_graphs, normalize=False, max_num_nodes=max_nodes,
            features=args.feature_type)
    train_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers)

    dataset_sampler = GraphSampler(val_graphs, normalize=False, max_num_nodes=max_nodes,
            features=args.feature_type)
    val_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers)

    dataset_sampler = GraphSampler(test_graphs, normalize=False, max_num_nodes=max_nodes,
            features=args.feature_type)
    test_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers)

    return train_dataset_loader, val_dataset_loader, test_dataset_loader, \
            dataset_sampler.max_num_nodes, dataset_sampler.feat_dim, dataset_sampler.assign_feat_dim

def prepare_val_data(graphs, args, val_idx, max_nodes=0):

    random.shuffle(graphs)
    val_size = len(graphs) // 10
    train_graphs = graphs[:val_idx * val_size]
    if val_idx < 9:
        train_graphs = train_graphs + graphs[(val_idx+1) * val_size :]
    val_graphs = graphs[val_idx*val_size: (val_idx+1)*val_size]
    print('Num training graphs: ', len(train_graphs),
          '; Num validation graphs: ', len(val_graphs))

    print('Number of graphs: ', len(graphs))
    print('Number of edges: ', sum([G.number_of_edges() for G in graphs]))
    print('Max, avg, std of graph size: ',
            max([G.number_of_nodes() for G in graphs]), ', '
            "{0:.2f}".format(np.mean([G.number_of_nodes() for G in graphs])), ', '
            "{0:.2f}".format(np.std([G.number_of_nodes() for G in graphs])))

    # minibatch
    dataset_sampler = GraphSampler(train_graphs, normalize=False, max_num_nodes=max_nodes,
            features=args.feature_type)
    train_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers)

    dataset_sampler = GraphSampler(val_graphs, normalize=False, max_num_nodes=max_nodes,
            features=args.feature_type)
    val_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers)

    return train_dataset_loader, val_dataset_loader, \
            dataset_sampler.max_num_nodes, dataset_sampler.feat_dim, dataset_sampler.assign_feat_dim

def prepare_val_data_tg(tg_list, args, val_idx, max_nodes=0):

    random.shuffle(tg_list)
    val_size = len(tg_list) // 10
    train_tg_list = tg_list[:val_idx * val_size]
    if val_idx < 9:
        train_tg_list = train_tg_list + tg_list[(val_idx+1) * val_size :]
    val_tg_list = tg_list[val_idx*val_size: (val_idx+1)*val_size]
    print('Num training graphs: ', len(train_tg_list),
          '; Num validation graphs: ', len(val_tg_list))

    print('Number of graphs: ', len(tg_list))
    print('Number of edges: ', sum([tg.edge_index.shape[1] for tg in tg_list]))
    print('Max, avg, std of graph size: ',
            max([len(tg.nodes_set[0]) for tg in tg_list]), ', '
            "{0:.2f}".format(np.mean([len(tg.nodes_set[0]) for tg in tg_list])), ', '
            "{0:.2f}".format(np.std([len(tg.nodes_set[0]) for tg in tg_list])))

    # minibatch
    dataset_sampler = GraphSampler(train_graphs, normalize=False, max_num_nodes=max_nodes,
            features=args.feature_type)
    train_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers)

    dataset_sampler = GraphSampler(val_graphs, normalize=False, max_num_nodes=max_nodes,
            features=args.feature_type)
    val_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers)

    return train_dataset_loader, val_dataset_loader, \
            dataset_sampler.max_num_nodes, dataset_sampler.feat_dim, dataset_sampler.assign_feat_dim

def nx_to_tg_data(graphs, features, edge_labels=None):
    data_list = []
    for i in range(len(graphs)):
        feature = features[i]
        graph = graphs[i].copy()
        graph.remove_edges_from(graph.selfloop_edges())

        # relabel graphs
        keys = list(graph.nodes)
        vals = range(graph.number_of_nodes())
        mapping = dict(zip(keys, vals))
        nx.relabel_nodes(graph, mapping, copy=False)

        x = np.zeros(feature.shape)
        graph_nodes = list(graph.nodes)
        for m in range(feature.shape[0]):
            x[graph_nodes[m]] = feature[m]
        x = torch.from_numpy(x).float()

        # get edges
        edge_index = np.array(list(graph.edges))
        edge_index = np.concatenate((edge_index, edge_index[:,::-1]), axis=0)
        edge_index = torch.from_numpy(edge_index).long().permute(1, 0)

        data = Data(x=x, edge_index=edge_index)

        # get edge_labels
        if edge_labels is not None:
            edge_label = edge_labels[i]
            mask_link_positive = np.stack(np.nonzero(edge_label))
            data.mask_link_positive = mask_link_positive
        data_list.append(data)

    return data_list


def Graph_load_batch(datadir, dataname, max_nodes=None, node_attributes = True, graph_labels=True):
    '''
    load many graphs, e.g. enzymes
    :return: a list of graphs
    '''
    print('Loading graph dataset: '+str(dataname))
    G = nx.Graph()
    # load data
    prefix = os.path.join(datadir, dataname, dataname)

    data_adj = np.loadtxt(prefix+'_A.txt', delimiter=',').astype(int)
    data_node_label = np.loadtxt(prefix + '_node_labels.txt', delimiter=',').astype(int)
    if node_attributes:
        data_node_att = np.loadtxt(prefix+'_node_attributes.txt', delimiter=',')
    else:
        data_node_att = data_node_label.reshape(-1, 1)
    data_graph_indicator = np.loadtxt(prefix+'_graph_indicator.txt', delimiter=',').astype(int)
    if graph_labels:
        data_graph_labels = np.loadtxt(prefix+'_graph_labels.txt', delimiter=',').astype(int)

    data_tuple = list(map(tuple, data_adj))

    # add edges
    G.add_edges_from(data_tuple)
    # add node attributes
    for i in range(data_node_label.shape[0]):
        G.add_node(i+1, feature = data_node_att[i])
        G.add_node(i+1, label = data_node_label[i])
    G.remove_nodes_from(list(nx.isolates(G)))

    # split into graphs
    graph_num = data_graph_indicator.max()
    node_list = np.arange(data_graph_indicator.shape[0])+1
    graphs = []
    features = []
    edge_labels = []
    for i in range(graph_num):
        # find the nodes for each graph
        nodes = node_list[data_graph_indicator==i+1]
        G_sub = G.subgraph(nodes)
        if graph_labels:
            G_sub.graph['label'] = data_graph_labels[i]

        if max_nodes is not None and G_sub.number_of_nodes() > max_nodes:
            continue
        else:
            graphs.append(G_sub)

        # assign edge labels
        n = len(nodes)
        label = np.zeros((n, n), dtype=int)
        for i, u in enumerate(G_sub.nodes()):
            for j, v in enumerate(G_sub.nodes()):
                if data_node_label[u - 1] == data_node_label[v - 1] and u > v:
                    label[i, j] = 1

        edge_labels.append(label)

        # assign node features
        idx = [node - 1 for node in nodes]
        feature = data_node_att[idx, :]
        features.append(feature)

    print('Loaded')
    return graphs, features, edge_labels


def load_tg_dataset(datadir, dataname, max_nodes=None, node_attributes = True, graph_labels=True):
    graphs, features, edge_labels = Graph_load_batch(
        datadir, dataname, max_nodes, node_attributes, graph_labels)

    return nx_to_tg_data(graphs, features, edge_labels)


def get_tg_dataset(args, node_attributes=True):
    # "Cora", "CiteSeer" and "PubMed"
    if args.bmname in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = tg.datasets.planetoid(root='datasets/' + args.bmname, name=args.bmname)
    else:
        dataset = load_tg_dataset(args.datadir, args.bmname, args.max_nodes, node_attributes=node_attributes)

    # precompute shortest path
    if not os.path.isdir('datasets/cache'):
        os.mkdir('datasets')
        os.mkdir('datasets/cache')
    f1_name = 'datasets/cache/' + args.bmname + str(args.approximate) + '_dists.dat'

    if args.cache and (os.path.isfile(f1_name) and args.task!='link'):
        with open(f1_name, 'rb') as f1:
            dists_list = pickle.load(f1)

        print('Cache loaded!')
        data_list = []
        for i, data in enumerate(dataset):
            data.dists = torch.from_numpy(dists_list[i]).float()
            if args.rm_feature:
                data.x = torch.ones((data.x.shape[0],1))
            data_list.append(data)
    else:
        data_list = []
        dists_list = []
        for i, data in enumerate(dataset):
            dists = precompute_dist_data(data.edge_index.numpy(), data.num_nodes, approximate=args.approximate)
            dists_list.append(dists)
            data.dists = torch.from_numpy(dists).float()
            if args.rm_feature:
                data.x = torch.ones((data.x.shape[0],1))
            data_list.append(data)

        with open(f1_name, 'wb') as f1:
            pickle.dump(dists_list, f1)
        print('Cache saved!')

    return data_list
