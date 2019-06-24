import community
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import tensorboardX
import multiprocessing as mp
import random
import torch

def imsave(fname, arr, vmin=None, vmax=None, cmap=None, format=None, origin=None):
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure

    fig = Figure(figsize=arr.shape[::-1], dpi=1, frameon=False)
    canvas = FigureCanvas(fig)
    fig.figimage(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin=origin)
    fig.savefig(fname, dpi=1, format=format)

def plot_graph(plt, G):
    plt.title('num of nodes: '+str(G.number_of_nodes()), fontsize = 4)
    parts = community.best_partition(G)
    values = [parts.get(node) for node in G.nodes()]
    colors = []
    for i in range(len(values)):
        if values[i] == 0:
            colors.append('red')
        if values[i] == 1:
            colors.append('green')
        if values[i] == 2:
            colors.append('blue')
        if values[i] == 3:
            colors.append('yellow')
        if values[i] == 4:
            colors.append('orange')
        if values[i] == 5:
            colors.append('pink')
        if values[i] == 6:
            colors.append('black')
    plt.axis("off")
    pos = nx.spring_layout(G)
    # pos = nx.spectral_layout(G)
    nx.draw_networkx(G, with_labels=True, node_size=4, width=0.3, font_size = 3, node_color=colors,pos=pos)

def draw_graph_list(G_list, row, col, fname = 'figs/test'):
    # draw graph view
    plt.switch_backend('agg')
    for i, G in enumerate(G_list):
        plt.subplot(row,col,i+1)
        plot_graph(plt, G)
        
    plt.tight_layout()
    plt.savefig(fname+'_view.png', dpi=600)
    plt.close()

    # draw degree distribution
    plt.switch_backend('agg')
    for i, G in enumerate(G_list):
        plt.subplot(row, col, i + 1)
        G_deg = np.array(list(G.degree(G.nodes()).values()))
        bins = np.arange(20)
        plt.hist(np.array(G_deg), bins=bins, align='left')
        plt.xlabel('degree', fontsize = 3)
        plt.ylabel('count', fontsize = 3)
        G_deg_mean = 2*G.number_of_edges()/float(G.number_of_nodes())
        plt.title('average degree: {:.2f}'.format(G_deg_mean), fontsize=4)
        plt.tick_params(axis='both', which='major', labelsize=3)
        plt.tick_params(axis='both', which='minor', labelsize=3)
    plt.tight_layout()
    plt.savefig(fname+'_degree.png', dpi=600)
    plt.close()

    # degree_sequence = sorted(nx.degree(G).values(), reverse=True)  # degree sequence
    # plt.loglog(degree_sequence, 'b-', marker='o')
    # plt.title("Degree rank plot")
    # plt.ylabel("degree")
    # plt.xlabel("rank")
    # plt.savefig('figures/degree_view_' + prefix + '.png', dpi=200)
    # plt.close()

    # draw clustering distribution
    #plt.switch_backend('agg')
    #for i, G in enumerate(G_list):
    #    plt.subplot(row, col, i + 1)
    #    G_cluster = list(nx.clustering(G).values())
    #    bins = np.linspace(0,1,20)
    #    plt.hist(np.array(G_cluster), bins=bins, align='left')
    #    plt.xlabel('clustering coefficient', fontsize=3)
    #    plt.ylabel('count', fontsize=3)
    #    G_cluster_mean = sum(G_cluster) / len(G_cluster)
    #    # if i % 2 == 0:
    #    #     plt.title('real average clustering: {:.4f}'.format(G_cluster_mean), fontsize=4)
    #    # else:
    #    #     plt.title('pred average clustering: {:.4f}'.format(G_cluster_mean), fontsize=4)
    #    plt.title('average clustering: {:.4f}'.format(G_cluster_mean), fontsize=4)
    #    plt.tick_params(axis='both', which='major', labelsize=3)
    #    plt.tick_params(axis='both', which='minor', labelsize=3)
    #plt.tight_layout()
    #plt.savefig(fname+'_clustering.png', dpi=600)
    #plt.close()

    ## draw circle distribution
    #plt.switch_backend('agg')
    #for i, G in enumerate(G_list):
    #    plt.subplot(row, col, i + 1)
    #    cycle_len = []
    #    cycle_all = nx.cycle_basis(G)
    #    for item in cycle_all:
    #        cycle_len.append(len(item))

    #    bins = np.arange(20)
    #    plt.hist(np.array(cycle_len), bins=bins, align='left')
    #    plt.xlabel('cycle length', fontsize=3)
    #    plt.ylabel('count', fontsize=3)
    #    G_cycle_mean = 0
    #    if len(cycle_len)>0:
    #        G_cycle_mean = sum(cycle_len) / len(cycle_len)
    #    # if i % 2 == 0:
    #    #     plt.title('real average cycle: {:.4f}'.format(G_cycle_mean), fontsize=4)
    #    # else:
    #    #     plt.title('pred average cycle: {:.4f}'.format(G_cycle_mean), fontsize=4)
    #    plt.title('average cycle: {:.4f}'.format(G_cycle_mean), fontsize=4)
    #    plt.tick_params(axis='both', which='major', labelsize=3)
    #    plt.tick_params(axis='both', which='minor', labelsize=3)
    #plt.tight_layout()
    #plt.savefig(fname+'_cycle.png', dpi=600)
    #plt.close()

    ## draw community distribution
    #plt.switch_backend('agg')
    #for i, G in enumerate(G_list):
    #    plt.subplot(row, col, i + 1)
    #    parts = community.best_partition(G)
    #    values = np.array([parts.get(node) for node in G.nodes()])
    #    counts = np.sort(np.bincount(values)[::-1])
    #    pos = np.arange(len(counts))
    #    plt.bar(pos,counts,align = 'edge')
    #    plt.xlabel('community ID', fontsize=3)
    #    plt.ylabel('count', fontsize=3)
    #    G_community_count = len(counts)
    #    # if i % 2 == 0:
    #    #     plt.title('real average clustering: {}'.format(G_community_count), fontsize=4)
    #    # else:
    #    #     plt.title('pred average clustering: {}'.format(G_community_count), fontsize=4)
    #    plt.title('average clustering: {}'.format(G_community_count), fontsize=4)
    #    plt.tick_params(axis='both', which='major', labelsize=3)
    #    plt.tick_params(axis='both', which='minor', labelsize=3)
    #plt.tight_layout()
    #plt.savefig(fname+'_community.png', dpi=600)
    #plt.close()


def exp_moving_avg(x, decay=0.9):
    shadow = x[0]
    a = [shadow]
    for v in x[1:]:
        shadow -= (1-decay) * (shadow-v)
        a.append(shadow)
    return a


def gen_prefix(args):
    if args.bmname is not None:
        name = args.bmname
    else:
        name = args.dataset
    name += '_' + args.method
    if args.method == 'soft-assign':
        name += '_l' + str(args.num_gc_layers) + 'x' + str(args.num_pool)
        name += '_ar' + str(int(args.assign_ratio*100))
        if args.linkpred:
            name += '_lp'
    else:
        name += '_l' + str(args.num_gc_layers)
    name += '_h' + str(args.hidden_dim) + '_o' + str(args.output_dim)
    if not args.bias:
        name += '_nobias'
    if len(args.name_suffix) > 0:
        name += '_' + args.name_suffix
    return name

def gen_train_plt_name(args):
    return 'results/' + gen_prefix(args) + '.png'

def log_assignment(assign_tensor, writer, epoch, batch_idx):
    plt.switch_backend('agg')
    fig = plt.figure(figsize=(8,6), dpi=300)

    # has to be smaller than args.batch_size
    for i in range(len(batch_idx)):
        plt.subplot(2, 2, i+1)
        plt.imshow(assign_tensor.cpu().data.numpy()[batch_idx[i]], cmap=plt.get_cmap('BuPu'))
        cbar = plt.colorbar()
        cbar.solids.set_edgecolor("face")
    plt.tight_layout()
    fig.canvas.draw()

    #data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    #data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = tensorboardX.utils.figure_to_image(fig)
    writer.add_image('assignment', data, epoch)

def log_graph(adj, batch_num_nodes, writer, epoch, batch_idx, assign_tensor=None):
    plt.switch_backend('agg')
    fig = plt.figure(figsize=(8,6), dpi=300)

    for i in range(len(batch_idx)):
        ax = plt.subplot(2, 2, i+1)
        num_nodes = batch_num_nodes[batch_idx[i]]
        adj_matrix = adj[batch_idx[i], :num_nodes, :num_nodes].cpu().data.numpy()
        G = nx.from_numpy_matrix(adj_matrix)
        nx.draw(G, pos=nx.spring_layout(G), with_labels=True, node_color='#336699',
                edge_color='grey', width=0.5, node_size=300,
                alpha=0.7)
        ax.xaxis.set_visible(False)

    plt.tight_layout()
    fig.canvas.draw()

    #data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    #data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = tensorboardX.utils.figure_to_image(fig)
    writer.add_image('graphs', data, epoch)

    # log a label-less version
    #fig = plt.figure(figsize=(8,6), dpi=300)
    #for i in range(len(batch_idx)):
    #    ax = plt.subplot(2, 2, i+1)
    #    num_nodes = batch_num_nodes[batch_idx[i]]
    #    adj_matrix = adj[batch_idx[i], :num_nodes, :num_nodes].cpu().data.numpy()
    #    G = nx.from_numpy_matrix(adj_matrix)
    #    nx.draw(G, pos=nx.spring_layout(G), with_labels=False, node_color='#336699',
    #            edge_color='grey', width=0.5, node_size=25,
    #            alpha=0.8)

    #plt.tight_layout()
    #fig.canvas.draw()

    #data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    #data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #writer.add_image('graphs_no_label', data, epoch)

    # colored according to assignment
    assignment = assign_tensor.cpu().data.numpy()
    fig = plt.figure(figsize=(8,6), dpi=300)

    num_clusters = assignment.shape[2]
    all_colors = np.array(range(num_clusters))

    for i in range(len(batch_idx)):
        ax = plt.subplot(2, 2, i+1)
        num_nodes = batch_num_nodes[batch_idx[i]]
        adj_matrix = adj[batch_idx[i], :num_nodes, :num_nodes].cpu().data.numpy()

        label = np.argmax(assignment[batch_idx[i]], axis=1).astype(int)
        label = label[: batch_num_nodes[batch_idx[i]]]
        node_colors = all_colors[label]

        G = nx.from_numpy_matrix(adj_matrix)
        nx.draw(G, pos=nx.spring_layout(G), with_labels=False, node_color=node_colors,
                edge_color='grey', width=0.4, node_size=50, cmap=plt.get_cmap('Set1'),
                vmin=0, vmax=num_clusters-1,
                alpha=0.8)

    plt.tight_layout()
    fig.canvas.draw()

    #data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    #data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = tensorboardX.utils.figure_to_image(fig)
    writer.add_image('graphs_colored', data, epoch)


def single_source_shortest_path_length_range(graph, node_range, cutoff):
    dists_dict = {}
    for node in node_range:
        dists_dict[node] = nx.single_source_shortest_path_length(graph, node, cutoff)
    return dists_dict


def merge_dicts(dicts):
    result = {}
    for dictionary in dicts:
        result.update(dictionary)
    return result


def all_pairs_shortest_path_length_parallel(graph,cutoff=None,num_workers=4):
    nodes = list(graph.nodes)
    random.shuffle(nodes)
    if len(nodes)<50:
        num_workers = int(num_workers/4)
    elif len(nodes)<400:
        num_workers = int(num_workers/2)

    pool = mp.Pool(processes=num_workers)
    results = [pool.apply_async(single_source_shortest_path_length_range,
            args=(graph, nodes[int(len(nodes)/num_workers*i):int(len(nodes)/num_workers*(i+1))], cutoff)) for i in range(num_workers)]
    output = [p.get() for p in results]
    dists_dict = merge_dicts(output)
    pool.close()
    pool.join()
    return dists_dict


def precompute_dist_data(edge_index, num_nodes, approximate=0):
        '''
        Here dist is 1/real_dist, higher actually means closer, 0 means disconnected
        :return:
        '''
        graph = nx.Graph()
        edge_list = edge_index.transpose(1,0).tolist()
        graph.add_edges_from(edge_list)

        n = num_nodes
        dists_array = np.zeros((n, n))
        # dists_dict = nx.all_pairs_shortest_path_length(graph,cutoff=approximate if approximate>0 else None)
        # dists_dict = {c[0]: c[1] for c in dists_dict}
        dists_dict = all_pairs_shortest_path_length_parallel(graph,cutoff=approximate if approximate>0 else None)
        for i, node_i in enumerate(graph.nodes()):
            shortest_dist = dists_dict[node_i]
            for j, node_j in enumerate(graph.nodes()):
                dist = shortest_dist.get(node_j, -1)
                if dist!=-1:
                    # dists_array[i, j] = 1 / (dist + 1)
                    dists_array[node_i, node_j] = 1 / (dist + 1)
        return dists_array


def get_random_anchorset(n,c=0.5):
    m = int(np.log2(n))
    copy = int(c*m)
    anchorset_id = []
    for i in range(m):
        anchor_size = int(n/np.exp2(i + 1))
        for j in range(copy):
            anchorset_id.append(np.random.choice(n,size=anchor_size,replace=False))
    return anchorset_id


def get_dist_max(anchorset_id, dist, device):
    dist_max = torch.zeros((dist.shape[0],len(anchorset_id))).to(device)
    dist_argmax = torch.zeros((dist.shape[0],len(anchorset_id))).long().to(device)
    for i in range(len(anchorset_id)):
        temp_id = anchorset_id[i]
        dist_temp = dist[:, temp_id]
        dist_max_temp, dist_argmax_temp = torch.max(dist_temp, dim=-1)
        dist_max[:,i] = dist_max_temp
        # Original dist_argmax which seems have some problem
        # dist_argmax[:,i] = dist_argmax_temp

        temp_id_ts = torch.LongTensor(temp_id).unsqueeze(0).repeat(dist.shape[0], 1).to(device)
        dist_argmax[:, i] = temp_id_ts.gather(1, dist_argmax_temp.view(-1, 1)).squeeze()

    return dist_max, dist_argmax


def preselect_anchor(data, layer_num=1, anchor_num=32, anchor_size_num=4, device='cpu'):

    data.anchor_size_num = anchor_size_num
    data.anchor_set = []
    anchor_num_per_size = anchor_num//anchor_size_num
    for i in range(anchor_size_num):
        anchor_size = 2**(i+1)-1
        anchors = np.random.choice(data.num_nodes, size=(layer_num,anchor_num_per_size,anchor_size), replace=True)
        data.anchor_set.append(anchors)
    data.anchor_set_indicator = np.zeros((layer_num, anchor_num, data.num_nodes), dtype=int)

    anchorset_id = get_random_anchorset(data.num_nodes,c=1)
    data.dists_max, data.dists_argmax = get_dist_max(anchorset_id, data.dists, device)


def replace_with_dict(array, origin, dest):

    # get argsort indices
    sidx = origin.argsort()
    ks = origin[sidx]
    vs = dest[sidx]

    return vs[np.searchsorted(ks, array)]


def preselect_anchor_pooling(data, pooling_num=0):

    original_nodes = data.edge_index.unique()
    data.nodes_set = []
    data.edges_set = []
    data.dists_set = []
    data.paired_nodes = []
    data.nodes_set.append(original_nodes)
    data.edges_set.append(data.edge_index)
    data.dists_set.append(data.dists)
    data.mapped_nodes_set = []
    # select anchor nodes
    for i in range(pooling_num):
        # random select the anchor nodes
        remaining_nodes = data.nodes_set[-1]
        num_anchor_nodes = int(len(remaining_nodes) / 2)
        if num_anchor_nodes == 0:
            break
        anchor_nodes_id = np.random.choice(len(remaining_nodes), num_anchor_nodes, replace=False)
        anchor_nodes = remaining_nodes[anchor_nodes_id]

        # get the anchor nodes pair
        tmp_dist = data.dists_set[-1]
        dist_temp = tmp_dist[:, anchor_nodes_id]
        dist_max_temp, dist_argmax_temp = torch.max(dist_temp, dim=-1)
        mapped_nodes = remaining_nodes[anchor_nodes_id[dist_argmax_temp]]
        # check whether the operation is correct or not
        assert torch.sum(remaining_nodes == mapped_nodes).item() == num_anchor_nodes
        edge_index_np = data.edges_set[-1].numpy()
        new_edge_index_np = replace_with_dict(edge_index_np, remaining_nodes.numpy(), mapped_nodes.numpy())
        # remove self-loop edges and repeat edges
        self_loop_idx = new_edge_index_np[0,:] != new_edge_index_np[1, :]
        new_edge_index_np = new_edge_index_np[:, self_loop_idx]
        new_edge_index_np = np.unique(new_edge_index_np, axis=1)
        new_edge_index = torch.from_numpy(new_edge_index_np).long()
        data.edges_set.append(new_edge_index)
        data.nodes_set.append(anchor_nodes)
        data.dists_set.append(dist_temp[anchor_nodes_id, :])
        data.mapped_nodes_set.append(mapped_nodes)

