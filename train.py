import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
from torch.autograd import Variable
from tensorboardX import SummaryWriter

import os
import pickle
import shutil
import time

import encoders
import gen.feat as featgen
import gen.data as datagen
import load_data
import util
from args import *

def evaluate(dataset, model, args, name='Validation', max_num_examples=None):
    model.eval()

    labels = []
    preds = []
    for batch_idx, data in enumerate(dataset):
        adj = Variable(data['adj'].float(), requires_grad=False).cuda()
        h0 = Variable(data['feats'].float()).cuda()
        labels.append(data['label'].long().numpy())
        batch_num_nodes = data['num_nodes'].int().numpy()
        assign_input = Variable(data['assign_feats'].float(), requires_grad=False).cuda()

        ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
        _, indices = torch.max(ypred, 1)
        preds.append(indices.cpu().data.numpy())

        if max_num_examples is not None:
            if (batch_idx+1)*args.batch_size > max_num_examples:
                break

    labels = np.hstack(labels)
    preds = np.hstack(preds)
    
    result = {'prec': metrics.precision_score(labels, preds, average='macro'),
              'recall': metrics.recall_score(labels, preds, average='macro'),
              'acc': metrics.accuracy_score(labels, preds),
              'F1': metrics.f1_score(labels, preds, average="micro")}
    print(name, " accuracy:", result['acc'])
    return result


def train(dataset, model, args, same_feat=True, val_dataset=None, test_dataset=None, writer=None,
        mask_nodes = True):
    writer_batch_idx = [0, 3, 6, 9]
    
    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=0.001)
    iter = 0
    best_val_result = {
            'epoch': 0,
            'loss': 0,
            'acc': 0}
    test_result = {
            'epoch': 0,
            'loss': 0,
            'acc': 0}
    train_accs = []
    train_epochs = []
    best_val_accs = []
    best_val_epochs = []
    test_accs = []
    test_epochs = []
    val_accs = []
    for epoch in range(args.num_epochs):
        total_time = 0
        avg_loss = 0.0
        model.train()
        print('Epoch: ', epoch)
        for batch_idx, data in enumerate(dataset):
            begin_time = time.time()
            model.zero_grad()
            adj = Variable(data['adj'].float(), requires_grad=False).cuda()
            h0 = Variable(data['feats'].float(), requires_grad=False).cuda()
            label = Variable(data['label'].long()).cuda()
            batch_num_nodes = data['num_nodes'].int().numpy() if mask_nodes else None
            assign_input = Variable(data['assign_feats'].float(), requires_grad=False).cuda()

            ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
            if not args.method == 'soft-assign' or not args.linkpred:
                loss = model.loss(ypred, label)
            else:
                loss = model.loss(ypred, label, adj, batch_num_nodes)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), args.clip)
            optimizer.step()
            iter += 1
            avg_loss += loss
            #if iter % 20 == 0:
            #    print('Iter: ', iter, ', loss: ', loss.data[0])
            elapsed = time.time() - begin_time
            total_time += elapsed

            # log once per XX epochs
            if epoch % 10 == 0 and batch_idx == len(dataset) // 2 and args.method == 'soft-assign' and writer is not None:
                util.log_assignment(model.assign_tensor, writer, epoch, writer_batch_idx)
                if args.log_graph:
                    util.log_graph(adj, batch_num_nodes, writer, epoch, writer_batch_idx, model.assign_tensor)
        avg_loss /= batch_idx + 1
        if writer is not None:
            writer.add_scalar('loss/avg_loss', avg_loss, epoch)
            if args.linkpred:
                writer.add_scalar('loss/linkpred_loss', model.link_loss, epoch)
        print('Avg loss: ', avg_loss, '; epoch time: ', total_time)
        result = evaluate(dataset, model, args, name='Train', max_num_examples=100)
        train_accs.append(result['acc'])
        train_epochs.append(epoch)
        if val_dataset is not None:
            val_result = evaluate(val_dataset, model, args, name='Validation')
            val_accs.append(val_result['acc'])
        if val_result['acc'] > best_val_result['acc'] - 1e-7:
            best_val_result['acc'] = val_result['acc']
            best_val_result['epoch'] = epoch
            best_val_result['loss'] = avg_loss
        if test_dataset is not None:
            test_result = evaluate(test_dataset, model, args, name='Test')
            test_result['epoch'] = epoch
        if writer is not None:
            writer.add_scalar('acc/train_acc', result['acc'], epoch)
            writer.add_scalar('acc/val_acc', val_result['acc'], epoch)
            writer.add_scalar('loss/best_val_loss', best_val_result['loss'], epoch)
            if test_dataset is not None:
                writer.add_scalar('acc/test_acc', test_result['acc'], epoch)

        print('Best val result: ', best_val_result)
        best_val_epochs.append(best_val_result['epoch'])
        best_val_accs.append(best_val_result['acc'])
        if test_dataset is not None:
            print('Test result: ', test_result)
            test_epochs.append(test_result['epoch'])
            test_accs.append(test_result['acc'])

    matplotlib.style.use('seaborn')
    plt.switch_backend('agg')
    plt.figure()
    plt.plot(train_epochs, util.exp_moving_avg(train_accs, 0.85), '-', lw=1)
    if test_dataset is not None:
        plt.plot(best_val_epochs, best_val_accs, 'bo', test_epochs, test_accs, 'go')
        plt.legend(['train', 'val', 'test'])
    else:
        plt.plot(best_val_epochs, best_val_accs, 'bo')
        plt.legend(['train', 'val'])
    plt.savefig(util.gen_train_plt_name(args), dpi=600)
    plt.close()
    matplotlib.style.use('default')

    return model, val_accs

def syn_community1v2(args, writer=None, export_graphs=False):

    # data
    graphs1 = datagen.gen_ba(range(40, 60), range(4, 5), 500, 
            featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float)))
    for G in graphs1:
        G.graph['label'] = 0
    if export_graphs:
        util.draw_graph_list(graphs1[:16], 4, 4, 'figs/ba')

    graphs2 = datagen.gen_2community_ba(range(20, 30), range(4, 5), 500, 0.3, 
            [featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))])
    for G in graphs2:
        G.graph['label'] = 1
    if export_graphs:
        util.draw_graph_list(graphs2[:16], 4, 4, 'figs/ba2')

    graphs = graphs1 + graphs2
    
    train_dataset, val_dataset, test_dataset, max_num_nodes, input_dim, assign_input_dim = load_data.prepare_data(graphs, args)
    if args.method == 'soft-assign':
        print('Method: soft-assign')
        model = encoders.SoftPoolingGcnEncoder(
                max_num_nodes, 
                input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
                args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
                bn=args.bn, linkpred=args.linkpred, assign_input_dim=assign_input_dim).cuda()
    elif args.method == 'base-set2set':
        print('Method: base-set2set')
        model = encoders.GcnSet2SetEncoder(input_dim, args.hidden_dim, args.output_dim, 2,
                args.num_gc_layers, bn=args.bn).cuda()
    else:
        print('Method: base')
        model = encoders.GcnEncoderGraph(input_dim, args.hidden_dim, args.output_dim, 2,
                args.num_gc_layers, bn=args.bn).cuda()

    train(train_dataset, model, args, val_dataset=val_dataset, test_dataset=test_dataset,
            writer=writer)

def syn_community2hier(args, writer=None):

    # data
    feat_gen = [featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))]
    graphs1 = datagen.gen_2hier(1000, [2,4], 10, range(4,5), 0.1, 0.03, feat_gen)
    graphs2 = datagen.gen_2hier(1000, [3,3], 10, range(4,5), 0.1, 0.03, feat_gen)
    graphs3 = datagen.gen_2community_ba(range(28, 33), range(4,7), 1000, 0.25, feat_gen)

    for G in graphs1:
        G.graph['label'] = 0
    for G in graphs2:
        G.graph['label'] = 1
    for G in graphs3:
        G.graph['label'] = 2

    graphs = graphs1 + graphs2 + graphs3

    train_dataset, val_dataset, test_dataset, max_num_nodes, input_dim, assign_input_dim = load_data.prepare_data(graphs, args)

    if args.method == 'soft-assign':
        print('Method: soft-assign')
        model = encoders.SoftPoolingGcnEncoder(
                max_num_nodes, 
                input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
                args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
                bn=args.bn, linkpred=args.linkpred, args=args, assign_input_dim=assign_input_dim).cuda()
    elif args.method == 'base-set2set':
        print('Method: base-set2set')
        model = encoders.GcnSet2SetEncoder(input_dim, args.hidden_dim, args.output_dim, 2,
                args.num_gc_layers, bn=args.bn, args=args, assign_input_dim=assign_input_dim).cuda()
    else:
        print('Method: base')
        model = encoders.GcnEncoderGraph(input_dim, args.hidden_dim, args.output_dim, 2,
                args.num_gc_layers, bn=args.bn, args=args).cuda()
    train(train_dataset, model, args, val_dataset=val_dataset, test_dataset=test_dataset,
            writer=writer)


def pkl_task(args, feat=None):
    with open(os.path.join(args.datadir, args.pkl_fname), 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    graphs = data[0]
    labels = data[1]
    test_graphs = data[2]
    test_labels = data[3]

    for i in range(len(graphs)):
        graphs[i].graph['label'] = labels[i]
    for i in range(len(test_graphs)):
        test_graphs[i].graph['label'] = test_labels[i]

    if feat is None:
        featgen_const = featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
        for G in graphs:
            featgen_const.gen_node_features(G)
        for G in test_graphs:
            featgen_const.gen_node_features(G)

    train_dataset, test_dataset, max_num_nodes = load_data.prepare_data(graphs, args, test_graphs=test_graphs)
    model = encoders.GcnEncoderGraph(
            args.input_dim, args.hidden_dim, args.output_dim, args.num_classes, 
            args.num_gc_layers, bn=args.bn).cuda()
    train(train_dataset, model, args, test_dataset=test_dataset)
    evaluate(test_dataset, model, args, 'Validation')

def benchmark_task(args, writer=None, feat='node-label'):
    """

    :param args:
    :param writer:
    :param feat:
    :return:
    """
    # load graphs from file: graphs = [nx.Graph]
    graphs = load_data.read_graphfile(args.datadir, args.bmname, max_nodes=args.max_nodes)

    if feat == 'node-feat' and 'feat_dim' in graphs[0].graph:
        print('Using node features')
        input_dim = graphs[0].graph['feat_dim']
    elif feat == 'node-label' and 'label' in graphs[0].node[0]:
        print('Using node labels')
        for G in graphs:
            for u in G.nodes():
                G.node[u]['feat'] = np.array(G.node[u]['label'])
    else:
        print('Using constant labels')
        featgen_const = featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
        for G in graphs:
            featgen_const.gen_node_features(G)

    train_dataset, val_dataset, test_dataset, max_num_nodes, input_dim, assign_input_dim = \
            load_data.prepare_data(graphs, args, max_nodes=args.max_nodes)
    if args.method == 'soft-assign':
        print('Method: soft-assign')
        model = encoders.SoftPoolingGcnEncoder(
                max_num_nodes, 
                input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
                args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
                bn=args.bn, dropout=args.dropout, linkpred=args.linkpred, args=args,
                assign_input_dim=assign_input_dim).cuda()
    elif args.method == 'base-set2set':
        print('Method: base-set2set')
        model = encoders.GcnSet2SetEncoder(
                input_dim, args.hidden_dim, args.output_dim, args.num_classes,
                args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args).cuda()
    else:
        print('Method: base')
        model = encoders.GcnEncoderGraph(
                input_dim, args.hidden_dim, args.output_dim, args.num_classes, 
                args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args).cuda()

    train(train_dataset, model, args, val_dataset=val_dataset, test_dataset=test_dataset,
            writer=writer)
    evaluate(test_dataset, model, args, 'Validation')


def benchmark_task_val(args, writer=None, feat='node-label'):
    all_vals = []
    # graphs = load_data.read_graphfile(args.datadir, args.bmname, max_nodes=args.max_nodes)
    
    if feat == 'node-feat':
        print('Using node features')
        graphs_all, features_all, _ = load_data.Graph_load_batch(
            args.datadir, args.bmname, max_nodes=args.max_nodes, node_attributes=True)
    elif feat == 'node-label':
        print('Using node labels')
        graphs_all, features_all, _ = load_data.Graph_load_batch(
            args.datadir, args.bmname, max_nodes=args.max_nodes, node_attributes=False)
    else:
        print('Using constant labels')
        graphs_all, features_all, _ = load_data.Graph_load_batch(
            args.datadir, args.bmname, max_nodes=args.max_nodes, node_attributes=False)
        featgen_const = featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
        for G in graphs_all:
            featgen_const.gen_node_features(G)

    # TODO: 1. cvt numpy to tg.Data; 2. Assign Adj Matrix
    dataset = load_data.nx_to_tg_data(graphs_all, features_all)


    for i in range(10):
        train_dataset, val_dataset, max_num_nodes, input_dim, assign_input_dim = \
                load_data.prepare_val_data(graphs, args, i, max_nodes=args.max_nodes)
        if args.method == 'soft-assign':
            print('Method: soft-assign')
            model = encoders.SoftPoolingGcnEncoder(
                    max_num_nodes, 
                    input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
                    args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
                    bn=args.bn, dropout=args.dropout, linkpred=args.linkpred, args=args,
                    assign_input_dim=assign_input_dim).cuda()
        elif args.method == 'base-set2set':
            print('Method: base-set2set')
            model = encoders.GcnSet2SetEncoder(
                    input_dim, args.hidden_dim, args.output_dim, args.num_classes,
                    args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args).cuda()
        else:
            print('Method: base')
            model = encoders.GcnEncoderGraph(
                    input_dim, args.hidden_dim, args.output_dim, args.num_classes, 
                    args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args).cuda()

        _, val_accs = train(train_dataset, model, args, val_dataset=val_dataset, test_dataset=None,
            writer=writer)
        all_vals.append(np.array(val_accs))
    all_vals = np.vstack(all_vals)
    all_vals = np.mean(all_vals, axis=0)
    print(all_vals)
    print(np.max(all_vals))
    print(np.argmax(all_vals))


def main():
    prog_args = arg_parse()

    # export scalar data to JSON for external processing
    path = os.path.join(prog_args.logdir, util.gen_prefix(prog_args))
    if os.path.isdir(path):
        print('Remove existing log dir: ', path)
        shutil.rmtree(path)
    writer = SummaryWriter(path)
    #writer = None

    os.environ['CUDA_VISIBLE_DEVICES'] = prog_args.cuda
    print('CUDA', prog_args.cuda)

    if prog_args.bmname is not None:
        benchmark_task_val(prog_args, writer=writer)
    elif prog_args.pkl_fname is not None:
        pkl_task(prog_args)
    elif prog_args.dataset is not None:
        if prog_args.dataset == 'syn1v2':
            syn_community1v2(prog_args, writer=writer)
        if prog_args.dataset == 'syn2hier':
            syn_community2hier(prog_args, writer=writer)

    writer.close()

if __name__ == "__main__":
    main()

