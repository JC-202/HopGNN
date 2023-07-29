import torch
import sys
import os.path as osp

from ogb.nodeproppred import PygNodePropPredDataset

sys.path.append('.')
from dgl.data import *
from torch_geometric.data import Data
from utils.graph_transform import sparse_normalize, remove_self_loop
from torch_sparse import SparseTensor
from datasets.hedata.dataset import load_nc_dataset
from datasets.hedata.data_utils import load_fixed_splits
from torch_geometric.datasets.flickr import Flickr
from torch_geometric.datasets.reddit import Reddit
import os


class NC_Data():
    def __init__(self, pyg_data, device):
        self.edge_index = pyg_data.edge_index.to(device)
        self.x = pyg_data.x.to(device)
        self.y = pyg_data.y.to(device)
        self.num_of_class = pyg_data.y.max().item()+1
        self.num_of_nodes = self.x.shape[0]
        self.name = pyg_data.name
        self.id_mask = torch.ones(self.x.shape[0]).bool().to(device)
        self.train_mask = pyg_data.train_mask.to(device)
        self.val_mask = pyg_data.val_mask.to(device)
        self.test_mask = pyg_data.test_mask.to(device)
        self.device = device
        self.init_matrix(pyg_data.edge_index)
        print('load %s dataset successfully!' % self.name)

    def init_matrix(self, edge_index):
        edge_index = remove_self_loop(edge_index)
        adj = SparseTensor(row=edge_index[0, :], col=edge_index[1, :], sparse_sizes=(self.num_of_nodes, self.num_of_nodes))
        adj = adj.to(self.device)
        if 'product' in self.name:
            sym_norm = False
        else:
            sym_norm = True
        self.adj = sparse_normalize(adj, sym_norm).to(self.device)

def load_data(data_name, device, split_id=0):
    data = load_graph_data(data_name, split_id=split_id)
    data = NC_Data(data, device)
    return data

def load_graph_data(data_name, split_id=0):
    if data_name in ['cora', 'citeseer', 'pubmed']:
        data = load_citation(data_name, split_id=split_id)
    elif data_name in ['chameleon', 'cornell', 'film', 'squirrel', 'texas', 'wisconsin']:
        data = load_heterophily_data(data_name, split_id=split_id)
    elif data_name in ['Flickr', 'Reddit']:
        data = load_inductive_data(data_name)
    elif 'product' in data_name:
        data = load_ogb_data()
    else:
        raise 'not implement graph dataset'
    data.name = data_name
    return data

def load_citation(data_name, split_id=0):
    if data_name == 'cora':
        dataset = CoraGraphDataset(verbose=False)
    elif data_name == 'citeseer':
        dataset = CiteseerGraphDataset(verbose=False)
    elif data_name == 'pubmed':
        dataset = PubmedGraphDataset(verbose=False)
    else:
        raise 'not implement citation dataset'

    g = dataset[0]
    edges = [a.long() for a in g.edges()]
    data = Data(x=g.ndata['feat'], y=g.ndata['label'], edge_index=torch.stack(edges))
    data.split_idxs, data.train_mask, data.val_mask, data.test_mask = load_pyg_splits(data_name, split_id)
    return data

def load_heterophily_data(dataset_name, split_id=0,):
    if dataset_name in ['chameleon', 'cornell', 'film', 'squirrel', 'texas', 'wisconsin']:
        dataset = load_nc_dataset(dataset_name, '')
        graph = dataset.graph
        data = Data(x=graph['node_feat'],
                    y=dataset.label,
                    edge_index=graph['edge_index'])
        data.split_idxs, data.train_mask, data.val_mask, data.test_mask = load_pyg_splits(dataset_name, split_id)
        return data
    else:
        raise 'not implement pyg dataset'

def load_pyg_splits(dataset_name, split_id=0):
    split_idx_lst = load_fixed_splits(dataset_name, '')
    train_mask = split_idx_lst[split_id]['train']
    val_mask = split_idx_lst[split_id]['valid']
    test_mask = split_idx_lst[split_id]['test']
    return split_idx_lst, train_mask, val_mask, test_mask

#load inductive
def load_inductive_data(data_name):
    if data_name in ['Flickr', 'Reddit']:
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..', 'dataset', data_name)
        root = path if os.path.exists(path) else './dataset/{}'.format(data_name)
        if data_name == 'Flickr':
            data = Flickr(root)[0]
        elif data_name == 'Reddit':
            data = Reddit(root)[0]
        else:
            raise 'not implement'
        id_mask = torch.arange(data.x.shape[0])
        data.train_mask = id_mask[data.train_mask]
        data.val_mask = id_mask[data.val_mask]
        data.test_mask = id_mask[data.test_mask]
    else:
        raise 'not implement'
    return data

#load ogb product dataset
def load_ogb_data():
    root = osp.join(osp.dirname(osp.realpath(__file__)), '..',  '..', 'dataset', 'ogbn-products')
    dataset = PygNodePropPredDataset(name='ogbn-products', root=root)
    splitted_idx = dataset.get_idx_split()
    train_mask, val_mask, test_mask = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
    data = dataset[0]
    data.y = data.y.squeeze()
    data.train_mask, data.val_mask, data.test_mask = train_mask, val_mask, test_mask
    return data