import torch
import sys

sys.path.append('..')
from models.hopgnn import HopGNN
import warnings
warnings.filterwarnings('ignore')
from datasets.dataloader import load_data
from utils.utils import *
import argparse

parser = argparse.ArgumentParser()

# Training hyper-parameters
parser.add_argument('--seed', type=int, default=202, help='set Random seed.')
parser.add_argument('--dataset', type=str, default='cora', help='name of dataset')
parser.add_argument('--split_id', type=int, default=0, help='split_id of dataset')
parser.add_argument('--cuda_id', type=str, default='0', help='CUDA id')
parser.add_argument('--epochs', type=int, default=2500, help='number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=40000, help="batch size for the mini-batch")
parser.add_argument('--log_dur', type=int, default=50, help='interval of epochs for log during training.')
parser.add_argument('--lr', type=float, default=5e-2, help='initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay (L2 loss on parameters).')
parser.add_argument('--ssl', type=bool, default=0, help="use hopgnn+ssl or not")
parser.add_argument('--alpha', type=float, default=0.5, help="the alpha of de-correlation in Barlow Twins")
parser.add_argument('--lambd', type=float, default=5e-4, help="the weight of SSL objective")

# Model hyper-parameters
parser.add_argument('--model', type=str, default='hopgnn', help='which models')
parser.add_argument('--hidden', type=int, default=128, help='number of hidden units.')
parser.add_argument('--num_layer', type=int, default=2, help='number of interaction layer')
parser.add_argument('--num_hop', type=int, default=6, help='number of hop information')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate (1 - keep probability).')
parser.add_argument('--interaction', type=str, default='attention', help='feature interaction type of HopGNN')
parser.add_argument('--fusion', type=str, default='mean', help='feature fusion type of HopGNN')
parser.add_argument('--activation', type=str, default='relu', help="activation function")
parser.add_argument('--norm_type', type=str, default='ln', help="the normalization type")
args = parser.parse_args()

def get_loss(model, features, labels, args):
    loss_func = torch.nn.CrossEntropyLoss()
    #HopGNN+ SSL
    if args.ssl == 1 and isinstance(model, HopGNN):
        ssl_func = BarlowLoss(args.alpha)
        (y1, y2), (view1, view2) = model.forward_plus(features)
        ce_loss = loss_func(y1, labels) + loss_func(y2, labels) + 1e-9
        ssl_loss = ssl_func(view1, view2)
        output = y1
        loss = ce_loss + args.lambd * ssl_loss
    else:
        output = model(features)
        loss = loss_func(output, labels) + 1e-9

    return output, loss

#mini-batch training
def roll_an_epoch(model, features, labels, mask, optimizer, batch_size, manner, args):
    if manner == 'train':
        # shuffle the train mask for mini-batch training
        mask = mask[torch.randperm(len(mask))]

    device = features.device
    total_loss = []
    total_output = []
    total_label = []

    for i in range(0, len(mask), batch_size):
        # generate batch index , features, label
        index = mask[i:i + batch_size]
        batch_features = features[index].to(device)
        batch_label = labels[index].to(device)
        if manner == 'train':
            model.train()
            batch_output, loss = get_loss(model, batch_features, batch_label, args)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                model.eval()
                batch_output, loss = get_loss(model, batch_features, batch_label, args)

        total_loss.append(loss.cpu().item())
        total_output.append(batch_output)
        total_label.append(batch_label)

    loss = np.mean(total_loss)
    total_output = torch.cat(total_output, dim=0)
    total_label = torch.cat(total_label)
    acc = get_accuracy(total_output, total_label)
    log_info = {'loss': loss, 'acc': acc}
    return log_info

def train(epoch, model, features, labels, train_mask, val_mask, test_mask, args):
    lr = args.lr
    weight_decay = args.weight_decay
    batch_size = args.batch_size
    set_seed(202)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loss_list, val_loss_list, train_acc_list, val_acc_list, test_acc_list = [], [], [], [], []
    best_val_acc, best_test_acc = 0, 0
    best_epoch = 0
    for iter in range(epoch):
        train_log_info = roll_an_epoch(model, features, labels, train_mask, optimizer, batch_size=batch_size, manner='train', args=args)
        val_log_info = roll_an_epoch(model, features, labels, val_mask, optimizer, batch_size=batch_size, manner='val', args=args)
        test_log_info = roll_an_epoch(model, features, labels, test_mask, optimizer, batch_size=batch_size, manner='test', args=args)

        #log info
        train_loss_list.append(train_log_info['loss'])
        val_loss_list.append(val_log_info['loss'])
        train_acc, val_acc, test_acc = train_log_info['acc'], val_log_info['acc'], test_log_info['acc']
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)

        #update best test via val
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_epoch = iter

        if (iter + 1) % args.log_dur == 0:
            print(
                "Epoch {:4d}, Train_loss {:.4f}, Val_loss {:.4f}, train_acc {:.4f},  val_acc {:.4f}, test_acc{:.4f}".format(
                    iter + 1, np.mean(train_loss_list), np.mean(val_loss_list), train_acc, val_acc, test_acc))
    print("Best at {} epoch, Val Accuracy {:.4f} Test Accuracy {:.4f}".format(best_epoch, best_val_acc, best_test_acc))
    return best_test_acc

if __name__ == "__main__":
    device = 'cuda:{}'.format(args.cuda_id) if args.cuda_id != 'cpu' else 'cpu'
    dataset = args.dataset
    set_seed(args.seed)

    data = load_data(dataset, device, split_id=0)
    graph = data.adj
    in_dim, hid_dim, out_dim = data.x.shape[1], args.hidden, data.num_of_class
    model = HopGNN(graph, in_dim, hid_dim, out_dim, args.num_hop, args.dropout,
                   feature_inter=args.interaction, activation=args.activation,
                   inter_layer=args.num_layer, feature_fusion=args.fusion, norm_type=args.norm_type).to(device)

    data.x = model.preprocess(graph, data.x)
    print('lr:', args.lr, 'wd:',args.weight_decay, 'hop:', args.num_hop, 'dropout:', args.dropout,
          args.interaction, 'inter_layers:', args.num_layer, args.fusion, args.norm_type)
    train(epoch=args.epochs, model=model, features=data.x, labels=data.y,
          train_mask=data.train_mask, val_mask=data.val_mask, test_mask=data.test_mask, args=args)