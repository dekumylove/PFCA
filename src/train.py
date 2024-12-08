import argparse
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Dataset import DiseasePredDataset
from model.DiseasePredModel import DiseasePredModel
from utils import llprint, get_accuracy

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.__version__)


torch.manual_seed(0)
np.random.seed(0)

saved_path = "../saved_model/"
model_name = "Our_model"
path = os.path.join(saved_path, model_name)
if not os.path.exists(path):
    os.makedirs(path)

def check_gradients(model):
    """Debug function to check gradient flow in the model"""
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f'{name}, grad_fn={param.grad_fn}, grad={param.grad.data.sum()}')

def evaluate(eval_model, dataloader, device, only_dipole, p, adj):
    """
    Evaluate the model on validation/test data
    """
    eval_model.eval()
    y_label = []
    y_pred = []
    total_loss = 0
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            # Move batch data to GPU
            feature_index, rel_index, feat_index, y = batch
            feature_index, rel_index, feat_index, y = (
                feature_index.cuda(args.device_id),
                rel_index.cuda(args.device_id),
                feat_index.cuda(args.device_id),
                y.cuda(args.device_id)
            )
            
            # Get model predictions
            output_f, output_c, output_t, path_attentions, causal_attentions = eval_model(
                feature_index, rel_index, feat_index, only_dipole, p
            )
            
            # Show interpretation if enabled
            if args.show_interpretation:
                sample_top_pathattn, sample_top_pathindex = eval_model.interpret(
                    path_attentions, causal_attentions
                )
                for index, sample in enumerate(sample_top_pathattn):
                    print(f"Sample {idx * args.batch_size + index}:")
                    top_pi = sample_top_pathindex[index]
                    for i in range(top_pi.size(-1)):
                        print(f"    Top path {i}: Attention Value = {sample_top_pathattn[index][i]}")
                        print(f"    Top path index is: {sample_top_pathindex[index][i]}")

            loss = regularization_loss(y, args.Lambda, adj, eval_model, args.beta, args.batch_size, output_f, output_c, output_t)
            y_label.extend(np.array(y.data.cpu()))
            y_pred.extend(np.array(output_c.data.cpu()))
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"\nTest average Loss: {avg_loss}")
    macro_auc, micro_auc, precision_mean, recall_mean, f1_mean, pr_auc = get_accuracy(
        y_label, y_pred
    )

    return macro_auc, micro_auc, precision_mean, recall_mean, f1_mean, pr_auc

def regularization_loss(target, Lambda, adj, model, beta, bs, output_f, output_c=None, output_t=None):
    """
    Calculate total loss including prediction/causal loss, intervention loss and trivial loss
    """
    # Extract weight matrices from the model
    hidden_size = model.hidden_dim
    W = model.dipole.rnn.weight_ih_l0
    W_ir = W[0:hidden_size, :]
    W_iz = W[hidden_size:2 * hidden_size, :]
    W_in = W[2 * hidden_size:3 * hidden_size, :]
    W_matrix = [W_ir, W_iz, W_in]

    relationship_loss = 0
    # A: (input_size , input_size) 01matrix
    for i in range(W_ir.shape[0]):
        for j in range(i, W_ir.shape[0]):
            if adj[i, j] != 0:
                for weight in W_matrix:
                    relationship_loss += torch.norm(weight[i] - weight[j], 2) ** 2

    # Calculate total loss based on model configuration
    if args.causal_analysis:
        kl_loss = nn.KLDivLoss()
        even_target = torch.ones(size=(target.size(0), target.size(1))) / 2
        even_target = even_target.cuda(args.device_id)
        loss3 = kl_loss(output_t, even_target)
        ce_loss = nn.BCELoss()
        loss1 = ce_loss(output_f, target)
        loss2 = ce_loss(output_c, target)
        total_loss = loss1 + Lambda * loss2 + Lambda * loss3 + (beta / bs) * relationship_loss
    else:
        ce_loss = nn.BCELoss()
        loss1 = ce_loss(output_f, target)
        total_loss = loss1 + (beta / bs) * relationship_loss
    return total_loss


def collate_fn(data):
    print("len(data): ", len(data))


def main(args, features, rel_index, feat_index, labels):
    """
    Main function to train and evaluate the model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    split_train_point = int(len(features) * 6.7 / 10)
    split_test_point = int(len(features) * 8.7 / 10)

    train_features, train_rel_index, train_feat_index, train_labels = (
        features[:split_train_point],
        rel_index[:split_train_point],
        feat_index[:split_train_point],
        labels[:split_train_point]
    )
    test_features, test_rel_index, test_feat_index, test_labels = (
        features[split_train_point:split_test_point],
        rel_index[split_train_point:split_test_point],
        feat_index[split_train_point:split_test_point],
        labels[split_train_point:split_test_point]
    )
    valid_features, valid_rel_index, valid_feat_index, valid_labels = (
        features[split_test_point:],
        rel_index[split_test_point:],
        feat_index[split_test_point:],
        labels[split_test_point:]
    )

    print("train_features: ", len(train_features), "train_rel_index: ", len(train_rel_index), "train_feat_index: ", len(train_feat_index), "train_labels: ", len(train_labels))
    print("test_features: ", len(test_features), "test_rel_index: ", len(test_rel_index), "test_feat_index: ", len(test_feat_index), "test_labels: ", len(test_labels))
    print("valid_features: ", len(valid_features), "valid_rel_index: ", len(valid_rel_index), "valid_feat_index: ", len(valid_feat_index), "valid_labels: ", len(valid_labels))

    train_data = DiseasePredDataset(train_features, train_rel_index, train_feat_index, train_labels)
    test_data = DiseasePredDataset(test_features, test_rel_index, test_feat_index, test_labels)
    valid_data = DiseasePredDataset(valid_features, valid_rel_index, valid_feat_index, valid_labels)

    with open('/data/wanghao/KDD2025/kdd_data/mimic-iii/adjacent_matrix.pkl', 'rb') as f:
        adj = pickle.load(f)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True)

    model = DiseasePredModel(
        path_filtering=args.path_filtering,
        joint_impact=args.joint_impact,
        causal_analysis=args.causal_analysis,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        hidden_dim=args.hidden_dim,
        embed_dim=256,
        num_path=args.num_path,
        threshold=args.threshold,
        dropout=args.dropout_ratio,
        alpha_CAPF=args.alpha_CAPF,
        bi_direction=args.bi_direction,
        device_id=args.device_id,
        device=device
    )

    epoch = 40
    optimzer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    model = model.cuda(args.device_id)

    best_eval_roc_auc = 0
    best_eval_epoch = 0

    best_test_roc_auc = 0
    best_test_epoch = 0
    for i in range(epoch):
        print("\nepoch {} --------------------------".format(i))
        total_loss = 0
        model.train()
        for idx, batch in enumerate(train_loader):
            feature_index, rel_index, feat_index, y = batch
            feature_index, rel_index, feat_index, y = (
                feature_index.cuda(args.device_id),
                rel_index.cuda(args.device_id),
                feat_index.cuda(args.device_id),
                y.cuda(args.device_id)
            )
            optimzer.zero_grad()
            if not args.only_dipole:
                if args.causal_analysis:
                    output_f, output_c, output_t, _, _ = model(feature_index, rel_index, feat_index, args.only_dipole, args.p)
                    loss = regularization_loss(y, args.Lambda, adj, model, args.beta, args.batch_size, output_f, output_c, output_t)
                else:
                    output_f = model(feature_index, rel_index, feat_index, args.only_dipole, args.p)
                    loss = regularization_loss(y, args.Lambda, adj, model, args.beta, args.batch_size, output_f)

            loss.backward()

            optimzer.step()
            llprint("\rtraining step: {} / {}".format(idx, len(train_loader)))
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {i}, Average Loss: {avg_loss}")



        # eval:
        if not args.only_dipole:
            macro_auc, micro_auc, precision_mean, recall_mean, f1_mean, roc_auc = evaluate(
                model,
                valid_loader,
                device,
                args.only_dipole,
                args.p,
                adj
            )
        print(
            f"\nValid Result:\n"
            f"\nmacro_auc:{macro_auc}, micro_auc:{micro_auc}, roc_auc:{roc_auc}"
            f"\nprecision_mean:{precision_mean}\nrecall_mean:{recall_mean}\nf1_mean:{f1_mean}"
        )
        if macro_auc > best_eval_roc_auc:
            best_eval_roc_auc = macro_auc
            best_eval_epoch = i

        # test:
        if not args.only_dipole:
            macro_auc, micro_auc, precision_mean, recall_mean, f1_mean, roc_auc = evaluate(
                model,
                test_loader,
                device,
                args.only_dipole,
                args.p,
                adj
            )
        print(
            f"\nTest Result:\n"
            f"\nmacro_auc:{macro_auc}, micro_auc:{micro_auc}, roc_auc:{roc_auc}"
            f"\nprecision_mean:{precision_mean}\nrecall_mean:{recall_mean}\nf1_mean:{f1_mean}"
        )

        if macro_auc > best_test_roc_auc:
            best_test_roc_auc = macro_auc
            best_test_epoch = i

        if i > 10:
            print(
                f"Nowbest Eval Epoch:{best_eval_epoch}, Nowbest_macro_auc:{best_eval_roc_auc}"
            )
            print(
                f"Nowbest Test Epoch:{best_test_epoch}, Nowbest_macro_auc:{best_test_roc_auc}"
            )
    print(f"Best Eval Epoch:{best_eval_epoch}, best_macro_auc:{best_eval_roc_auc}")
    print(f"Best Test Epoch:{best_test_epoch}, best_macro_auc:{best_test_roc_auc}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="PFCA",
        choices=["Dip_l", "Dip_g", "Dip_c", "Retain", "LSTM", "GraphCare", "MedPath", "StageAware", "PFCA"],
        help="model",
    )
    parser.add_argument("--device_id", type=int, default=0, help="the id of GPU device")
    parser.add_argument("--lambda_HAR", type=float, default=0.1, help="the lambda of HAR")
    parser.add_argument("--gamma_GraphCare", type=float, default=0.1, help="the gamma of GraphCare")
    parser.add_argument("--alpha_CAPF", type=float, default=0.2, help="the alpha of CAPF")
    parser.add_argument("--dropout_ratio", type=float, default=0.1, help="the dropout_ratio")
    parser.add_argument("--K", type=int, default=3, help="the path length")
    parser.add_argument("--data_type", type=str, default='mimic-iii', help="the type of data")
    parser.add_argument("--num_path", type=int, default=3, help="the maximum number of the paths of each pair")
    parser.add_argument("--threshold", type=int, default=0.005, help="the threshold of the path")
    parser.add_argument("--input_dim", type=int, default=2850, help="input_dim (feature_size)")
    parser.add_argument("--hidden_dim", type=int, default=256, help="hidden_dim")
    parser.add_argument("--output_dim", type=int, default=90, help="output_dim")
    parser.add_argument("--bi_direction", action="store_true", default=True, help="bi_direction")
    parser.add_argument("--batch_size", type=int, default=8, help="batch_size")
    parser.add_argument("--decay", type=float, default=0.0001, help="weight_decay")
    parser.add_argument("--beta", type=float, default=0.0001, help="KG factor in loss")
    parser.add_argument("--p", type=float, default=0.9, help="Proportion of the Left part")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--only_dipole", action="store_true", default=False, help="use only diploe moudle")
    parser.add_argument("--Lambda", type=float, default=0.5, help="lambda")
    parser.add_argument("--show_interpretation", action="store_true", default=False, help="show significant paths for interpretation")
    parser.add_argument("--joint_impact", action="store_true", default=True, help="use joint impact")
    parser.add_argument("--path_filtering", action="store_true", default=True, help="use path filtering")
    parser.add_argument("--causal_analysis", action="store_true", default=True, help="use causal analysis")

    args = parser.parse_args()

    print("loading dataset...")
    base_path = os.path.join('../../kdd_data/', args.data_type)
    features = torch.load(os.path.join(base_path, 'features_one_hot.pt'))
    rel_index = torch.load(os.path.join(base_path, f'rel_index_{args.K}.pt'))
    feat_index = torch.load(os.path.join(base_path, f'feat_index_{args.K}.pt'))
    labels = torch.load(os.path.join(base_path, f'new_label_one_hot.pt'))

    print(
        f"features_len:{len(features)}, rel_index_len:{len(rel_index)}, feat_index_len:{len(feat_index)}, labels_len:{len(labels)}"
    )

    main(args, features, rel_index, feat_index, labels)