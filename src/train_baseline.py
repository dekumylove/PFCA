import argparse
import os
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Dataset_baseline import DiseasePredDataset
from model.DiseasePredModel_baseline import DiseasePredModel
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

def evaluate(eval_model, dataloader, device, only_dipole, p):
    """
    Evaluate the model on validation/test data
    """
    eval_model.eval()
    y_label = []
    y_pred = []
    total_loss = 0
    total_time = 0
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            p_, p2c, feature_index, rel_index, neighbor_index, feat_index, y = batch
            p_, p2c, feature_index, rel_index, neighbor_index, feat_index, y = (
                p_.cuda(args.device_id),
                p2c.cuda(args.device_id),
                feature_index.cuda(args.device_id),
                rel_index.cuda(args.device_id),
                neighbor_index.cuda(args.device_id),
                feat_index.cuda(args.device_id),
                y.cuda(args.device_id)
            )
            output, batch_time = eval_model(
                p_, p2c, feature_index, rel_index, neighbor_index, feat_index, only_dipole, p
            )
            loss = regularization_loss(output, y)
            y_label.extend(np.array(y.data.cpu()))
            y_pred.extend(np.array(output.data.cpu()))
            total_loss += loss.item()
            total_time += batch_time
    avg_loss = total_loss / len(dataloader)
    print(f"\nTest average Loss: {avg_loss}, Test Time: {total_time}")
    macro_auc, micro_auc, precision_mean, recall_mean, f1_mean, pr_auc = get_accuracy(
        y_label, y_pred
    )

    return macro_auc, micro_auc, precision_mean, recall_mean, f1_mean, pr_auc

def regularization_loss(output, target):
    """
    Calculate prediction loss
    """
    ce_loss = nn.BCELoss()
    loss = ce_loss(output, target)

    return loss

def collate_fn(data):
    print("len(data): ", len(data))


def main(args, p, p2c, features, rel_index, neighbor_index, feat_index, labels):
    """
    Main function to train and evaluate the model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    split_train_point = int(len(features) * 6.7 / 10)
    split_test_point = int(len(features) * 8.7 / 10)

    train_p, train_p2c, train_features, train_rel_index, train_neighbor_index, train_feat_index, train_labels = (
        p[:split_train_point],
        p2c[:split_train_point],
        features[:split_train_point],
        rel_index[:split_train_point],
        neighbor_index[:split_train_point],
        feat_index[:split_train_point],
        labels[:split_train_point]
    )
    test_p, test_p2c, test_features, test_rel_index, test_neighbor_index, test_feat_index, test_labels = (
        p[split_train_point:split_test_point],
        p2c[split_train_point:split_test_point],
        features[split_train_point:split_test_point],
        rel_index[split_train_point:split_test_point],
        neighbor_index[split_train_point:split_test_point],
        feat_index[split_train_point:split_test_point],
        labels[split_train_point:split_test_point]
    )
    valid_p, valid_p2c, valid_features, valid_rel_index, valid_neighbor_index, valid_feat_index, valid_labels = (
        p[split_test_point:],
        p2c[split_test_point:],
        features[split_test_point:],
        rel_index[split_test_point:],
        neighbor_index[split_test_point:],
        feat_index[split_test_point:],
        labels[split_test_point:]
    )

    print("train_p: ", len(train_p), "train_p2c: ", len(train_p2c), "train_features: ", len(train_features), "train_rel_index: ", len(train_rel_index), "train_neighbor_index: ", len(train_neighbor_index), "train_feat_index: ", len(train_feat_index), "train_labels: ", len(train_labels))
    print("test_p: ", len(test_p),"test_p2c: ", len(test_p2c),"test_features: ", len(test_features), "test_rel_index: ", len(test_rel_index), "test_neighbor_index: ", len(test_neighbor_index), "test_feat_index: ", len(test_feat_index), "test_labels: ", len(test_labels))
    print("valid_p: ", len(valid_p),"valid_p2c: ", len(valid_p2c),"valid_features: ", len(valid_features), "valid_rel_index: ", len(valid_rel_index), "valid_neighbor_index: ", len(valid_neighbor_index), "valid_feat_index: ", len(valid_feat_index), "valid_labels: ", len(valid_labels))

    train_data = DiseasePredDataset(train_p, train_p2c, train_features, train_rel_index, train_neighbor_index, train_feat_index, train_labels)
    test_data = DiseasePredDataset(test_p, test_p2c, test_features, test_rel_index, test_neighbor_index, test_feat_index, test_labels)
    valid_data = DiseasePredDataset(valid_p, valid_p2c, valid_features, valid_rel_index, valid_neighbor_index, valid_feat_index, valid_labels)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True)

    model = DiseasePredModel(
        model_type=args.model,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        hidden_dim=args.hidden_dim,
        embed_dim=128,
        num_path=args.num_path,
        threshold=args.threshold,
        dropout=args.dropout_ratio,
        alpha_CAPF=args.alpha_CAPF,
        gamma_GraphCare=args.gamma_GraphCare,
        lambda_HAR=args.lambda_HAR,
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
        total_time = 0
        model.train()
        for idx, batch in enumerate(train_loader):
            p, p2c, feature_index, rel_index, neighbor_index, feat_index, y = batch
            p, p2c, feature_index, rel_index, neighbor_index, feat_index, y = (
                p.cuda(args.device_id),
                p2c.cuda(args.device_id),
                feature_index.cuda(args.device_id),
                rel_index.cuda(args.device_id),
                neighbor_index.cuda(args.device_id),
                feat_index.cuda(args.device_id),
                y.cuda(args.device_id)
            )
            optimzer.zero_grad()
            output, batch_time = model(
                p, p2c, feature_index, rel_index, neighbor_index, feat_index, args.only_dipole, args.p
            )
            loss = regularization_loss(output, y)

            loss.backward()

            optimzer.step()
            llprint("\rtraining step: {} / {}".format(idx, len(train_loader)))
            total_loss += loss.item()
            total_time += batch_time

        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {i}, Average Loss: {avg_loss}, Training Time: {total_time}")

        # eval:
        macro_auc, micro_auc, precision_mean, recall_mean, f1_mean, roc_auc = evaluate(
            model,
            valid_loader,
            device,
            args.only_dipole,
            args.p
        )
        print(
            f"\nValid Result:\n"
            f"\nmacro_auc:{macro_auc}, micro_auc:{micro_auc}, roc_auc:{roc_auc}"
            f"\nprecision_mean:{precision_mean}\nrecall_mean:{recall_mean}\nf1_mean:{f1_mean}"
        )
        if roc_auc > best_eval_roc_auc:
            best_eval_roc_auc = roc_auc
            best_eval_epoch = i

        # test:
        macro_auc, micro_auc, precision_mean, recall_mean, f1_mean, roc_auc = evaluate(
            model,
            test_loader,
            device,
            args.only_dipole,
            args.p
        )
        print(
            f"\nTest Result:\n"
            f"\nmacro_auc:{macro_auc}, micro_auc:{micro_auc}, roc_auc:{roc_auc}"
            f"\nprecision_mean:{precision_mean}\nrecall_mean:{recall_mean}\nf1_mean:{f1_mean}"
        )

        if roc_auc > best_test_roc_auc:
            best_test_roc_auc = roc_auc
            best_test_epoch = i

        if i > 10:
            print(
                f"Nowbest Eval Epoch:{best_eval_epoch}, Nowbest_roc_auc:{best_eval_roc_auc}"
            )
            print(
                f"Nowbest Test Epoch:{best_test_epoch}, Nowbest_roc_auc:{best_test_roc_auc}"
            )
    print(f"Best Eval Epoch:{best_eval_epoch}, best_roc_auc:{best_eval_roc_auc}")
    print(f"Best Test Epoch:{best_test_epoch}, best_roc_auc:{best_test_roc_auc}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="MedPath",
        choices=["Dip_l", "Dip_g", "Dip_c", "Retain", "LSTM", "HAP", "GraphCare", "MedPath", "StageAware", "PFCA"],
        help="model",
    )
    parser.add_argument("--device_id", type=int, default=1, help="the id of GPU device")
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
    parser.add_argument("--output_dim", type=int, default=1, help="output_dim")
    parser.add_argument("--bi_direction", action="store_true", default=True, help="bi_direction")
    parser.add_argument("--batch_size", type=int, default=1, help="batch_size")
    parser.add_argument("--decay", type=float, default=0.0001, help="weight_decay")
    parser.add_argument("--beta", type=float, default=0.0001, help="KG factor in loss")
    parser.add_argument("--p", type=float, default=0.9, help="Proportion of the Left part")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--only_dipole", action="store_true", default=False, help="use only diploe moudle")
    parser.add_argument("--Lambda", type=float, default=0.9, help="lambda")
    parser.add_argument("--show_interpretation", action="store_true", default=False, help="show significant paths for interpretation")

    args = parser.parse_args()

    print("loading dataset...")
    with open('/data/wanghao/KDD2025/kdd_data/mimic-iii/graphcare_data/readmission_feat_index_rocauc.pkl', 'rb') as f:
        feat_index = pickle.load(f)
    with open('/data/wanghao/KDD2025/kdd_data/mimic-iii/graphcare_data/readmission_neighbor_index_rocauc.pkl', 'rb') as f:
        neighbor_index = pickle.load(f)
    
    base_path = os.path.join('../../kdd_data/', args.data_type)
    p = torch.load(os.path.join(base_path, 'hap_data/rp_p_rocauc.pt'))
    p2c = torch.load(os.path.join(base_path, 'hap_data/rp_p2c_rocauc.pt'))
    features = torch.load(os.path.join(base_path, 'readmission_features_one_hot_rocauc.pt'))
    rel_index = torch.load(os.path.join(base_path, f'readmission_rel_index_{args.K}_rocauc.pt'))
    labels = torch.load(os.path.join(base_path, f'readmission_label_one_hot_rocauc.pt'))

    main(args, p, p2c, features, rel_index, neighbor_index, feat_index, labels)