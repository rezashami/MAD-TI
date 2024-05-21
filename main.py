#!/usr/bin/env python

from utils.tools import SaveEvaluation, index_generator, parse_minibatch
from utils.Logger import Logger
from module.readData.read import getData
from module.model.MAGNN_lp import MAGNN_lp
import argparse
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")


num_ntype = 4
dropout_rate = 0.4
lr = 0.001
weight_decay = 0.001
etypes_lists = [
    [[0, 1], [2, 3], [4, 5], [None], [None]],
    [[1, 0], [6, 7], [None], [None]],
]
use_masks = {
    "drug_protein_drug": True,
    "drug_se_drug": False,
    "drug_disease_drug": False,
    "drug_drug": False,
    "drug_sim": False,
    "protein_disease_protein": False,
    "protein_drug_protein": True,
    "protein_protein": False,
    "protein_sim": False,
}
no_masks = {
    "drug_protein_drug": False,
    "drug_se_drug": False,
    "drug_disease_drug": False,
    "drug_drug": False,
    "drug_sim": False,
    "protein_disease_protein": False,
    "protein_drug_protein": False,
    "protein_protein": False,
    "protein_sim": False,
}


def get_pos_neg(Data):
    """
    This function used for spliting data into negative and positive lists.

    Input:
        Data: dataset list. Each element in Data list is something like `[drug_i, protein_j, edge_(i,j)]`
    """
    pos = []
    neg = []
    for ele in Data:
        if ele[2] == 1:
            pos.append([ele[0], ele[1]])
        elif ele[2] == 0:
            neg.append([ele[0], ele[1]])
    return np.array(pos, dtype=int), np.array(neg, dtype=int)


def run(
    mode,
    hidden_dim,
    out_size,
    num_heads,
    attn_vec_dim,
    rnn_type,
    num_epochs,
    patience,
    batch_size,
    repeat,
    feature_length,
    k_CV,
):
    """
    Main function. This function used for training and testing.

    Parameters:
    ------------
        * mode: Mode of ratio. Possible valuse: 0,1,2,3,4
        * hidden_dim: Hidden dim of NN model
        * out_size: Output dim of NN model
        * num_heads: Number of attention heads
        * attn_vec_dim: Attention featur length
        * rnn_type: Type of RNN (MetaPath encoder)
        * num_epochs: Number of epochs
        * patience: Patience threshold for early stopping
        * batch_size: Size of batch
        * repeat: Repeate count
        * dt: Drug similarity threshold
        * pt: Protein similarity threshold
        * feature_length: Length of initial features
        * k_CV: Cross validation data split ratio
    """
    ev = SaveEvaluation("result-auc-aupr/")
    logger = Logger()
    PATH = "./saved_models/"
    Path(PATH).mkdir(parents=True, exist_ok=True)
    metaPath_adj_dict, metaPath_dict, adjMat, type_mask, data_set, _ = getData(
        feature_length, mode
    )
    logger.log("info", f"Data loaded! with mode {mode}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    features_list = []
    in_dims = []
    for k in adjMat.ntypes:
        features_list.append(adjMat.ndata["h"][k].to(device))
        in_dims.append(feature_length)
    auc_list = []
    ap_list = []
    for r in range(repeat):
        logger.log("info", f"In the repeat number {r}")
        rs = np.random.randint(0, 1000, 1)[0]
        kf = StratifiedKFold(n_splits=k_CV, shuffle=True, random_state=rs)
        for train_index, test_index in kf.split(data_set[:, :2], data_set[:, 2]):

            # Get Train and test data from slicing the dataset list by test_index and train_index
            DTItrain, DTItest = data_set[train_index], data_set[test_index]
            # Get Validation data set from training data set
            DTItrain, DTIvalid = train_test_split(
                DTItrain, test_size=0.05, random_state=rs
            )
            # Split the train data into positive and negative
            train_pos, train_neg = get_pos_neg(DTItrain)
            # Split the validation data into positive and negative
            val_pos, val_neg = get_pos_neg(DTIvalid)
            # Split the test data into positive and negative
            test_pos, test_neg = get_pos_neg(DTItest)

            # Get positive trining batch controller object
            train_pos_idx_generator = index_generator(
                batch_size=batch_size, num_data=len(train_pos)
            )
            # Get negative trining batch controller object
            train_neg_idx_generator = index_generator(
                batch_size=batch_size, num_data=len(train_neg)
            )
            # Get positive vaildation batch controller object
            val_pos_idx_generator = index_generator(
                batch_size=batch_size, num_data=len(val_pos), shuffle=False
            )
            # Get negative validation batch controller object
            val_neg_idx_generator = index_generator(
                batch_size=batch_size, num_data=len(val_neg), shuffle=False
            )
            # Define the neuoral network
            net = MAGNN_lp(
                [3, 2],
                8,
                etypes_lists,
                in_dims,
                hidden_dim,
                out_size,
                num_heads,
                attn_vec_dim,
                rnn_type,
                dropout_rate,
            )
            net.to(device)
            # Define optimizer object
            optimizer = torch.optim.Adam(
                net.parameters(), lr=lr, weight_decay=weight_decay
            )

            for epoch in range(num_epochs):
                net.train()
                for iteration in range(train_pos_idx_generator.num_iterations()):
                    # Get Next batch of positive training data
                    train_pos_idx_batch = train_pos_idx_generator.next()
                    train_pos_idx_batch.sort()
                    train_pos_drug_protein_batch = train_pos[
                        train_pos_idx_batch
                    ].tolist()
                    # Generate metapath based graphs and indices
                    (
                        train_pos_g_lists,
                        train_pos_indices_lists,
                        train_pos_idx_batch_mapped_lists,
                    ) = parse_minibatch(
                        metaPath_adj_dict,
                        metaPath_dict,
                        train_pos_drug_protein_batch,
                        device,
                        use_masks,
                    )

                    for j in range(train_neg_idx_generator.num_iterations()):
                        # Get Next batch of negative training data
                        train_neg_idx_batch = train_neg_idx_generator.next()
                        train_neg_idx_batch.sort()
                        train_neg_batch = train_neg[train_neg_idx_batch].tolist()
                        # Generate metapath based graphs and indices
                        (
                            train_neg_g_lists,
                            train_neg_indices_lists,
                            train_neg_idx_batch_mapped_lists,
                        ) = parse_minibatch(
                            metaPath_adj_dict,
                            metaPath_dict,
                            train_neg_batch,
                            device,
                            no_masks,
                        )

                        [neg_embedding_user, neg_embedding_artist] = net(
                            (
                                train_neg_g_lists,
                                features_list,
                                type_mask,
                                train_neg_indices_lists,
                                train_neg_idx_batch_mapped_lists,
                            )
                        )

                        neg_embedding_user = neg_embedding_user.view(
                            -1, 1, neg_embedding_user.shape[1]
                        )
                        neg_embedding_artist = neg_embedding_artist.view(
                            -1, neg_embedding_artist.shape[1], 1
                        )

                        neg_out = -torch.bmm(neg_embedding_user, neg_embedding_artist)

                        [pos_embedding_user, pos_embedding_artist] = net(
                            (
                                train_pos_g_lists,
                                features_list,
                                type_mask,
                                train_pos_indices_lists,
                                train_pos_idx_batch_mapped_lists,
                            )
                        )
                        pos_embedding_user = pos_embedding_user.view(
                            -1, 1, pos_embedding_user.shape[1]
                        )
                        pos_embedding_artist = pos_embedding_artist.view(
                            -1, pos_embedding_artist.shape[1], 1
                        )
                        pos_out = torch.bmm(pos_embedding_user, pos_embedding_artist)
                        train_loss = -torch.mean(
                            F.logsigmoid(pos_out) + F.logsigmoid(neg_out)
                        )

                        optimizer.zero_grad()
                        train_loss.backward()
                        optimizer.step()

                        if j % 10 == 0:
                            logger.log(
                                "debug",
                                "#Epoch:{:05d}|Pos_Iteration:{:05d}|Neg_Iteration:{:05d}|Train_Loss:{:.4f}".format(
                                    epoch, iteration, j, train_loss.item()
                                ),
                            )
                            print(
                                "Epoch:{:05d}|Pos_Iteration:{:05d}|Neg_Iteration:{:05d}|Train_Loss:{:.4f}".format(
                                    epoch, iteration, j, train_loss.item()
                                )
                            )
                net.eval()
                val_loss = []
                with torch.no_grad():
                    for iteration in range(val_pos_idx_generator.num_iterations()):
                        val_idx_batch = val_pos_idx_generator.next()
                        val_pos_batch = val_pos[val_idx_batch].tolist()
                        (
                            val_pos_g_lists,
                            val_pos_indices_lists,
                            val_pos_idx_batch_mapped_lists,
                        ) = parse_minibatch(
                            metaPath_adj_dict,
                            metaPath_dict,
                            val_pos_batch,
                            device,
                            no_masks,
                        )

                        for j in range(val_neg_idx_generator.num_iterations()):
                            val_neg_batch_idx = val_neg_idx_generator.next()
                            val_neg_batch_idx.sort()
                            val_neg_batch = val_neg[val_neg_batch_idx].tolist()

                            (
                                val_neg_g_lists,
                                val_neg_indices_lists,
                                val_neg_idx_batch_mapped_lists,
                            ) = parse_minibatch(
                                metaPath_adj_dict,
                                metaPath_dict,
                                val_neg_batch,
                                device,
                                no_masks,
                            )

                            [neg_embedding_user, neg_embedding_artist] = net(
                                (
                                    val_neg_g_lists,
                                    features_list,
                                    type_mask,
                                    val_neg_indices_lists,
                                    val_neg_idx_batch_mapped_lists,
                                )
                            )

                            neg_embedding_user = neg_embedding_user.view(
                                -1, 1, neg_embedding_user.shape[1]
                            )
                            neg_embedding_artist = neg_embedding_artist.view(
                                -1, neg_embedding_artist.shape[1], 1
                            )
                            [pos_embedding_user, pos_embedding_artist] = net(
                                (
                                    val_pos_g_lists,
                                    features_list,
                                    type_mask,
                                    val_pos_indices_lists,
                                    val_pos_idx_batch_mapped_lists,
                                )
                            )
                            pos_embedding_user = pos_embedding_user.view(
                                -1, 1, pos_embedding_user.shape[1]
                            )
                            pos_embedding_artist = pos_embedding_artist.view(
                                -1, pos_embedding_artist.shape[1], 1
                            )
                            pos_out = torch.bmm(
                                pos_embedding_user, pos_embedding_artist
                            )
                            neg_out = -torch.bmm(
                                neg_embedding_user, neg_embedding_artist
                            )
                            val_loss.append(
                                -torch.mean(
                                    F.logsigmoid(pos_out) + F.logsigmoid(neg_out)
                                )
                            )
                val_loss = torch.mean(torch.tensor(val_loss))
                # print validation info
                print("Epoch:{:05d}|Val_Loss:{:.4f}".format(epoch, val_loss.item()))
                logger.log("debug", f"#Epoch:{epoch}|Val_Loss:{val_loss.item()}")

            pos_proba_list = []
            neg_proba_list = []
            test_pos_idx_generator = index_generator(
                batch_size=batch_size, num_data=len(test_pos), shuffle=False
            )
            test_neg_idx_generator = index_generator(
                batch_size=batch_size, num_data=len(test_neg), shuffle=False
            )
            n_pos = test_pos_idx_generator.num_iterations()
            n_neg = test_neg_idx_generator.num_iterations()
            y_true_test = np.array(
                [1] * (n_pos * batch_size) + [0] * (n_neg * batch_size)
            )

            with torch.no_grad():
                for iteration in range(test_pos_idx_generator.num_iterations()):

                    test_idx_batch = test_pos_idx_generator.next()
                    test_pos_batch = test_pos[test_idx_batch].tolist()

                    (
                        test_pos_g_lists,
                        test_pos_indices_lists,
                        test_pos_idx_batch_mapped_lists,
                    ) = parse_minibatch(
                        metaPath_adj_dict,
                        metaPath_dict,
                        test_pos_batch,
                        device,
                        no_masks,
                    )

                    [pos_embedding_user, pos_embedding_artist] = net(
                        (
                            test_pos_g_lists,
                            features_list,
                            type_mask,
                            test_pos_indices_lists,
                            test_pos_idx_batch_mapped_lists,
                        )
                    )
                    pos_embedding_user = pos_embedding_user.view(
                        -1, 1, pos_embedding_user.shape[1]
                    )
                    pos_embedding_artist = pos_embedding_artist.view(
                        -1, pos_embedding_artist.shape[1], 1
                    )
                    pos_out = torch.bmm(
                        pos_embedding_user, pos_embedding_artist
                    ).flatten()
                    pos_proba_list.append(pos_out)

                for j in range(test_neg_idx_generator.num_iterations()):
                    test_neg_batch_idx = test_neg_idx_generator.next()
                    test_neg_batch_idx.sort()
                    test_neg_batch = test_neg[test_neg_batch_idx].tolist()

                    (
                        test_neg_g_lists,
                        test_neg_indices_lists,
                        test_neg_idx_batch_mapped_lists,
                    ) = parse_minibatch(
                        metaPath_adj_dict,
                        metaPath_dict,
                        test_neg_batch,
                        device,
                        no_masks,
                    )

                    [neg_embedding_user, neg_embedding_artist] = net(
                        (
                            test_neg_g_lists,
                            features_list,
                            type_mask,
                            test_neg_indices_lists,
                            test_neg_idx_batch_mapped_lists,
                        )
                    )

                    neg_embedding_user = neg_embedding_user.view(
                        -1, 1, neg_embedding_user.shape[1]
                    )
                    neg_embedding_artist = neg_embedding_artist.view(
                        -1, neg_embedding_artist.shape[1], 1
                    )
                    neg_out = torch.bmm(
                        neg_embedding_user, neg_embedding_artist
                    ).flatten()
                    neg_proba_list.append(neg_out)

                y_proba_test = torch.cat(pos_proba_list + neg_proba_list)
                y_proba_test = y_proba_test.cpu().numpy()
                s = "["
                for item in y_proba_test:
                    s += str(item) + ","
                s = s[:-1] + "]"
                logger.log("info", s)
                auc = roc_auc_score(y_true_test, y_proba_test)
                ap = average_precision_score(y_true_test, y_proba_test)

                ev.save_roc_curve_info(y_true_test, y_proba_test, r)
                ev.save_aupr_curve_info(y_true_test, y_proba_test, r)
                print(f"Link Prediction Test for Iteration {r}")
                logger.log("info", f"Link Prediction Test for Iteration {r}")
                print("AUC = {}".format(auc))
                logger.log("info", "AUC = {}".format(auc))
                print("AUPR = {}".format(ap))
                logger.log("info", "AUPR = {}".format(ap))
                auc_list.append(auc)
                ap_list.append(ap)
            out_model_name = PATH + datetime.now().strftime("%Y-%m-%d %H-%M-%S") + ".pt"
            net.save_model(out_model_name)
            ev.update_counter()

        print("----------------------------------------------------------------")
        print("Link Prediction Tests Summary")
        print("AUC_mean = {}, AUC_std = {}".format(np.mean(auc_list), np.std(auc_list)))
        logger.log(
            "info",
            "AUC_mean = {}, AUC_std = {} for repeate {}".format(
                np.mean(auc_list), np.std(auc_list), r
            ),
        )
        print("AUPR_mean = {}, AUPR_std = {}".format(np.mean(ap_list), np.std(ap_list)))
        logger.log(
            "info",
            "AUPR_mean = {}, AUPR_std = {} for repeate {}".format(
                np.mean(ap_list), np.std(ap_list), r
            ),
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="MRGNN testing for the recommendation dataset"
    )
    ap.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Dimension of the node hidden state. Default is 64.",
    )
    ap.add_argument(
        "--num-heads",
        type=int,
        default=1,
        help="Number of the attention heads. Default is 8.",
    )
    ap.add_argument(
        "--attn-vec-dim",
        type=int,
        default=8,
        help="Dimension of the attention vector. Default is 128.",
    )
    ap.add_argument(
        "--rnn-type",
        default="max-pooling",
        help="Type of the aggregator. Default is RotatE0.",
    )
    ap.add_argument(
        "--epoch", type=int, default=3, help="Number of epochs. Default is 100."
    )
    ap.add_argument("--patience", type=int, default=10, help="Patience. Default is 5.")
    ap.add_argument(
        "--batch-size", type=int, default=64, help="Batch size. Default is 64."
    )
    ap.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Repeat the training and testing for N times. Default is 3.",
    )
    cmd = {}

    cmd["max-pooling"] = {
        "mode": [0, 1, 2, 3],
        "f_length": 512,
        "epoch": 0,
        "repeat": 2,
        "k_CV": 10,
    }
    cmd["RotatE0"] = {
        "mode": [0, 1, 2, 3],
        "f_length": 512,
        "epoch": 2,
        "repeat": 2,
        "k_CV": 10,
    }
    cmd["average"] = {
        "mode": [0, 1, 2, 3],
        "f_length": 512,
        "epoch": 2,
        "repeat": 2,
        "k_CV": 10,
    }
    cmd["linear"] = {
        "mode": [0, 1, 2, 3],
        "f_length": 512,
        "epoch": 2,
        "repeat": 2,
        "k_CV": 10,
    }
    cmd["neighbor"] = {
        "mode": [0, 1, 2, 3],
        "f_length": 512,
        "epoch": 2,
        "repeat": 2,
        "k_CV": 10,
    }
    cmd["neighbor-linear"] = {
        "mode": [0, 1, 2, 3],
        "f_length": 512,
        "epoch": 2,
        "repeat": 2,
        "k_CV": 10,
    }
    out_size = 128
    args = ap.parse_args()
    logger = Logger()

    for k, v in cmd.items():
        logger.log("info", "Runnig RNN type = " + k)
        for i in v["mode"]:
            logger.log("info", f"Going to run {i+1} mode with rnn {k}")
            run(
                i,
                args.hidden_dim,
                out_size,
                args.num_heads,
                args.attn_vec_dim,
                k,
                v["epoch"],
                args.patience,
                args.batch_size,
                v["repeat"],
                v["f_length"],
                v["k_CV"],
            )

    # feature_length = 256
    # dt = 1
    # pt = 1
    # out_size = 128
    # args = ap.parse_args()
    # logger = Logger()
    # logger.log('info', f'Going to run 1 mode')
    # for i in range(4):
    #     run(i, args.hidden_dim, out_size, args.num_heads, args.attn_vec_dim, args.rnn_type,
    #         args.epoch, args.patience, args.batch_size, args.repeat, dt, pt, feature_length)
