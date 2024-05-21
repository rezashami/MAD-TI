import torch
import torch.nn as nn
import numpy as np

from module.model.base_MAGNN import MAGNN_ctr_ntype_specific


# for link prediction task
class MAGNN_lp_layer(nn.Module):
    def __init__(self,
                 num_metapaths_list,
                 num_edge_type,
                 etypes_lists,
                 in_dim,
                 out_dim,
                 num_heads,
                 attn_vec_dim,
                 rnn_type='gru',
                 attn_drop=0.5):
        super(MAGNN_lp_layer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads

        # etype-specific parameters
        r_vec = None
        if rnn_type == 'TransE0':
            r_vec = nn.Parameter(torch.empty(
                size=(num_edge_type // 2, in_dim)))
        elif rnn_type == 'TransE1':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type, in_dim)))
        elif rnn_type == 'RotatE0':
            r_vec = nn.Parameter(torch.empty(
                size=(num_edge_type // 2, in_dim // 2, 2)))
        elif rnn_type == 'RotatE1':
            r_vec = nn.Parameter(torch.empty(
                size=(num_edge_type, in_dim // 2, 2)))
        if r_vec is not None:
            nn.init.xavier_normal_(r_vec.data, gain=1.414)

        # ctr_ntype-specific layers
        self.drug_layer = MAGNN_ctr_ntype_specific(num_metapaths_list[0],
                                                   etypes_lists[0],
                                                   in_dim,
                                                   num_heads,
                                                   attn_vec_dim,
                                                   rnn_type,
                                                   r_vec,
                                                   attn_drop,
                                                   use_minibatch=True)
        self.protein_layer = MAGNN_ctr_ntype_specific(num_metapaths_list[1],
                                                   etypes_lists[1],
                                                   in_dim,
                                                   num_heads,
                                                   attn_vec_dim,
                                                   rnn_type,
                                                   r_vec,
                                                   attn_drop,
                                                   use_minibatch=True)
        
        fc_inp = in_dim * num_heads

        if rnn_type == 'imchgan':
            fc_inp*=2
        # note that the acutal input dimension should consider the number of heads
        # as multiple head outputs are concatenated together
        self.fc_drug = nn.Linear(fc_inp, out_dim, bias=True)
        self.fc_protein = nn.Linear(fc_inp, out_dim, bias=True)
        nn.init.xavier_normal_(self.fc_drug.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc_protein.weight, gain=1.414)

    def forward(self, inputs):
        g_dict, features, type_mask, mp_route_l_pos, mapping_pos = inputs

        # ctr_ntype-specific layers
        h_drug = self.drug_layer(
            (g_dict[0], features, type_mask, mp_route_l_pos[0], mapping_pos[0]))
        h_protein = self.protein_layer(
            (g_dict[1], features, type_mask, mp_route_l_pos[1], mapping_pos[1]))

        logits_drug = self.fc_drug(h_drug)
        logits_protein= self.fc_protein(h_protein)
        return [logits_drug, logits_protein], [h_drug, h_protein]


class MAGNN_lp(nn.Module):
    def __init__(self,
                 num_metapaths_list,
                 num_edge_type,
                 etypes_lists,
                 feats_dim_list,
                 hidden_dim,
                 out_dim,
                 num_heads,
                 attn_vec_dim,
                 rnn_type='gru',
                 dropout_rate=0.5):
        super(MAGNN_lp, self).__init__()
        self.hidden_dim = hidden_dim

        # ntype-specific transformation
        self.fc_list = nn.ModuleList(
            [nn.Linear(feats_dim, hidden_dim, bias=True) for feats_dim in feats_dim_list])
        # feature dropout after trainsformation
        if dropout_rate > 0:
            self.feat_drop = nn.Dropout(dropout_rate)
        else:
            self.feat_drop = lambda x: x
        # initialization of fc layers
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        # MAGNN_lp layers
        self.layer1 = MAGNN_lp_layer(num_metapaths_list,
                                     num_edge_type,
                                     etypes_lists,
                                     hidden_dim,
                                     out_dim,
                                     num_heads,
                                     attn_vec_dim,
                                     rnn_type,
                                     attn_drop=dropout_rate)

    def save_model(self, PATH):
        torch.save(self.state_dict(), PATH)

    def load_model(self, PATH):
        self.load_state_dict(torch.load(PATH))

    def forward(self, inputs):
        g_dict, features_list, type_mask, mp_route_l_pos, mapping_pos = inputs

        # ntype-specific transformation
        transformed_features = torch.zeros(
            type_mask.shape[0], self.hidden_dim, device=features_list[0].device)
        for i, fc in enumerate(self.fc_list):
            node_indices = np.where(type_mask == i)[0]
            transformed_features[node_indices] = fc(features_list[i])
        transformed_features = self.feat_drop(transformed_features)

        # hidden layers
        [logits_drug, logits_protein], _ = self.layer1(
            (g_dict, transformed_features, type_mask, mp_route_l_pos, mapping_pos))
        
        return [logits_drug, logits_protein] 
