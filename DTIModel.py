import torch.nn as nn
import numpy as np
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F

import torch

from torch_geometric.utils import add_self_loops, degree, remove_self_loops
from torch_geometric.nn import MessagePassing
from transformers import BertModel
import math

from utilts import make_cuda

embed_dim = 128
device = torch.device('cuda')


class MHAtt(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super(MHAtt, self).__init__()

        self.linear_v = nn.Linear(hid_dim, hid_dim)
        self.linear_k = nn.Linear(hid_dim, hid_dim)
        self.linear_q = nn.Linear(hid_dim, hid_dim)
        self.linear_merge = nn.Linear(hid_dim, hid_dim)

        self.hid_dim = hid_dim
        self.dropout = dropout
        self.nhead = n_heads


        self.dropout = nn.Dropout(dropout)
        self.hidden_size_head = int(self.hid_dim / self.nhead)


    def forward(self, v, k, q, mask):

        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.nhead,
            self.hidden_size_head
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.nhead,
            self.hidden_size_head
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.nhead,
            self.hidden_size_head
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)


        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.hid_dim
        )
        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)


        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)


        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)


        return torch.matmul(att_map, value)


class DPA(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super(DPA, self).__init__()

        self.mhatt1 = MHAtt(hid_dim, n_heads, dropout)



        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hid_dim)



        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_size = hid_dim // n_heads
        assert (self.head_size * n_heads == self.hid_dim)

    def forward(self, x, y, y_mask=None):
        # x as V while y as Q and K
        # x = self.norm1(x + self.dropout1(
        #     self.mhatt1(x, x, y, y_mask)
        # ))
        x = self.norm1(x + self.dropout1(self.mhatt1(y, y, x, y_mask)))
        # x = self.norm1(x + self.dropout1(
        #     self.mhatt1(x, y, y_mask, y_mask)
        # ))

        return x


class SA(nn.Module):

    def __init__(self, hid_dim, n_heads, dropout):
        super(SA, self).__init__()

        self.mhatt1 = MHAtt(hid_dim, n_heads, dropout)
        # self.mhatt1 = MultiAttn(hid_dim, n_heads)

        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hid_dim)


        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_size = hid_dim // n_heads
        assert (self.head_size * n_heads == self.hid_dim)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(x, x, x, mask)
        ))
        # x = self.norm1(x + self.dropout1(
        #     self.mhatt1(x, x, mask, mask)
        # ))

        return x


class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation.

        self.lin = torch.nn.Linear(in_channels, out_channels)


    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.

        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3-5: Start propagating messages.

        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)


    def message(self, x_j, edge_index, size):
        # x_j has shape [E, out_channels]

        # Step 3: Normalize node features.
        # row, col = edge_index: 将边的索引矩阵拆分为两个数组，分别表示起始节点和目标节点的索引
        row, col = edge_index

        deg = degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return norm.view(-1, 1) * x_j


    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]

        # Step 5: Return new node embeddings.

        return aggr_out


class SAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels):

        super(SAGEConv, self).__init__(aggr='max')  # "Max" aggregation.

        self.lin = torch.nn.Linear(in_channels, out_channels)

        self.act = torch.nn.ReLU()

        self.update_lin = torch.nn.Linear(in_channels + out_channels, in_channels, bias=False)
        self.update_act = torch.nn.ReLU()


    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        a = edge_index

        edge_index, _ = remove_self_loops(edge_index)

        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))


        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j):



        x_j = self.lin(x_j)
        x_j = self.act(x_j)

        return x_j


    def update(self, aggr_out, x):
        # aggr_out has shape [N, out_channels]

        new_embedding = torch.cat([aggr_out, x], dim=1)

        new_embedding = self.update_lin(new_embedding)
        new_embedding = self.update_act(new_embedding)


        return new_embedding


class gnn(nn.Module):
    def __init__(self, n_fingerprint, pooling):
        super(gnn, self).__init__()
        self.pooling = pooling

        self.embed_fingerprint = nn.Embedding(num_embeddings=n_fingerprint, embedding_dim=embed_dim)
        self.conv1 = SAGEConv(embed_dim, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = SAGEConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = SAGEConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.8)
        self.linp1 = torch.nn.Linear(256, 128)
        self.linp2 = torch.nn.Linear(128, 512)

        # self.lin1 = torch.nn.Linear(128, 256)
        # self.lin2 = torch.nn.Linear(256, 512)
        # self.lin3 = torch.nn.Linear(64, 90)
        self.lin = torch.nn.Linear(128, 512)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()

    def forward(self, data):
        # x, edge_index, batch = data.x, data.edge_index, data.batch
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.embed_fingerprint(x)


        x = x.squeeze(1)

        x = F.relu(self.conv1(x, edge_index))


        if self.pooling:
            x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)

            x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        # print(x.shape)
        if self.pooling:
            x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
            x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # x = F.relu(self.conv3(x, edge_index))

        if self.pooling:
            x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
            x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

            x = x1 + x2 + x3
            x = self.linp1(x)
            x = self.act1(x)
            x = self.linp2(x)
        if not self.pooling:
            # x = self.lin1(x)
            # x = self.act1(x)
            # x = self.lin2(x)
            x = self.lin(x)
        # x = self.lin(x)
        # x = self.act2(x)
        # x = self.lin3(x)
        # x = F.dropout(x, p=0.5, training=self.training)

        return x


class inter_cross_att(nn.Module):

    def __init__(self, dim, nhead, dropout):
        super(inter_cross_att, self).__init__()
        self.sca = SA(dim, nhead, dropout)
        self.spa = SA(dim, nhead, dropout)
        self.coa_pc = DPA(dim, nhead, dropout)
        self.coa_cp = DPA(dim, nhead, dropout)

    def forward(self, protein_vector, compound_vector):
        compound_vector = self.sca(compound_vector, None)  # self-attention
        protein_vector = self.spa(protein_vector, None)  # self-attention

        compound_covector = self.coa_pc(compound_vector, protein_vector, None)  # co-attention
        protein_covector = self.coa_cp(protein_vector, compound_vector, None)  # co-attention


        return protein_covector, compound_covector

class encoder_cross_att_old(nn.Module):
    def __init__(self, dim, nhead, dropout):
        super(encoder_cross_att_old, self).__init__()
        self.sca = SA(dim, nhead, dropout)
        self.spa = SA(dim, nhead, dropout)
        self.coa_cp = DPA(dim, nhead, dropout)

    def forward(self, protein_vector, compound_vector):
        compound_vector = self.sca(compound_vector, None)  # self-attention
        protein_vector = self.spa(protein_vector, None)  # self-attention
        protein_covector = self.coa_cp(protein_vector, compound_vector, None)  # co-attention


        return protein_covector, compound_vector


class stack_cross_att(nn.Module):
    def __init__(self, dim, nhead, dropout):
        super(stack_cross_att, self).__init__()
        self.sca = SA(dim, nhead, dropout)
        self.spa = SA(dim, nhead, dropout)
        self.coa_cp = DPA(dim, nhead, dropout)

    def forward(self, protein_vector, compound_vector):
        compound_vector = self.sca(compound_vector, None)  # self-attention
        protein_vector = self.spa(protein_vector, None)  # self-attention
        protein_covector = self.coa_cp(protein_vector, compound_vector, None)  # co-attention

        return protein_covector, compound_vector


class encoder_cross_att(nn.Module):
    def __init__(self, dim, nhead, dropout, layers):
        super(encoder_cross_att, self).__init__()
        # self.encoder_layers = nn.ModuleList([SEA(dim, dropout) for _ in range(layers)])
        self.encoder_layers = nn.ModuleList([SA(dim, nhead, dropout) for _ in range(layers)])
        self.decoder_sa = nn.ModuleList([SA(dim, nhead, dropout) for _ in range(layers)])
        self.decoder_coa = nn.ModuleList([DPA(dim, nhead, dropout) for _ in range(layers)])
        self.layer_coa = layers

    def forward(self, protein_vector, compound_vector):
        for i in range(self.layer_coa):
            compound_vector = self.encoder_layers[i](compound_vector, None)  # self-attention
        for i in range(self.layer_coa):
            protein_vector = self.decoder_sa[i](protein_vector, None)
            protein_vector = self.decoder_coa[i](protein_vector, compound_vector, None)  # co-attention

        return protein_vector, compound_vector


class ProteinCNN(nn.Module):

    def __init__(self, embedding_dim, num_filters, kernel_size, padding=True):
        super(ProteinCNN, self).__init__()

        in_ch = [embedding_dim] + num_filters

        self.in_ch = in_ch[-1]
        kernels = kernel_size




        self.conv1 = nn.Conv1d(in_channels=in_ch[0], out_channels=in_ch[1], kernel_size=kernels[0])

        self.bn1 = nn.BatchNorm1d(in_ch[1])
        self.conv2 = nn.Conv1d(in_channels=in_ch[1], out_channels=in_ch[2], kernel_size=kernels[1])
        self.bn2 = nn.BatchNorm1d(in_ch[2])
        self.conv3 = nn.Conv1d(in_channels=in_ch[2], out_channels=in_ch[3], kernel_size=kernels[2])
        self.bn3 = nn.BatchNorm1d(in_ch[3])

    def forward(self, v):


        v = v.transpose(2, 1)
        v = self.bn1(F.relu(self.conv1(v)))
        v = self.bn2(F.relu(self.conv2(v)))
        v = self.bn3(F.relu(self.conv3(v)))

        v = v.view(v.size(0), v.size(2), -1)
        return v


class Dtis(nn.Module):
    def __init__(self, n_fingerprint, dim, n_word, layer_output,
                 layer_coa, nhead=8, dropout=0.1, co_attention=None,
                 gcn_pooling=False, bert_l=None):
        super(Dtis, self).__init__()
        self.co_attention = co_attention
        self.layer_output = layer_output
        self.layer_coa = layer_coa

        self.embed_word = nn.Embedding(n_word, dim)  # c.elegans n_word=8001 dim=512
        self.gnn = gnn(n_fingerprint, gcn_pooling)
        self.bert_l = bert_l

        self.W_attention = nn.Linear(dim, dim)
        self.sca_1 = SA(dim, nhead, dropout)
        self.sca_2 = SA(dim, nhead, dropout)
        self.sca_3 = SA(dim, nhead, dropout)

        # self-protein-attention layers
        self.spa_1 = SA(dim, nhead, dropout)
        self.spa_2 = SA(dim, nhead, dropout)
        self.spa_3 = SA(dim, nhead, dropout)



        self.coa_pc_1 = DPA(dim, nhead, dropout)
        self.coa_pc_2 = DPA(dim, nhead, dropout)
        self.coa_cp_1 = DPA(dim, nhead, dropout)
        self.coa_cp_2 = DPA(dim, nhead, dropout)
        self.coa_pc_3 = DPA(dim, nhead, dropout)
        self.coa_cp_3 = DPA(dim, nhead, dropout)

        self.W_out = nn.ModuleList([nn.Linear(2 * dim, dim), nn.Linear(dim, 128), nn.Linear(128, 64)
                                    ])

        self.W_out = nn.ModuleList([nn.Linear(2 * dim, dim), nn.Linear(dim, 128), nn.Linear(128, 64)
                                    ])

        self.encoder_coa_layers = encoder_cross_att(128, nhead, dropout, layer_coa)
        self.inter_coa_layers = nn.ModuleList([inter_cross_att(128, nhead, dropout) for _ in range(layer_coa)])
        self.stack_coa_layers = nn.ModuleList([stack_cross_att(128, nhead, dropout) for _ in range(layer_coa)])

        self.lin = nn.Linear(768, 512)

        self.W_interaction = nn.Linear(64, 2)
        self.lin768 = nn.Linear(512, 768)
        self.lin128 = nn.Linear(512, 128)
        self.protein_extractor = ProteinCNN(512, [128, 128, 128], [3, 6, 9], True)
        self.linbert = nn.Linear(128, 768)
        self.lin_ban = nn.Linear(100, 64)
        self.lin64 = nn.Linear(256, 64)
        self.lin2 = nn.Linear(512, 64)

    def forward(self, inputs, proteins):

        """Compound vector with GNN."""
        compound_vector = self.gnn(inputs)

        compound_vector = torch.unsqueeze(compound_vector, 0)  # sequence-like GNN ouput
        compound_vector_bert = self.lin768(compound_vector)
        compound_vector = self.lin128(compound_vector)

        """Protein vector with CNN."""
        protein_vector = self.embed_word(proteins)
        protein_vector = self.protein_extractor(protein_vector)

        protein_vector_bert = self.linbert(protein_vector)

        """BERT for Protein vector and Compound vector."""
        if (compound_vector_bert.size()[1] + protein_vector_bert.size()[1]) <= 512:
            cat_cp_vector = torch.cat((compound_vector_bert, protein_vector_bert), 1)

            token_type_ids = [0 for i in range(compound_vector_bert.size()[1])] + [1 for i in
                                                                                   range(protein_vector_bert.size()[1])]
        else:
            if compound_vector_bert.size()[1] >= 256 and protein_vector_bert.size()[1] >= 256:
                a_c = compound_vector_bert[:, :256, :]
                a_p = protein_vector_bert[:, :256, :]
            elif compound_vector_bert.size()[1] > 256 and protein_vector_bert.size()[1] < 256:
                a_p = protein_vector_bert
                a_c = compound_vector_bert[:, :512 - protein_vector_bert.size()[1], :]
            elif compound_vector_bert.size()[1] < 256 and protein_vector_bert.size()[1] > 256:
                a_c = compound_vector_bert
                a_p = protein_vector_bert[:, :512 - compound_vector_bert.size()[1], :]
            cat_cp_vector = torch.cat((a_c, a_p), 1)
            token_type_ids = [0 for i in range(a_c.size()[1])] + [1 for i in range(a_p.size()[1])]


        segments_tensors = torch.IntTensor(token_type_ids)
        segments_tensors = torch.unsqueeze(segments_tensors, 0)


        segments_tensors = make_cuda(segments_tensors)
        cat_cp_vector = make_cuda(cat_cp_vector)

        outputs = self.bert(input_ids=None, token_type_ids=segments_tensors, attention_mask=None,
                            inputs_embeds=cat_cp_vector)


        CLS_bert = outputs[0][:, 0, :]
        CLS_bert = self.lin(CLS_bert)
        CLS_bert = self.lin2(CLS_bert)

        """Intersection for Protein vector and Compound vector."""
        if 'encoder' in self.co_attention:
            protein_vector, compound_vector = self.encoder_coa_layers(protein_vector, compound_vector)
        elif 'stack' in self.co_attention:
            for i in range(self.layer_coa):
                protein_vector, compound_vector = self.stack_coa_layers[i](protein_vector, compound_vector)
        else:
            for i in range(self.layer_coa):
                protein_vector, compound_vector = self.inter_coa_layers[i](protein_vector, compound_vector)
        protein_vector = protein_vector.mean(dim=1)
        compound_vector = compound_vector.mean(dim=1)
        coa_vector = torch.cat((compound_vector, protein_vector), 1)

        coa_vector = torch.tanh(self.lin64(coa_vector) + self.bert_l * CLS_bert)

        interaction = self.W_interaction(coa_vector)

        return interaction

    def __call__(self, data, proteins, train=True):

        # inputs = data.x, data.edge_index, data.protein
        correct_interaction = data.y

        predicted_interaction = self.forward(data, proteins)

        if train:
            criterion = torch.nn.CrossEntropyLoss().to(device)
            loss = criterion(predicted_interaction, correct_interaction)
            return loss, predicted_interaction
        else:
            correct_labels = correct_interaction.to('cpu').data.numpy()
            ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            return correct_labels, predicted_labels, predicted_scores
