import torch
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set

from code.gnns.gnn_conv import GnnNode, GnnNodeVirtualNode


class HierarchicalGNNEncoder(torch.nn.Module):

    def __init__(
            self, num_tasks, num_layer=5, emb_dim=300, g_emb_dim=768,
            gnn_type='gin', virtual_node=True, residual=False,
            drop_ratio=0.5, JK="last", graph_pooling="mean"
    ):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(HierarchicalGNNEncoder, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.g_emb_dim = g_emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        # GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GnnNodeVirtualNode(
                num_layer, emb_dim, JK=JK, drop_ratio=drop_ratio,
                residual=residual, gnn_type=gnn_type
            )
        else:
            self.gnn_node = GnnNode(
                num_layer, emb_dim, JK=JK, drop_ratio=drop_ratio,
                residual=residual, gnn_type=gnn_type
            )

        # Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(
                gate_nn=torch.nn.Sequential(
                    torch.nn.Linear(emb_dim, 2 * emb_dim),
                    torch.nn.BatchNorm1d(2 * emb_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(2 * emb_dim, 1)
                )
            )
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)

        h_graph = self.pool(h_node, batched_data.batch)

        return h_graph


class HierarchicalGNN(torch.nn.Module):

    def __init__(
            self, num_tasks, num_layer=5, emb_dim=300, g_emb_dim=768,
            gnn_type='gin', virtual_node=True, residual=False,
            drop_ratio=0.5, JK="last", graph_pooling="mean"
    ):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(HierarchicalGNN, self).__init__()

        self.emb_dim = emb_dim
        self.g_emb_dim = g_emb_dim
        self.num_tasks = num_tasks

        self.encoder = HierarchicalGNNEncoder(
            num_tasks=num_tasks, num_layer=num_layer, emb_dim=emb_dim,
            g_emb_dim=g_emb_dim, gnn_type=gnn_type, virtual_node=virtual_node,
            residual=residual, drop_ratio=drop_ratio, JK=JK, graph_pooling=graph_pooling
        )

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2 * (self.emb_dim + self.g_emb_dim), self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim + self.g_emb_dim, self.num_tasks)

    def forward(self, batched_data):

        h_graph = self.encoder(batched_data)

        if self.g_emb_dim > 0:
            return self.graph_pred_linear(torch.concat((h_graph, batched_data.graph_x), dim=1))
        else:
            return self.graph_pred_linear(h_graph)


if __name__ == '__main__':
    HierarchicalGNN(num_tasks=10)