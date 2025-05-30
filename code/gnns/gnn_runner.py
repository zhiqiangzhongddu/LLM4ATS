from tqdm import tqdm
import numpy as np
from pathlib import PurePath

import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from ogb.graphproppred import Evaluator

from code.data_utils.dataset import DatasetLoader
from code.gnns.hierarchical_gnn import HierarchicalGNN
from code.utils import time_logger, project_root_path, init_path


class GNNRunner():
    def __init__(self, cfg):
        self.dataset = cfg.dataset
        self.seed = cfg.seed
        self.device = cfg.device if torch.cuda.is_available() else torch.device('cpu')

        self.model_name = cfg.model.name
        self.hidden_dim = cfg.model.gnn.hidden_dim
        self.num_layers = cfg.model.gnn.num_layers
        self.batch_size = cfg.model.gnn.batch_size
        self.dropout = cfg.model.gnn.dropout
        self.lr = cfg.model.gnn.lr

        self._get_evaluator()
        self.cls_criterion = torch.nn.BCEWithLogitsLoss()
        self.reg_criterion = torch.nn.MSELoss()

        self.epochs = cfg.model.gnn.epochs

        self.output_dir = PurePath(
            project_root_path, "output", "gnns", self.dataset,
            "{}-seed{}".format(self.model_name, self.seed)
        )

        (self.dataset, self.text, self.train_loader, self.valid_loader,
         self.test_loader, self.data_loader) = self.preprocess_data()
        self.eval_metric = self.dataset.eval_metric
        self.num_tasks = self.dataset.num_tasks
        self.task_type = self.dataset.task_type
        self.num_classes = self.dataset.num_classes
        self.g_emb_dim = self.dataset.data.g_x.size(1) \
            if hasattr(self.dataset._data, 'graph_x') else 0

        self.model = self.setup_model()
        self.optimizer = self.setup_optimizers()
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Num. of parameters: {}".format(trainable_params))

    def preprocess_data(self):
        # Preprocess data
        dataloader, text = DatasetLoader(
            name=self.dataset
        )
        dataset = dataloader.dataset
        split_idx = dataset.get_idx_split()

        train_loader = DataLoader(
            dataset[split_idx["train"]], batch_size=self.batch_size, shuffle=True,
        )
        valid_loader = DataLoader(
            dataset[split_idx["valid"]], batch_size=self.batch_size, shuffle=False,
        )
        test_loader = DataLoader(
            dataset[split_idx["test"]], batch_size=self.batch_size, shuffle=False,
        )

        data_loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False,
        )

        return dataset, text, train_loader, valid_loader, test_loader, data_loader

    def setup_model(self):
        if self.model_name == 'gin':
            model = HierarchicalGNN(
                gnn_type='gin', num_tasks=self.num_tasks, num_layer=self.num_layers,
                emb_dim=self.hidden_dim, g_emb_dim=self.g_emb_dim,
                drop_ratio=self.dropout, virtual_node=False
            )
        elif self.model_name == 'gin-v':
            model = HierarchicalGNN(
                gnn_type='gin', num_tasks=self.num_tasks, num_layer=self.num_layers,
                emb_dim=self.hidden_dim, g_emb_dim=self.g_emb_dim,
                drop_ratio=self.dropout, virtual_node=True
            )
        elif self.model_name == 'gcn':
            model = HierarchicalGNN(
                gnn_type='gcn', num_tasks=self.num_tasks, num_layer=self.num_layers,
                emb_dim=self.hidden_dim, g_emb_dim=self.g_emb_dim,
                drop_ratio=self.dropout, virtual_node=False
            )
        elif self.model_name == 'gcn-v':
            model = HierarchicalGNN(
                gnn_type='gcn', num_tasks=self.num_tasks, num_layer=self.num_layers,
                emb_dim=self.hidden_dim, g_emb_dim=self.g_emb_dim,
                drop_ratio=self.dropout, virtual_node=True
            )
        else:
            raise ValueError('Invalid GNN type')

        return model

    def setup_optimizers(self):
        optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr)
        return optimizer


    def _get_evaluator(self):
        self.evaluator = Evaluator(name=self.dataset)

    @time_logger
    @torch.no_grad()
    def get_pred(self, loader):
        self.model.eval()

        y_pred = []

        for step, batch in enumerate(tqdm(loader, desc="Iteration")):
            batch = batch.to(self.device)

            if batch.x.shape[0] == 1:
                pass
            else:
                with torch.no_grad():
                    pred = self.model.to(self.device)(batch)

                y_pred.append(pred.detach().cpu())

        y_pred = torch.cat(y_pred, dim=0)

        return y_pred

    def _train(self, loader):
        self.model.train()

        for step, batch in enumerate(tqdm(loader, desc="Iteration")):
            batch = batch.to(self.device)

            if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
                pass
            else:
                pred = self.model.to(self.device)(batch)
                self.optimizer.zero_grad()
                is_labeled = batch.y == batch.y
                if "classification" in self.task_type:
                    loss = self.cls_criterion(
                        pred.to(torch.float32)[is_labeled],
                        batch.y.to(torch.float32)[is_labeled]
                    )
                else:
                    loss = self.reg_criterion(
                        pred.to(torch.float32)[is_labeled],
                        batch.y.to(torch.float32)[is_labeled]
                    )
                loss.backward()
                self.optimizer.step()

    @time_logger
    @torch.no_grad()
    def _eval(self, loader):
        self.model.eval()
        y_true = []
        y_pred = []

        for step, batch in enumerate(tqdm(loader, desc="Iteration")):
            batch = batch.to(self.device)

            if batch.x.shape[0] == 1:
                pass
            else:
                with torch.no_grad():
                    pred = self.model.to(self.device)(batch)

                y_true.append(batch.y.view(pred.shape).detach().cpu())
                y_pred.append(pred.detach().cpu())

        y_true = torch.cat(y_true, dim=0).numpy()
        y_pred = torch.cat(y_pred, dim=0).numpy()

        input_dict = {"y_true": y_true, "y_pred": y_pred}

        return self.evaluator.eval(input_dict)

    @time_logger
    def train_and_eval(self):

        train_curve = []
        valid_curve = []
        test_curve = []
        prediction_list = []

        for epoch in range(1, self.epochs + 1):
            print("=====Epoch {}=====".format(epoch))
            print('Training...')
            self._train(loader=self.train_loader)

            print('Evaluating...')
            train_perf = self._eval(loader=self.train_loader)
            valid_perf = self._eval(loader=self.valid_loader)
            test_perf = self._eval(loader=self.test_loader)

            print('Train: ', train_perf, 'Validation: ', valid_perf, 'Test: ', test_perf)
            train_curve.append(train_perf[self.eval_metric])
            valid_curve.append(valid_perf[self.eval_metric])
            test_curve.append(test_perf[self.eval_metric])

            print('Obtaining predictions...')
            prediction = self.get_pred(self.data_loader)
            prediction_list.append(prediction)

        if 'classification' in self.task_type:
            best_val_epoch = np.argmax(np.array(valid_curve))
        else:
            best_val_epoch = np.argmin(np.array(valid_curve))

        print('Best epoch: ', best_val_epoch)
        print('Best validation score: {:.4f}'.format(valid_curve[best_val_epoch]))
        print('Test score: {:.4f}'.format(test_curve[best_val_epoch]))
        self.save_predictions(prediction=prediction_list[best_val_epoch])

    @torch.no_grad()
    def save_predictions(self, prediction):
        file_path = "{}/predictions.pt".format(self.output_dir)

        init_path(dir_or_file=file_path)
        torch.save(prediction, file_path)
