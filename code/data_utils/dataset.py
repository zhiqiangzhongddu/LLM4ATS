import torch
from ogb.graphproppred import PygGraphPropPredDataset
from pathlib import PurePath

from code.utils import project_root_path


class DatasetLoader():
    def __init__(
            self, name="ogbg-molbace"
    ):
        self.name = name

        self.dataset = self.load_data()

    def load_data(self):
        # Download and process data at root
        dataset = PygGraphPropPredDataset(
            name=self.name, root=PurePath(project_root_path, "data")
        )

        return dataset
