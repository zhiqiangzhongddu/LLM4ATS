import pandas as pd
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
            name=self.name, root=str(PurePath(project_root_path, "data"))
        )

        df = pd.read_csv(
            filepath_or_buffer=PurePath(
                project_root_path, "data",
                "ogbg_{}".format(self.name.split("-")[1]),
                "mapping", "mol.csv.gz"
            ),
            compression='gzip'
        )
        text = df.smiles.tolist()

        return dataset, text
