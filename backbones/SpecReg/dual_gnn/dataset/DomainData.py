"""Citation-domain graph loading with explicit label isolation.

Source datasets cache features, edges, and labels in ``processed/data.pt``.
Their split masks are always regenerated in memory from ``split_seed``. Target
datasets used for training cache only features and edges in
``processed/unlabeled_data.pt`` and never open or deserialize labels.
"""

import os.path as osp

import numpy as np
import torch
import torch_geometric as tgeom
from sklearn.decomposition import TruncatedSVD
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.io import read_txt_array


class DomainData(InMemoryDataset):
    """Load one citation graph in labeled-source or unlabeled-target mode."""

    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1

    def __init__(
        self,
        root,
        name,
        split_seed=42,
        load_labels=True,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        load_spectral=True,
    ):
        if not name or osp.basename(osp.normpath(root)) != name:
            raise ValueError(
                "Dataset root/name mismatch: expected root to end with "
                f"'{name}', but received root='{root}'."
            )

        self.name = name
        self.split_seed = int(split_seed)
        self.load_labels = bool(load_labels)
        self.load_spectral = bool(load_spectral)
        super().__init__(root, transform, pre_transform, pre_filter)

        processed_data, self.slices = self._torch_load(self.processed_paths[0])
        self.data = processed_data
        self._data = processed_data
        self._processed_data = processed_data
        self._validate_processed_cache()

        if self.load_labels:
            self._apply_source_split()
            self._print_split_statistics()

        if self.load_spectral:
            self._load_or_create_spectral_cache()

    @staticmethod
    def _torch_load(path):
        """Load project caches across older and newer PyTorch versions."""
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    @property
    def raw_file_names(self):
        names = [
            f"{self.name}_docs.txt",
            f"{self.name}_edgelist.txt",
        ]
        if self.load_labels:
            names.append(f"{self.name}_labels.txt")
        return names

    @property
    def processed_file_names(self):
        # Never let the target path deserialize a legacy label-containing
        # data.pt: unlabeled graphs have a distinct cache file.
        return ["data.pt" if self.load_labels else "unlabeled_data.pt"]

    @property
    def spectral_paths(self):
        # Preserve the original SpecReg cache names and cache semantics.
        base_path = osp.join(self.processed_dir, "data.pt")
        return base_path + "eival.pt", base_path + "eivec.pt"

    def _raw_path(self, suffix):
        return osp.join(self.raw_dir, f"{self.name}_{suffix}.txt")

    def download(self):
        missing = [path for path in self.raw_paths if not osp.isfile(path)]
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            f"Missing raw files for dataset '{self.name}':\n{formatted}\n"
            "Expected filenames use the dataset name as their prefix."
        )

    def _raw_metadata(self):
        docs_path = self._raw_path("docs")
        num_nodes = 0
        num_features = None
        with open(docs_path, "r", encoding="utf-8") as docs_file:
            for line in docs_file:
                stripped = line.strip()
                if not stripped:
                    continue
                width = len(stripped.split(","))
                if num_features is None:
                    num_features = width
                elif width != num_features:
                    raise ValueError(
                        f"Inconsistent feature width in '{docs_path}': "
                        f"expected {num_features}, found {width}."
                    )
                num_nodes += 1

        if not num_nodes or num_features is None:
            raise ValueError(
                f"Raw feature file is empty for dataset '{self.name}': "
                f"{docs_path}"
            )

        metadata = {"num_nodes": num_nodes, "num_features": num_features}
        if self.load_labels:
            labels = self._read_label_file()
            metadata["num_labels"] = int(labels.numel())
            metadata["num_classes"] = int(torch.unique(labels).numel())
        return metadata

    def _cache_error(self, detail):
        raise RuntimeError(
            f"Processed cache mismatch for dataset '{self.name}': {detail}\n"
            f"Raw directory: {self.raw_dir}\n"
            f"Processed cache: {self.processed_paths[0]}\n"
            f"Remove '{self.processed_paths[0]}' and run again to rebuild "
            "that cache from the current raw files."
        )

    @classmethod
    def build_split_masks(cls, num_nodes, split_seed):
        """Build deterministic 80/10/10 masks without using global RNG."""
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(split_seed))
        permutation = torch.randperm(num_nodes, generator=generator)

        num_train = int(cls.TRAIN_RATIO * num_nodes)
        num_val = int(cls.VAL_RATIO * num_nodes)
        num_test = num_nodes - num_train - num_val

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_end = num_train
        val_end = train_end + num_val
        train_mask[permutation[:train_end]] = True
        val_mask[permutation[train_end:val_end]] = True
        test_mask[permutation[val_end:val_end + num_test]] = True

        if int(train_mask.sum() + val_mask.sum() + test_mask.sum()) != num_nodes:
            raise RuntimeError("Source split does not cover every node.")
        if (
            torch.any(train_mask & val_mask)
            or torch.any(train_mask & test_mask)
            or torch.any(val_mask & test_mask)
        ):
            raise RuntimeError("Source split masks overlap.")
        return train_mask, val_mask, test_mask

    def _apply_source_split(self):
        data = self._processed_data
        train_mask, val_mask, test_mask = self.build_split_masks(
            int(data.x.size(0)), self.split_seed
        )
        # Cached masks and metadata, if present, are deliberately overwritten.
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        data.split_seed = self.split_seed
        data.train_ratio = self.TRAIN_RATIO
        data.val_ratio = self.VAL_RATIO
        data.test_ratio = self.TEST_RATIO

    def _validate_processed_cache(self):
        expected = self._raw_metadata()
        data = self._processed_data

        if data.x is None or data.x.dim() != 2:
            self._cache_error("cached data.x is missing or not two-dimensional")
        actual_nodes = int(data.x.size(0))
        actual_features = int(data.x.size(1))
        if actual_nodes != expected["num_nodes"]:
            self._cache_error(
                f"raw features have {expected['num_nodes']} nodes, but the "
                f"cache has {actual_nodes} nodes"
            )
        if actual_features != expected["num_features"]:
            self._cache_error(
                f"raw features have {expected['num_features']} columns, but "
                f"the cache has {actual_features}"
            )
        if data.edge_index is None or data.edge_index.dim() != 2:
            self._cache_error("cached edge_index is missing or invalid")

        cached_labels = getattr(data, "y", None)
        if self.load_labels:
            if cached_labels is None:
                self._cache_error("cached data.y is missing")
            if cached_labels.numel() != expected["num_labels"]:
                self._cache_error(
                    f"raw file has {expected['num_labels']} labels, but the "
                    f"cache has {cached_labels.numel()}"
                )
            actual_classes = int(torch.unique(cached_labels).numel())
            if actual_classes != expected["num_classes"]:
                self._cache_error(
                    f"raw labels contain {expected['num_classes']} classes, "
                    f"but the cache contains {actual_classes}"
                )
        elif cached_labels is not None:
            self._cache_error("unlabeled target cache unexpectedly contains data.y")

    def _print_split_statistics(self):
        data = self._processed_data
        print(f"{self.name} source split:")
        print(f"nodes={data.num_nodes}")
        print(f"train={int(data.train_mask.sum())}")
        print(f"validation={int(data.val_mask.sum())}")
        print(f"test={int(data.test_mask.sum())}")
        print(f"split_seed={self.split_seed}")

    def _load_or_create_spectral_cache(self):
        eival_path, eivec_path = self.spectral_paths
        cache_exists = osp.isfile(eival_path) and osp.isfile(eivec_path)

        if not cache_exists:
            num_nodes = self._processed_data.x.size(0)
            edge_index, edge_weight = tgeom.utils.get_laplacian(
                self._processed_data.edge_index,
                num_nodes=num_nodes,
            )
            edge_index = edge_index.numpy()
            edge_weight = edge_weight.numpy()
            adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
            adj[edge_index[0, :], edge_index[1, :]] = edge_weight

            # These are the original SpecReg decomposition parameters.
            pca = TruncatedSVD(
                n_components=1000,
                n_iter=20,
                random_state=42,
            )
            pca.fit(adj)
            torch.save(
                torch.tensor(
                    pca.explained_variance_ ** 0.5,
                    dtype=torch.float32,
                ),
                eival_path,
            )
            torch.save(
                torch.tensor(pca.components_, dtype=torch.float32),
                eivec_path,
            )

        self.eival = self._torch_load(eival_path)
        self.eivec = self._torch_load(eivec_path)

        num_nodes = self._processed_data.x.size(0)
        if self.eivec.dim() != 2 or self.eivec.size(1) != num_nodes:
            raise RuntimeError(
                f"Spectral cache mismatch for dataset '{self.name}': expected "
                f"eigenvectors for {num_nodes} nodes, but found shape "
                f"{list(self.eivec.shape)}.\nRemove '{eival_path}' and "
                f"'{eivec_path}' and run again to rebuild the spectral cache."
            )

    def _read_label_file(self):
        labels_path = self._raw_path("labels")
        if not osp.isfile(labels_path):
            raise FileNotFoundError(
                f"Missing label file for dataset '{self.name}': {labels_path}"
            )
        return torch.from_numpy(
            np.loadtxt(labels_path, dtype=np.int64, ndmin=1)
        ).to(torch.int64)

    def load_evaluation_labels(self, expected_num_classes=None):
        """Read labels for final evaluation; callers enforce protocol timing."""
        labels = self._read_label_file()
        num_nodes = int(self._processed_data.x.size(0))
        if labels.numel() != num_nodes:
            raise ValueError(
                f"Node count mismatch for dataset '{self.name}': expected "
                f"{num_nodes} labels, found {labels.numel()}."
            )
        if expected_num_classes is not None and labels.numel():
            if (
                int(labels.min()) < 0
                or int(labels.max()) >= int(expected_num_classes)
            ):
                raise ValueError(
                    f"Target labels for '{self.name}' are outside classifier "
                    f"range [0, {int(expected_num_classes) - 1}]."
                )
        return labels

    def process(self):
        docs_path = self._raw_path("docs")
        edge_path = self._raw_path("edgelist")
        edge_index = read_txt_array(
            edge_path,
            sep=",",
            dtype=torch.long,
        ).t().contiguous()
        x = torch.from_numpy(
            np.loadtxt(
                docs_path,
                delimiter=",",
                dtype=np.float32,
                ndmin=2,
            )
        )

        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(
                f"Invalid edge list for dataset '{self.name}': expected two "
                f"comma-separated columns in {edge_path}."
            )
        if edge_index.numel() and (
            int(edge_index.min()) < 0 or int(edge_index.max()) >= x.size(0)
        ):
            raise ValueError(
                f"Edge index out of range for dataset '{self.name}': valid "
                f"node IDs are 0 to {x.size(0) - 1}."
            )

        if self.load_labels:
            y = self._read_label_file()
            if x.size(0) != y.numel():
                raise ValueError(
                    f"Node count mismatch for dataset '{self.name}': "
                    f"{docs_path} contains {x.size(0)} feature rows, but the "
                    f"label file contains {y.numel()} labels."
                )
            data = Data(edge_index=edge_index, x=x, y=y)
        else:
            data = Data(edge_index=edge_index, x=x)

        if self.pre_filter is not None and not self.pre_filter(data):
            raise ValueError(
                f"pre_filter rejected the only graph in dataset '{self.name}'."
            )
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])
