import os.path as osp
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.io import read_txt_array
import torch
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


def _read_citation_graph_inputs(raw_dir, name):
    edge_path = osp.join(raw_dir, '{}_edgelist.txt'.format(name))
    edge_index = read_txt_array(edge_path, sep=',', dtype=torch.long).t()

    docs_path = osp.join(raw_dir, '{}_docs.txt'.format(name))
    content_list = []
    with open(docs_path, 'rb') as docs_file:
        for line in docs_file.readlines():
            line = str(line, encoding="utf-8")
            content_list.append(line.split(","))
    x = torch.from_numpy(np.array(content_list, dtype=float)).to(torch.float)
    return x, edge_index


def load_unlabeled_citation_graph(root, name):
    """Load target features/edges directly, without labels or labeled caches."""
    x, edge_index = _read_citation_graph_inputs(osp.join(root, 'raw'), name)
    return Data(x=x, edge_index=edge_index, num_nodes=x.shape[0])

class CitationDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 name,
                 split_seed=42,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        self.name = name
        self.root = root
        self.split_seed = int(split_seed)
        super(CitationDataset, self).__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])
        self._apply_deterministic_split()

    def _apply_deterministic_split(self):
        """Replace any cached masks with a split controlled by split_seed."""
        num_nodes = int(self.data.num_nodes)
        generator = torch.Generator()
        generator.manual_seed(self.split_seed)
        indices = torch.randperm(num_nodes, generator=generator)

        training_size = int(num_nodes * 0.8)
        validation_size = int(num_nodes * 0.1)
        train_indices = indices[:training_size]
        validation_indices = indices[
            training_size:training_size + validation_size
        ]
        test_indices = indices[training_size + validation_size:]

        self.data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self.data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self.data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self.data.train_mask[train_indices] = True
        self.data.val_mask[validation_indices] = True
        self.data.test_mask[test_indices] = True

    @property
    def raw_file_names(self):
        return ["docs.txt", "edgelist.txt", "labels.txt"]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        x, edge_index = _read_citation_graph_inputs(
            self.raw_dir, self.name
        )

        label_path = osp.join(self.raw_dir, '{}_labels.txt'.format(self.name))
        content_list = []
        with open(label_path, 'rb') as label_file:
            for line in label_file.readlines():
                line = str(line, encoding="utf-8")
                line = line.replace("\r", "").replace("\n", "")
                content_list.append(line)
        y = np.array(content_list, dtype=int)
        y = torch.from_numpy(y).to(torch.int64)

        data_list = []
        data = Data(edge_index=edge_index, x=x, y=y)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        data_list.append(data)

        data, slices = self.collate([data])

        torch.save((data, slices), self.processed_paths[0])


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)),
                                       int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)),
                                       int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = (torch.sum(L2_distance.data) + 1e-6) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)


def MMD(source_feat, target_feat, sampling_num=1000, times=5):
    source_num = source_feat.size(0)
    target_num = target_feat.size(0)

    source_sample = torch.randint(source_num, (times, sampling_num))
    target_sample = torch.randint(target_num, (times, sampling_num))

    mmd = 0
    for i in range(times):
        source_sample_feat = source_feat[source_sample[i]]
        target_sample_feat = target_feat[target_sample[i]]

        mmd = mmd + get_MMD(source_sample_feat, target_sample_feat)

    mmd = mmd / times
    return mmd


def get_MMD(source_feat, target_feat, kernel_mul=2.0, kernel_num=5
            , fix_sigma=None):
    kernels = guassian_kernel(source_feat,
                              target_feat,
                              kernel_mul=kernel_mul,
                              kernel_num=kernel_num,
                              fix_sigma=fix_sigma)

    batch_size = min(int(source_feat.size()[0]), int(target_feat.size()[0]))

    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss
