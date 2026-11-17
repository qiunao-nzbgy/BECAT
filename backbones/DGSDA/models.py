import torch
from torch import nn
import torch.nn.functional as F
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.utils import get_laplacian
from scipy.special import comb


class Bern_prop(MessagePassing):
    def __init__(self, K, is_source_domain=True, bias=True, **kwargs):
        super(Bern_prop, self).__init__(aggr='add', **kwargs)

        self.K = K
        self.is_source_domain = is_source_domain
        self.cached_terms = None
        self.cached_coefs = None
        self.temp = nn.Parameter(torch.Tensor(self.K + 1), requires_grad=is_source_domain)
        self.reset_parameters()

    def reset_parameters(self):
        if self.is_source_domain:
            self.temp.data.fill_(1)
        else:
            self.temp.data = torch.linspace(1, 0, self.K + 1)

    def get_filter(self):
        TEMP = F.relu(self.temp)
        H = 0

        for k in range(self.K + 1):
            H = H + TEMP[k] * self.cached_coefs[k] * self.cached_terms[k]

        return H

    def forward(self, x, edge_index, edge_weight=None):
        TEMP = F.relu(self.temp)

        edge_index1, norm1 = get_laplacian(edge_index, edge_weight, normalization='sym', dtype=x.dtype,
                                           num_nodes=x.size(self.node_dim))
        edge_index2, norm2 = add_self_loops(edge_index1, -norm1, fill_value=2., num_nodes=x.size(self.node_dim))

        tmp = []
        tmp.append(x)
        for i in range(self.K):
            x = self.propagate(edge_index2, x=x, norm=norm2, size=None)
            tmp.append(x)

        out = (comb(self.K, 0) / (2 ** self.K)) * TEMP[0] * tmp[self.K]

        for i in range(self.K):
            x = tmp[self.K - i - 1]
            x = self.propagate(edge_index1, x=x, norm=norm1, size=None)
            for j in range(i):
                x = self.propagate(edge_index1, x=x, norm=norm1, size=None)

            out = out + (comb(self.K, i + 1) / (2 ** self.K)) * TEMP[i + 1] * x
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)


class BernNet(torch.nn.Module):
    def __init__(self, features, hidden, classes, dropout, dprate=0.0, K=15,
                 use_becat=False, becat_config=None):
        super(BernNet, self).__init__()
        self.lin1 = nn.Linear(features, hidden)
        self.lin2 = nn.Linear(hidden, classes)
        self.prop1 = Bern_prop(K)
        self.prop2 = Bern_prop(K)
        self.prop3 = Bern_prop(K)
        self.prop4 = Bern_prop(K, False)

        self.dprate = dprate
        self.dropout = dropout
        self.use_becat = use_becat
        self.becat = None
        self.becat_active = use_becat

        if self.use_becat:
            from becat.torch import LatentGeometryAdapter

            self.becat = LatentGeometryAdapter(becat_config)
            # BECAT creates its projection layers lazily. Initialize them
            # now so model.parameters() includes them when the optimizer is made.
            with torch.no_grad():
                dummy = torch.zeros(1, hidden)
                self.becat(dummy, dummy, training=False)

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, data, is_source_domain=True):
        return self.forward_with_latent(data, is_source_domain)

    def get_latent(self, data, is_source_domain=True):
        """Return H after the domain-specific Bernoulli propagation."""
        return self.get_latent_from_tensors(
            data.x, data.edge_index, is_source_domain
        )

    def encode_attributes(self, x):
        """Apply the shared attribute encoder with the original dropout."""
        x = F.dropout(x, p=self.dropout, training=self.training)
        return F.relu(self.lin1(x))

    def encode_graph(self, attribute_embedding, edge_index,
                     is_source_domain=True):
        """Apply the original domain-specific spectral propagation."""
        x = F.dropout(
            attribute_embedding, p=self.dprate, training=self.training
        )
        if is_source_domain:
            return self.prop1(x, edge_index)
        return self.prop2(x, edge_index)

    def get_latent_from_tensors(self, x, edge_index,
                                is_source_domain=True):
        """Return H from explicitly placed feature and edge tensors."""
        attribute_embedding = self.encode_attributes(x)
        return self.encode_graph(
            attribute_embedding, edge_index, is_source_domain
        )

    def set_becat_active(self, active=True):
        """Enable or disable BECAT, for example during warmup epochs."""
        self.becat_active = self.use_becat and bool(active)

    def adapt_latent_pair(self, source_latent, target_latent):
        """Jointly adapt an explicitly placed source/target latent pair."""
        if not self.becat_active:
            return source_latent, target_latent, {}

        # The chunk size limits BECAT's query rows; each domain remains the
        # complete reference set for its own latent TopK graph.
        source_geo, target_geo, geo_loss, geo_stats = self.becat.forward_chunked(
            source_latent,
            target_latent,
            max_nodes_per_call=256,
            training=self.training,
        )

        detached_stats = {
            name: value.detach() if isinstance(value, torch.Tensor) else value
            for name, value in geo_stats.items()
        }
        return source_geo, target_geo, {
            'becat_loss': geo_loss,
            'becat_stats': detached_stats,
        }

    def _adapt_latent_pair(self, source_latent, target_latent):
        """Backward-compatible private alias for the public adapter stage."""
        return self.adapt_latent_pair(source_latent, target_latent)

    def classify_latent(self, latent, edge_index):
        """Apply the unchanged classifier and final Bernoulli propagation."""
        x = F.dropout(latent, p=self.dropout, training=self.training)
        x = self.lin2(x)

        x = F.dropout(x, p=self.dprate, training=self.training)
        return self.prop3(x, edge_index)

    def forward_with_latent(self, data, is_source_domain=True, return_latent=False):
        latent = self.get_latent(data, is_source_domain)
        latent_geo, _, extras = self._adapt_latent_pair(latent, latent)
        output = self.classify_latent(latent_geo, data.edge_index)

        if return_latent:
            return output, latent_geo, extras
        return output

    def forward_pair(self, source_data, target_data, return_latent=False):
        """Adapt source/target H jointly and return their final predictions."""
        source_latent = self.get_latent(source_data, True)
        target_latent = self.get_latent(target_data, False)
        source_geo, target_geo, extras = self._adapt_latent_pair(
            source_latent, target_latent
        )
        source_output = self.classify_latent(source_geo, source_data.edge_index)
        target_output = self.classify_latent(target_geo, target_data.edge_index)

        if return_latent:
            extras = dict(extras)
            extras['source_latent'] = source_geo
            extras['target_latent'] = target_geo
        return source_output, target_output, extras

    def get_props(self, x, edge_index, is_source_domain=True):
        return self.get_latent_from_tensors(
            x, edge_index, is_source_domain
        )
