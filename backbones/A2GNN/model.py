"""A2GNN-MMD model with an optional shared BECAT refinement stage."""

from dataclasses import fields

import torch
import torch.nn.functional as F

from becat.torch import GeometryAdapterConfig, LatentGeometryAdapter

from .layers import GCNConv


class Model(torch.nn.Module):
    """Preserve the original A2GNN encoder/classifier and optional BECAT path."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio

        self.conv1 = GCNConv(self.num_features, self.nhid)
        # Kept for checkpoint/architecture compatibility with original A2GNN.
        self.conv2 = GCNConv(self.nhid, self.nhid)
        self.cls = GCNConv(self.nhid, self.num_classes)

        self.use_becat = bool(args.use_becat)
        self.becat = None
        if self.use_becat:
            config = GeometryAdapterConfig(
                **{
                    field.name: getattr(args, field.name, field.default)
                    for field in fields(GeometryAdapterConfig)
                }
            )
            self.becat = LatentGeometryAdapter(config)

            # BECAT creates its transformer layers lazily. Materialize them
            # before optimizer construction so they are optimized/checkpointed.
            initialization_h = torch.zeros(1, self.nhid)
            with torch.no_grad():
                self.becat(
                    initialization_h,
                    initialization_h,
                    training=False,
                )

        self.encoder_device = None
        self.geometry_device = None

    def place_modules(self, encoder_device, geometry_device):
        """Keep encoder/MMD and BECAT/classifier on their configured devices."""
        self.encoder_device = torch.device(encoder_device)
        self.geometry_device = torch.device(geometry_device)

        self.conv1.to(self.encoder_device)
        self.conv2.to(self.encoder_device)
        if self.use_becat:
            self.becat.to(self.geometry_device)
            self.cls.to(self.geometry_device)
        else:
            self.cls.to(self.encoder_device)
        return self

    @staticmethod
    def _module_device(module):
        try:
            return next(module.parameters()).device
        except StopIteration:
            return next(module.buffers()).device

    def parameter_placement(self):
        """Return stage placement without changing model state."""
        return {
            "encoder": self._module_device(self.conv1),
            "becat": (
                self._module_device(self.becat)
                if self.becat is not None
                else "disabled"
            ),
            "classifier": self._module_device(self.cls),
        }

    def feat_bottleneck(self, x, edge_index, conv_time=30):
        """Original A2GNN raw latent representation H."""
        x = self.conv1(x, edge_index, conv_time)
        return F.relu(x)

    def adapt_pair(self, raw_source_h, raw_target_h):
        """Map raw H_s/H_t to classifier inputs O_s/O_t when BECAT is active."""
        if not self.use_becat:
            return raw_source_h, raw_target_h, raw_source_h.new_zeros(())

        source_h = raw_source_h.to(self.geometry_device, non_blocking=True)
        target_h = raw_target_h.to(self.geometry_device, non_blocking=True)
        source_h, target_h, becat_loss, _ = self.becat(
            source_h,
            target_h,
            training=self.training,
        )
        return source_h, target_h, becat_loss

    def feat_classifier(self, x, edge_index, conv_time=1):
        """Original A2GNN GCN classifier."""
        return self.cls(x, edge_index, conv_time)

    def encode_pair(self, source_data, target_data):
        """Compute raw and classifier-path embeddings for both domains."""
        raw_source_h = self.feat_bottleneck(
            source_data.x,
            source_data.edge_index,
            self.args.source_pnum,
        )
        raw_target_h = self.feat_bottleneck(
            target_data.x,
            target_data.edge_index,
            self.args.target_pnum,
        )
        source_h, target_h, becat_loss = self.adapt_pair(
            raw_source_h,
            raw_target_h,
        )
        return raw_source_h, raw_target_h, source_h, target_h, becat_loss

    def classify_source(self, source_h, source_edge_index):
        return self.feat_classifier(source_h, source_edge_index)

    def classify_target(self, target_h, target_edge_index):
        return self.feat_classifier(target_h, target_edge_index)
