import torch.nn as nn

from modules import backbones


class gaze_network(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True, use_face=False, num_glimpses=1):
        super(gaze_network, self).__init__()

        backbone_fn = backbones.get(backbone)
        if backbone_fn is None:
            raise ValueError(
                "Unknown backbone '{}', available: {}".format(
                    backbone, sorted(backbones.keys())))
        self.gaze_network = backbone_fn(pretrained=pretrained)

        # fc in_features == 512 * block.expansion (512 for 18/34, 2048 for the rest)
        feature_dim = self.gaze_network.fc.in_features

        self.gaze_fc = nn.Sequential(
            nn.Linear(feature_dim, 2),
        )

    def forward(self, x):
        feature = self.gaze_network(x)
        feature = feature.view(feature.size(0), -1)
        gaze = self.gaze_fc(feature)

        return gaze
