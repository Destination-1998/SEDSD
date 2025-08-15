import torch.nn as nn
import torch.nn.functional as F
import math
import torch


class Aggregator(nn.Module):
    def __init__(self, dim_in, number_stage, number_net):
        super(Aggregator, self).__init__()
        self.number_stage = number_stage
        self.number_net = number_net
        for i in range(self.number_net):
            setattr(self, 'proj_head_' + str(i), nn.Sequential(
            nn.Linear(dim_in[i] * self.number_stage, dim_in[i]),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in[i], 2)
            ))

    def forward(self, embeddings, logits):
        aggregated_logits = []
        for i in range(self.number_net):
            feature = torch.cat(embeddings,dim=1)
            feature = feature.permute(0, 2, 3, 4, 1).contiguous()
            logit = getattr(self, 'proj_head_' + str(i))(feature)
            weights = F.softmax(logit, dim=-1).permute(0,4,1,2,3).contiguous()
            weighted_logit = 0.
            # for j in range(self.number_stage):
            weighted_logit=(logits[1] * weights.unsqueeze(0))
            # aggregated_logits.append(weighted_logit)
        # return aggregated_logits
        return weighted_logit