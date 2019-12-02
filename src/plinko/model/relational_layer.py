import torch.nn as nn
from ..misc import utils

class RelationalLayer(nn.Module):
    def __init__(self,
                 object_embed_size,
                 ball_embed_size,
                 output_size
                 ):
        super().__init__()
        self.ball_embed_size = ball_embed_size
        self.object_embed_size = object_embed_size
        self.output_size = output_size

        self.mlp = MLP(input_size=[ball_embed_size, object_embed_size],
                       hidden_layer_size=[24, 24],
                       output_size=[output_size, 1])  # vector, score logit

    def forward(self, objects, ball):
        """
        objects: tensor of shape [B, K, Eo] (batch, number of objects, embedding)
        ball: tensor of shape [B, T, Eb] (batch, timesteps, embedding)
        return tensor of shape [B, T, D] (batch, timestemps, output embedding size)
        """
        # want ball: [B, T, K, Eb]
        # want objects: [B, T, K, Eo]
        B, T, Eb = ball.shape
        B, K, Eo = objects.shape
        ball = utils.expand_along_dim(ball, K, 2)
        objects = utils.expand_along_dim(objects, T, 1)
        h, logit = self.mlp(ball, objects)  # h: [B, T, K, D], # logit: [B, T, K, 1]
        scores = logit.squeeze(-1).softmax(-1)  # [B, T, K]
        h = utils.broadcast_multiply(scores, h).sum(2)
        return h