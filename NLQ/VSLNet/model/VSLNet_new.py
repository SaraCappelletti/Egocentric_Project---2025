import torch
from model.vsl_base import VSLBase
from model.layers import HighLightLayer


class VSLNet(VSLBase):
    def __init__(self, configs, word_vectors):
        super(VSLNet, self).__init__(configs, word_vectors)
        self.highlight_layer = HighLightLayer(dim=configs.dim)

    def forward(self, word_ids, char_ids, video_features, v_mask, q_mask):
        video_features = self.video_affine(video_features)
        query_features = self.encode_query(word_ids, char_ids)

        query_features = self.feature_encoder(query_features, mask=q_mask)
        video_features = self.feature_encoder(video_features, mask=v_mask)

        features = self.cq_attention(video_features, query_features, v_mask, q_mask)
        features = self.cq_concat(features, query_features, q_mask)

        h_score = self.highlight_layer(features, v_mask)
        features = features * h_score.unsqueeze(2)

        start_logits, end_logits = self.predictor(features, mask=v_mask)
        return h_score, start_logits, end_logits

    def compute_highlight_loss(self, scores, labels, mask):
        return self.highlight_layer.compute_loss(scores=scores, labels=labels, mask=mask)
