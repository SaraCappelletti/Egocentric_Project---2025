import torch
import torch.nn as nn

from model.layers import (
    VisualProjection,
    FeatureEncoder,
    CQAttention,
    CQConcatenate,
    ConditionedPredictor,
    Embedding,
    BertEmbedding,
)


class VSLBase(nn.Module):
    def __init__(self, configs, word_vectors):
        super(VSLBase, self).__init__()
        self.configs = configs

        # Video projection to common dim
        self.video_affine = VisualProjection(
            visual_dim=configs.video_feature_dim,
            dim=configs.dim,
            drop_rate=configs.drop_rate,
        )

        # Feature encoder (video and query)
        self.feature_encoder = FeatureEncoder(
            dim=configs.dim,
            num_heads=configs.num_heads,
            kernel_size=7,
            num_layers=4,
            max_pos_len=configs.max_pos_len,
            drop_rate=configs.drop_rate,
        )

        # Cross attention and fusion
        self.cq_attention = CQAttention(dim=configs.dim, drop_rate=configs.drop_rate)
        self.cq_concat = CQConcatenate(dim=configs.dim)

        # Final predictor
        self.predictor = ConditionedPredictor(
            dim=configs.dim,
            num_heads=configs.num_heads,
            drop_rate=configs.drop_rate,
            max_pos_len=configs.max_pos_len,
            predictor=configs.predictor,
        )

        # Query embedding
        if configs.predictor == "bert":
            self.query_affine = nn.Linear(768, configs.dim)
            self.embedding_net = BertEmbedding(configs.text_agnostic)
        else:
            self.embedding_net = Embedding(
                num_words=configs.word_size,
                num_chars=configs.char_size,
                out_dim=configs.dim,
                word_dim=configs.word_dim,
                char_dim=configs.char_dim,
                word_vectors=word_vectors,
                drop_rate=configs.drop_rate,
            )

        self.init_parameters()

    def init_parameters(self):
        def init_weights(m):
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                m.reset_parameters()

        self.apply(init_weights)

    def encode_query(self, word_ids, char_ids):
        if self.configs.predictor == "bert":
            return self.query_affine(self.embedding_net(word_ids))
        else:
            return self.embedding_net(word_ids, char_ids)

    def forward(self, word_ids, char_ids, video_features, v_mask, q_mask):
        video_features = self.video_affine(video_features)
        query_features = self.encode_query(word_ids, char_ids)

        query_features = self.feature_encoder(query_features, mask=q_mask)
        video_features = self.feature_encoder(video_features, mask=v_mask)

        features = self.cq_attention(video_features, query_features, v_mask, q_mask)
        features = self.cq_concat(features, query_features, q_mask)

        start_logits, end_logits = self.predictor(features, mask=v_mask)
        return start_logits, end_logits

    def extract_index(self, start_logits, end_logits):
        return self.predictor.extract_index(start_logits, end_logits)

    def compute_loss(self, start_logits, end_logits, start_labels, end_labels):
        return self.predictor.compute_cross_entropy_loss(
            start_logits=start_logits,
            end_logits=end_logits,
            start_labels=start_labels,
            end_labels=end_labels,
        )
