"""Experiment-specific models (CLIP baseline)."""

import torch
import torch.nn.functional as F
from torch import nn


class CLIPBaseline(nn.Module):
    """Pure CLIP baseline: position embedding → avg pooling → MLP classifier.

    This model has no Mamba encoder, no temporal modeling beyond trivial
    frame-wise averaging.  It serves as the lower-bound reference for the
    ablation study.

    The interface matches ``MambaVAD_Dual`` so that the same training and
    testing functions from ``ucf_train_mamba_vad_dual`` can be used
    without modification.
    """
    def __init__(self, visual_width=512, num_class=14, dropout=0.0,
                 pred_stride=2, device='cuda'):
        super().__init__()
        self.single_mamba = True       # match MambaVAD_Dual interface
        self.pred_stride = pred_stride
        self.device = device

        # Position embeddings (same init as MambaVAD_Dual)
        self.frame_position_embeddings = nn.Embedding(1024, visual_width)
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)

        # Dummy Mamba module (Identity — no learnable params)
        self.mamba = nn.Identity()

        # MLP classifier (matches mamba_only_cls path)
        self.classifier = nn.Sequential(
            nn.Linear(visual_width, visual_width // 2),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(visual_width // 2, 1),
        )

        # Dummy future score head (satisfies interface, not used effectively)
        self.future_score_head = nn.Linear(visual_width, 1)

    def forward(self, visual, lengths=None, text=None, normal_mask=None):
        B, T, D = visual.shape

        # Position embedding
        pos_ids = torch.arange(T, device=visual.device)
        pos_ids = pos_ids.unsqueeze(0).expand(B, -1)
        pos_emb = self.frame_position_embeddings(pos_ids)
        x = visual + pos_emb

        # Dummy error: zeros [B, T-1] – satisfies the train/test interface
        # (loss_pred from error is computed but never added to loss at level 2)
        err_len = max(1, T - self.pred_stride)
        dummy_error = torch.zeros(B, err_len, device=visual.device)

        # Hidden state (used for prototype loss in training)
        outputs = dict(
            error=dummy_error,
            hidden_state_fused=x,       # [B, T, D]
            hidden_state=x.permute(1, 0, 2),  # [T, B, D] for error calc
        )

        # Frame-level classifier logits
        logits1 = self.classifier(x)    # [B, T, 1]
        outputs['logits1'] = logits1

        # Dummy score_future (satisfies future loss computation)
        if T > 1:
            score_future = self.future_score_head(x[:, :-1, :])
            outputs['score_future'] = F.pad(
                score_future, (0, 0, 0, 1), mode='replicate')
        else:
            outputs['score_future'] = torch.zeros(B, 1, 1, device=visual.device)

        return outputs
