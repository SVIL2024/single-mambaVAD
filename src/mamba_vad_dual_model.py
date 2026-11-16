from collections import OrderedDict
import torch
import torch.nn.functional as F
from torch import nn

from mamba_model import MambaTemporalEncoder, BiMambaEncoder, LayerNorm, QuickGELU
from clip import clip
from utils.layers import GraphConvolution, DistanceAdj


class MambaVAD_Dual(nn.Module):
    """Dual-Mamba VAD via causal frame prediction.

    Core idea: two Mamba encoders, each specialized for one domain.
      Mamba_N → trained on Normal videos → expert on normal temporal dynamics
      Mamba_A → trained on Anomaly videos → expert on anomaly temporal dynamics

    This decouples the conflicting gradient problem in single-Mamba prediction:
     - Mamba_N only needs to predict well on normal data (single objective)
     - Mamba_A only needs to predict well on anomaly data (single objective)

    Test scoring: log(error_N) - log(error_A)  positive = anomalous.

    Three ablation levels:
      Level 1: dual prediction only
      Level 2: +GCN classifier on fused (concat N + A) features
      Level 3: +Ghost Teacher (text-regularized frame alignment)
    """

    def __init__(self, visual_width=512, visual_length=256,
                 mamba_layers=2, d_state=16, d_conv=4, expand=2,
                 dropout=0.0, level=1, pred_stride=1,
                 num_class=14, embed_dim=512,
                 prompt_prefix=10, prompt_postfix=10,
                 device='cuda', mamba_version='mamba1',
                 mamba_only_cls=False, multi_scale=False,
                 multi_scale_factors='1,8,16', single_mamba=False,
                 bidirectional=False, mlp_scale=4,
                 future_align='shift',
                 **mamba_kwargs):
        super().__init__()

        self.level = level
        self.pred_stride = pred_stride
        self.visual_length = visual_length
        self.visual_width = visual_width
        self.device = device
        self.prompt_prefix = prompt_prefix
        self.prompt_postfix = prompt_postfix
        self.num_class = num_class
        self.mamba_only_cls = mamba_only_cls
        self.multi_scale = multi_scale
        self.single_mamba = single_mamba
        self.bidirectional = bidirectional
        # 'shift'  -> score_future[t] = head(h_t) for every frame, and the
        #             training loss is responsible for the time shift
        #             (h_t predicts s_{t+k}); nothing padded enters the loss.
        # 'legacy' -> the submitted behaviour: head applied to x[:, :-1] only,
        #             padded at the tail, compared slot-wise with the score of
        #             the same frame.
        self.future_align = str(future_align).lower()

        # ===== Position embedding (always shared) =====
        max_pos = max(visual_length, 1024)
        self.frame_position_embeddings = nn.Embedding(max_pos, visual_width)
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)

        if single_mamba:
            if bidirectional:
                self.mamba = BiMambaEncoder(
                    d_model=visual_width, n_layer=mamba_layers,
                    d_state=d_state, d_conv=d_conv, expand=expand,
                    dropout=dropout, mlp_scale=mlp_scale,
                    mamba_version=mamba_version, **mamba_kwargs)
            else:
                self.mamba = MambaTemporalEncoder(
                    d_model=visual_width, n_layer=mamba_layers,
                    d_state=d_state, d_conv=d_conv, expand=expand,
                    dropout=dropout, attn_window=0, mlp_scale=mlp_scale,
                    mamba_version=mamba_version, **mamba_kwargs)

            self.pred = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(visual_width, visual_width * 4)),
                ("gelu", QuickGELU()),
                ("c_proj", nn.Linear(visual_width * 4, visual_width))
            ]))
            self.future_score_head = nn.Linear(visual_width, 1)
        else:
            mamba_cls = BiMambaEncoder if bidirectional else MambaTemporalEncoder
            mamba_kw_enc = dict(d_model=visual_width, n_layer=mamba_layers,
                                d_state=d_state, d_conv=d_conv, expand=expand,
                                dropout=dropout, mlp_scale=mlp_scale,
                                mamba_version=mamba_version, **mamba_kwargs)
            if not bidirectional:
                mamba_kw_enc['attn_window'] = 0
            self.mamba_normal = mamba_cls(**mamba_kw_enc)
            self.mamba_anomaly = mamba_cls(**mamba_kw_enc)

            self.pred_n = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(visual_width, visual_width * 4)),
                ("gelu", QuickGELU()),
                ("c_proj", nn.Linear(visual_width * 4, visual_width))
            ]))
            self.pred_a = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(visual_width, visual_width * 4)),
                ("gelu", QuickGELU()),
                ("c_proj", nn.Linear(visual_width * 4, visual_width))
            ]))

        # ===== Multi-scale fusion =====
        if multi_scale:
            if isinstance(multi_scale_factors, str):
                parsed = tuple(int(x) for x in multi_scale_factors.split(','))
            else:
                parsed = tuple(multi_scale_factors)
            self.multi_scale_factors = parsed
            num_scales = len([f for f in parsed if f > 1]) + 1
        else:
            self.multi_scale_factors = ()
            num_scales = 1
        feat_mult = 1 if single_mamba else 2
        self.fusion_linear = nn.Linear(visual_width * feat_mult * num_scales, visual_width)

        # ===== Level 2: classification branch =====
        if level >= 2:
            if mamba_only_cls:
                self.classifier = nn.Sequential(
                    nn.Linear(visual_width, visual_width // 2),
                    nn.Dropout(0.5),
                    nn.ReLU(),
                    nn.Linear(visual_width // 2, 1)
                )
            else:
                width = int(visual_width / 2)
                self.gc1 = GraphConvolution(visual_width, width, residual=True)
                self.gc2 = GraphConvolution(width, width, residual=True)
                self.gc3 = GraphConvolution(visual_width, width, residual=True)
                self.gc4 = GraphConvolution(width, width, residual=True)
                self.disAdj = DistanceAdj()
                self.linear = nn.Linear(visual_width, visual_width)
                self.gelu = QuickGELU()
                self.mlp2 = nn.Sequential(OrderedDict([
                    ("c_fc", nn.Linear(visual_width, visual_width * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(visual_width * 4, visual_width))
                ]))
                self.classifier = nn.Linear(visual_width, 1)

        # ===== Level 3: Ghost Teacher (text-regularized Mamba) =====
        if level >= 3:
            self.clipmodel, _ = clip.load("ViT-B/16", device)
            for clip_param in self.clipmodel.parameters():
                clip_param.requires_grad = False
            self.text_prompt_embeddings = nn.Embedding(77, embed_dim)
            self.mlp1 = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(embed_dim, embed_dim * 4)),
                ("gelu", QuickGELU()),
                ("c_proj", nn.Linear(embed_dim * 4, embed_dim))
            ]))
            self.frozen_clip_text_ln = LayerNorm(embed_dim)
            nn.init.normal_(self.text_prompt_embeddings.weight, std=0.01)
            self.ghost_proj = nn.Linear(visual_width, visual_width, bias=False)

    # ──────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────

    def forward(self, visual, lengths=None, text=None, normal_mask=None):
        return self._forward_full(visual, lengths, text, normal_mask=normal_mask)

    def _forward_full(self, visual, lengths, text, normal_mask=None):
        B, T, D = visual.shape

        if self.single_mamba:
            x = self._encode_video(visual, self.mamba)
            x_vis = x.permute(1, 0, 2)
            error = self._compute_prediction_error(visual, x, self.pred, lengths)

            # Future score prediction from raw Mamba hidden state.
            # 'shift': predict from every frame and let the loss apply the
            #          time shift, so no replicated padding is supervised.
            # 'legacy': the submitted alignment (see __init__).
            if self.future_align == 'legacy':
                score_pred = self.future_score_head(x_vis[:, :-1, :])
                score_pred = F.pad(score_pred, (0, 0, 0, 1), mode='replicate')
            else:
                score_pred = self.future_score_head(x_vis)

            if self.multi_scale:
                scales = [x_vis]
                for f in self.multi_scale_factors:
                    if f == 1:
                        continue
                    pooled = F.avg_pool1d(visual.float().permute(0, 2, 1),
                                          kernel_size=f, stride=f).permute(0, 2, 1).to(visual.dtype)
                    xs = self._encode_video(pooled, self.mamba).permute(1, 0, 2)
                    xs = F.interpolate(xs.permute(0, 2, 1), size=T, mode='linear', align_corners=False).permute(0, 2, 1)
                    scales.append(xs)
                x_vis = torch.cat(scales, dim=-1)

            outputs = dict(error=error, hidden_state=x_vis, score_future=score_pred)
            x_fused = self.fusion_linear(x_vis)

        else:
            x_N = self._encode_video(visual, self.mamba_normal)
            x_A = self._encode_video(visual, self.mamba_anomaly)
            x_vis_N_1, x_vis_A_1 = x_N.permute(1, 0, 2), x_A.permute(1, 0, 2)

            error_N = self._compute_prediction_error(visual, x_N, self.pred_n, lengths)
            error_A = self._compute_prediction_error(visual, x_A, self.pred_a, lengths)

            valid_T = error_N.shape[1]
            k = max(1, min(valid_T, int(valid_T * 0.1)))
            score_N = error_N.topk(k, dim=1)[0].mean(dim=1)
            score_A = error_A.topk(k, dim=1)[0].mean(dim=1)

            if self.multi_scale:
                scale_N = [x_vis_N_1]
                scale_A = [x_vis_A_1]
                for f in self.multi_scale_factors:
                    if f == 1:
                        continue
                    pooled = F.avg_pool1d(visual.float().permute(0, 2, 1),
                                          kernel_size=f, stride=f).permute(0, 2, 1).to(visual.dtype)
                    x_Ns = self._encode_video(pooled, self.mamba_normal).permute(1, 0, 2)
                    x_As = self._encode_video(pooled, self.mamba_anomaly).permute(1, 0, 2)
                    x_Ns = F.interpolate(x_Ns.permute(0, 2, 1), size=T, mode='linear', align_corners=False).permute(0, 2, 1)
                    x_As = F.interpolate(x_As.permute(0, 2, 1), size=T, mode='linear', align_corners=False).permute(0, 2, 1)
                    scale_N.append(x_Ns)
                    scale_A.append(x_As)
                x_vis_N = torch.cat(scale_N, dim=-1)
                x_vis_A = torch.cat(scale_A, dim=-1)
            else:
                x_vis_N, x_vis_A = x_vis_N_1, x_vis_A_1

            outputs = dict(error_N=error_N, error_A=error_A,
                           score_N=score_N, score_A=score_A,
                           hidden_state_N=x_vis_N, hidden_state_A=x_vis_A)
            x_fused = self.fusion_linear(torch.cat([x_vis_N, x_vis_A], dim=-1))

        outputs['hidden_state_fused'] = x_fused

        if self.level >= 2:
            if self.mamba_only_cls:
                outputs['logits1'] = self.classifier(x_fused)
            else:
                self._apply_gcn(outputs, lengths)
        if self.level >= 3 and text is not None:
            self._apply_text(outputs, text)
        return outputs

    def _apply_gcn(self, outputs, lengths):
        x = outputs['hidden_state_fused']
        B, T, _ = x.shape
        adj = self._build_adj(x, lengths)
        disadj = self.disAdj(B, T)

        x1_h = self.gelu(self.gc1(x, adj))
        x2_h = self.gelu(self.gc3(x, disadj))
        x1 = self.gelu(self.gc2(x1_h, adj))
        x2 = self.gelu(self.gc4(x2_h, disadj))
        x_gcn = torch.cat((x1, x2), 2)
        x_gcn = self.linear(x_gcn)

        outputs['logits1'] = self.classifier(x_gcn + self.mlp2(x_gcn))

    def _apply_text(self, outputs, text):
        """Ghost Teacher: frame-level MSE targets via hard top-K MIL selection."""
        z_text = self._encode_text(text)
        z_text = z_text.detach()

        x_fused = outputs['hidden_state_fused']
        h_proj = self.ghost_proj(x_fused)

        outputs['text_features'] = z_text
        outputs['h_proj'] = h_proj

    # ──────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────

    def _encode_video(self, images, mamba_module):
        """[B, T, D] → position embed → mamba → [T, B, D]"""
        B, T, D = images.shape
        images = images.permute(1, 0, 2)
        pos_ids = torch.arange(T, device=self.device)
        pos_ids = pos_ids.unsqueeze(0).expand(B, -1)
        pos_emb = self.frame_position_embeddings(pos_ids)
        pos_emb = pos_emb.permute(1, 0, 2)
        images = images + pos_emb
        x = mamba_module(images)
        return x

    def _compute_prediction_error(self, visual, mamba_out, pred_head, lengths=None):
        """Predict f_{t+stride} from h_t. Returns [B, T-stride] per-frame MSE."""
        stride = self.pred_stride
        mamba_b = mamba_out.permute(1, 0, 2)
        pred_next = pred_head(mamba_b[:, :-stride, :])
        target_next = visual[:, stride:, :]
        error = (pred_next - target_next).pow(2).mean(dim=-1)

        if lengths is not None:
            mask = torch.arange(error.shape[1], device=error.device)
            mask = mask.unsqueeze(0) < (lengths - stride).clamp(min=0).unsqueeze(1)
            error = error * mask.float()

        return error

    def _build_adj(self, x, lengths):
        """Similarity-based adjacency (from VadCLIP)."""
        x2 = x.matmul(x.permute(0, 2, 1))
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True)
        x_norm_x = x_norm.matmul(x_norm.permute(0, 2, 1))
        x2 = x2 / (x_norm_x + 1e-20)

        output = torch.zeros_like(x2).to(x.device)
        if lengths is None:
            for i in range(x.shape[0]):
                adj2 = F.threshold(x2[i], 0.7, 0)
                output[i] = F.softmax(adj2, dim=1)
        else:
            for i in range(len(lengths)):
                seq_len = int(lengths[i].item())
                tmp = x2[i, :seq_len, :seq_len]
                adj2 = F.threshold(tmp, 0.7, 0)
                output[i, :seq_len, :seq_len] = F.softmax(adj2, dim=1)
        return output

    def _encode_text(self, text):
        """Same as VadCLIP encode_textprompt."""
        word_tokens = clip.tokenize(text).to(self.device)
        word_embedding = self.clipmodel.encode_token(word_tokens)
        text_emb = self.text_prompt_embeddings(
            torch.arange(77).to(self.device))
        text_emb = text_emb.unsqueeze(0).repeat([len(text), 1, 1])
        text_tokens = torch.zeros(len(text), 77).to(self.device)

        for i in range(len(text)):
            ind = torch.argmax(word_tokens[i], -1)
            text_emb[i, 0] = word_embedding[i, 0]
            pp = self.prompt_prefix
            po = self.prompt_postfix
            text_emb[i, pp + 1: pp + ind] = word_embedding[i, 1: ind]
            text_emb[i, pp + ind + po] = word_embedding[i, ind]
            text_tokens[i, pp + ind + po] = word_tokens[i, ind]

        text_features = self.clipmodel.encode_text(text_emb, text_tokens)
        return text_features
