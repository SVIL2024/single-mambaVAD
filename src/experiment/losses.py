"""Auxiliary objectives for MambaVAD, isolated so they can be tested without
``mamba_ssm`` (which is CUDA-only and therefore unavailable on a CPU-only dev
box).

Two terms live here, both motivated by the KBS review:

``prototype_loss``
    Normal-prototype contrastive term.  The repulsive half of the original
    implementation acted on the *video-level mean* of the hidden states, so a
    video that contains a short anomaly also pushes its many normal frames away
    from the prototype (reviewer 1, point 2).  ``pool`` selects what the
    repulsive term operates on:

    ``'mean'``  the video mean, i.e. the original behaviour (default, and
                numerically identical to
                ``ucf_train_mamba_vad_dual.normal_prototype_loss``),
    ``'topk'``  the k highest-scoring frames of each anomalous video, where the
                scores come from the classifier head and are detached,
    ``'soft'``  a softmax-weighted frame average, temperature ``tau``.

``future_consistency_loss``
    The future-score bridge.  The original code aligned the prediction from
    ``h_t`` with the *score of frame t itself* (the head saw ``x[:, :-1]`` and the
    result was then padded at the tail), so the term was a self-consistency
    constraint and, with a bidirectional encoder, almost free to satisfy.  This
    implementation states the shift explicitly: ``h_t`` predicts ``s_{t+k}``,
    padded positions are masked out of the loss, and ``k`` is a free parameter
    so that a genuine multi-step forecast can be measured.  ``align='legacy'``
    reproduces the original (padded, unshifted) objective for bit-exact reruns.

Every function is pure tensor code: no Mamba, no CUDA, no dataloader.
"""
import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────

def valid_mask(lengths, num_frames, batch=1, device=None):
    """[B, T] boolean mask of the frames that belong to the clip."""
    if lengths is None:
        dev = device if device is not None else torch.device('cpu')
        return torch.ones(batch, num_frames, dtype=torch.bool, device=dev)
    lengths = lengths.to(torch.long)
    ar = torch.arange(num_frames, device=lengths.device).unsqueeze(0)
    return ar < lengths.unsqueeze(1)


def masked_mean_features(h, lengths):
    """Mean over the valid frames of ``h`` [B, T, D] -> [B, D].

    Same arithmetic as the training loop (mask, sum, divide by length) so that
    ``pool='mean'`` stays comparable with the numbers already reported.
    """
    if lengths is None:
        return h.mean(dim=1)
    m = valid_mask(lengths, h.size(1), batch=h.size(0), device=h.device)
    m = m.to(h.dtype).unsqueeze(-1)
    denom = lengths.to(h.dtype).unsqueeze(1).clamp(min=1)
    return (h * m).sum(dim=1) / denom


def pooled_repulsion_features(h_frames, scores, lengths, pool='mean',
                              topk_ratio=0.1, tau=0.1):
    """Pool the frames of each video into one vector for the repulsive term.

    ``h_frames`` [B, T, D] hidden states, ``scores`` [B, T] detached anomaly
    scores, ``lengths`` [B] valid frame counts.

    Returns ``(pooled [B, D], coverage [B])`` where coverage is the fraction of
    valid frames that actually enter the repulsion.  For ``'mean'`` it is 1.0 by
    definition and the returned tensor equals :func:`masked_mean_features`.
    """
    B, T, D = h_frames.shape
    pool = (pool or 'mean').lower()
    if pool == 'mean':
        cov = torch.ones(B, device=h_frames.device)
        return masked_mean_features(h_frames, lengths), cov

    m = valid_mask(lengths, T, batch=B, device=h_frames.device)
    if pool == 'topk':
        neg = torch.finfo(scores.dtype).min
        s = scores.masked_fill(~m, neg)
        r = 1.0 if topk_ratio is None else float(topk_ratio)
        lens = (lengths.to(torch.long) if lengths is not None
                else torch.full((B,), T, dtype=torch.long,
                                device=h_frames.device))
        k = torch.ceil(lens.to(torch.float32) * r).to(torch.long).clamp(min=1)
        k = torch.minimum(k, lens)
        k_max = int(k.max().item())
        idx = s.topk(k_max, dim=1).indices                    # [B, k_max]
        keep = (torch.arange(k_max, device=h_frames.device)
                .unsqueeze(0) < k.unsqueeze(1))               # [B, k_max]
        w = keep.to(h_frames.dtype)                           # [B, k_max]
        feat = h_frames.gather(
            1, idx.unsqueeze(-1).expand(-1, -1, D))            # [B, k_max, D]
        pooled = (feat * w.unsqueeze(-1)).sum(dim=1) / w.sum(
            dim=1, keepdim=True).clamp(min=1)
        cov = (k.to(h_frames.dtype) / lens.clamp(min=1).to(h_frames.dtype))
        return pooled, cov

    if pool == 'soft':
        neg = torch.finfo(scores.dtype).min
        s = scores.masked_fill(~m, neg)
        w = torch.softmax(s / max(float(tau), 1e-6), dim=1) * m.to(scores.dtype)
        w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-12)
        pooled = (h_frames * w.unsqueeze(-1)).sum(dim=1)
        # effective number of contributing frames (perplexity-like)
        cov = torch.exp(-(w.clamp(min=1e-12) * w.clamp(min=1e-12).log()
                          ).sum(dim=1))
        if lengths is not None:
            cov = cov / lengths.to(cov.dtype).clamp(min=1)
        return pooled, cov

    raise ValueError("unknown pooling mode: %r (expected mean|topk|soft)" % (pool,))


# ──────────────────────────────────────────────────────────────
# prototype contrastive term
# ──────────────────────────────────────────────────────────────

def prototype_loss(h_pooled, binary_labels, margin=0.5, device='cuda',
                   h_frames=None, scores=None, lengths=None, pool='mean',
                   topk_ratio=0.1, tau=0.1):
    """Normal-prototype contrastive loss with configurable repulsion pooling.

    Parameters
    ----------
    h_pooled : [B, D]  video-level (masked-mean) hidden states.
    binary_labels : [B] 0 = normal video, 1 = anomalous video.
    h_frames : [B, T, D] required when ``pool != 'mean'``.
    scores : [B, T] anomaly scores used by ``'topk'``/``'soft'``; detached
        inside this function, so the pooling cannot backprop into the
        classifier that produced them.
    lengths : [B] valid frame counts (``None`` means all frames are valid).

    Returns a dict with ``loss``, ``proto``, per-group similarities and the
    repulsion ``coverage`` (fraction of anomalous frames actually repelled).
    """
    if pool is None:
        pool = 'mean'
    h_norm = F.normalize(h_pooled, dim=-1)
    normal_sel = (binary_labels == 0)
    anomaly_sel = (binary_labels == 1)
    h_normal = h_norm[normal_sel]
    h_anomaly = h_norm[anomaly_sel]

    if h_normal.numel() == 0 or h_anomaly.numel() == 0:
        zero = torch.tensor(0.0, device=device, requires_grad=False)
        return dict(loss=zero, proto=h_norm.mean(dim=0).detach(),
                    sim_normal=zero.detach(), sim_anomaly=zero.detach(),
                    coverage=zero.detach(), n_normal=int(h_normal.shape[0]),
                    n_anomaly=int(h_anomaly.shape[0]))

    proto = F.normalize(h_normal.mean(dim=0), dim=-1)

    if str(pool).lower() == 'mean':
        sim_normal = h_normal @ proto
        sim_anomaly = h_anomaly @ proto
        coverage = torch.ones(1, device=h_normal.device).squeeze()
    else:
        if h_frames is None or scores is None:
            raise ValueError("pool=%r needs h_frames and scores" % (pool,))
        s = scores.detach()
        len_anom = None if lengths is None else lengths[anomaly_sel]
        pooled_a, cov = pooled_repulsion_features(
            h_frames[anomaly_sel], s[anomaly_sel], len_anom,
            pool=pool, topk_ratio=topk_ratio, tau=tau)
        sim_normal = h_normal @ proto
        sim_anomaly = F.normalize(pooled_a, dim=-1) @ proto
        coverage = cov.mean()

    loss_normal = (1.0 - sim_normal).mean()
    loss_anomaly = F.relu(sim_anomaly + margin).mean()
    return dict(loss=loss_normal + loss_anomaly,
                proto=proto.detach(),
                sim_normal=sim_normal.mean().detach(),
                sim_anomaly=sim_anomaly.mean().detach(),
                coverage=(coverage.detach() if torch.is_tensor(coverage)
                          else torch.tensor(float(coverage))),
                n_normal=int(h_normal.shape[0]),
                n_anomaly=int(h_anomaly.shape[0]))


# ──────────────────────────────────────────────────────────────
# future-score consistency term
# ──────────────────────────────────────────────────────────────

def future_consistency_loss(score_pred, score_target, lengths=None, horizon=1):
    """MSE between ``head(h_t)`` and the score of frame ``t + horizon``.

    ``score_pred`` and ``score_target`` are [B, T, 1] with entry ``t`` the
    prediction made from frame ``t`` and the (detached) classifier score of
    frame ``t``.  Pairs whose target frame falls outside the clip are skipped.

    Returns a dict with ``loss``, ``n_pairs`` and ``coverage`` (fraction of
    prediction slots that are supervised).
    """
    k = int(horizon)
    if k <= 0:
        raise ValueError("horizon must be >= 1, got %r" % (horizon,))
    B, T, _ = score_pred.shape
    if T <= k:
        zero = score_pred.sum() * 0.0
        return dict(loss=zero, n_pairs=0, coverage=torch.zeros(()).squeeze())

    pred = score_pred[:, :T - k, :]
    tgt = score_target[:, k:, :]
    if lengths is None:
        m = torch.ones(pred.shape[:2], dtype=torch.bool,
                       device=pred.device)
    else:
        lens = lengths.to(torch.long)
        ar = torch.arange(T - k, device=pred.device).unsqueeze(0)
        m = (ar + k) < lens.unsqueeze(1)
    mf = m.to(pred.dtype).unsqueeze(-1)
    denom = mf.sum().clamp(min=1.0)
    loss = (((pred - tgt) ** 2) * mf).sum() / denom
    return dict(loss=loss,
                n_pairs=int(m.sum().item()),
                coverage=m.to(torch.float32).mean().detach())


def legacy_future_loss(score_pred, score_target):
    """Exactly the objective the submitted code optimised.

    The prediction for slot ``t`` came from ``h_t`` (the head was applied to
    ``x[:, :-1]`` and the result padded at the tail), and it was compared with
    the score of the same frame; the padded slot contributed a constant.
    """
    return F.mse_loss(score_pred, score_target.detach())


# ──────────────────────────────────────────────────────────────
# glue used by the training loop
# ──────────────────────────────────────────────────────────────

def compute_aux_terms(h_fused, logits1, score_future, lengths, normal_mask,
                      margin=0.5, pool='mean', topk_ratio=0.1, tau=0.1,
                      horizon=1, align='shift'):
    """Assemble both auxiliary terms from the model outputs.

    Mirrors the block that used to live inline in ``run_one.train_and_eval`` so
    that it can be tested with a fake model and so that both terms see exactly
    the same tensors as the classifier loss.
    """
    h_pooled = masked_mean_features(h_fused, lengths)
    binary_labels = (~normal_mask).to(torch.long)
    scores = torch.sigmoid(logits1.squeeze(-1)).detach()

    p = prototype_loss(h_pooled, binary_labels, margin=margin,
                       device=h_fused.device, h_frames=h_fused,
                       scores=scores, lengths=lengths, pool=pool,
                       topk_ratio=topk_ratio, tau=tau)

    if score_future is None:
        f = dict(loss=None, n_pairs=0, coverage=torch.zeros(()))
    elif str(align).lower() == 'legacy':
        f = dict(loss=legacy_future_loss(score_future, logits1),
                 n_pairs=int(score_future.shape[0] * score_future.shape[1]),
                 coverage=torch.ones(()))
    else:
        f = future_consistency_loss(score_future, logits1.detach(),
                                    lengths=lengths, horizon=horizon)

    return dict(loss_proto=p['loss'], proto=p['proto'],
                sim_normal=p['sim_normal'], sim_anomaly=p['sim_anomaly'],
                proto_coverage=p['coverage'],
                loss_future=f['loss'], future_pairs=f['n_pairs'],
                future_coverage=f['coverage'])
