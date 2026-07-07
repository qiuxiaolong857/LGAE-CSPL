import torch
import torch.nn.functional as F

ID2NAME = {
    0: "fruitlet",
    1: "hard",
    2: "mature",
    3: "first_dilatation",
    4: "growing",
    5: "second_dilatation"
}
NAME2ID = {v: k for k, v in ID2NAME.items()}

groups = [
    ["fruitlet", "first_dilatation"],      # coarse 0  -> fine_head_0: [fruitlet, first_dilatation]
    ["growing", "hard"],                   # coarse 1  -> fine_head_1: [growing, hard]
    ["second_dilatation", "mature"],       # coarse 2  -> fine_head_2: [second_dilatation, mature]
]

EPS = 1e-8

def stage_to_coarse_fine_soft(q_stage_6: torch.Tensor):
    """
    q_stage_6: (B,6)  soft distribution over 6 stages (sum=1)
    returns:
      q_coarse: (B,3)
      q_fine  : (B,3,2)  order matches `groups`
    """
    B = q_stage_6.size(0)
    q_coarse = []
    q_fine = []

    for g, (a_name, b_name) in enumerate(groups):
        a = NAME2ID[a_name]
        b = NAME2ID[b_name]

        qa = q_stage_6[:, a]
        qb = q_stage_6[:, b]

        qc = qa + qb                       # (B,)
        q_coarse.append(qc)

        denom = qc + EPS
        qf = torch.stack([qa/denom, qb/denom], dim=1)  # (B,2) 顺序与groups一致
        q_fine.append(qf)

    q_coarse = torch.stack(q_coarse, dim=1)  # (B,3)
    q_fine = torch.stack(q_fine, dim=1)      # (B,3,2)

    # 保险：归一化
    q_coarse = q_coarse / (q_coarse.sum(dim=1, keepdim=True) + EPS)

    return q_coarse, q_fine


def hard_stage_to_coarse_fine(y_stage: torch.Tensor):
    """
    y_stage: (B,) stage id in 0..5
    returns:
      y_coarse: (B,) in 0..2
      y_fine  : (B,) in 0..1   (在该coarse组内，0表示groups[g][0]，1表示groups[g][1])
    """
    # 建一个 stage->(coarse,fine) 的查表
    stage_to_pair = {}
    for g, (a_name, b_name) in enumerate(groups):
        a = NAME2ID[a_name]
        b = NAME2ID[b_name]
        stage_to_pair[a] = (g, 0)
        stage_to_pair[b] = (g, 1)

    y_coarse = torch.empty_like(y_stage)
    y_fine = torch.empty_like(y_stage)

    for i in range(y_stage.numel()):
        g, f = stage_to_pair[int(y_stage[i].item())]
        y_coarse[i] = g
        y_fine[i] = f

    return y_coarse, y_fine


def kl_distill(student_logits, teacher_probs, T=4.0):
    """
    student_logits: (B,C)
    teacher_probs : (B,C) already prob distribution
    """
    log_p_s = F.log_softmax(student_logits / T, dim=-1)
    return F.kl_div(log_p_s, teacher_probs, reduction="batchmean") * (T*T)


def distill_loss_for_student(coarse_logits, fine_logits_all, y_stage, q_table, alpha=0.3, T=4.0):
    """
    q_table: (6,6) tensor, row k is prototype distribution q_k over 6 stages
    """
    B = y_stage.size(0)
    device = coarse_logits.device

    # ---- hard targets for your hierarchical classifier ----
    y_coarse, y_fine = hard_stage_to_coarse_fine(y_stage)

    # hard CE
    L_hard_coarse = F.cross_entropy(coarse_logits, y_coarse)
    # only the matched branch gets hard CE
    idx = torch.arange(B, device=device)
    L_hard_fine = F.cross_entropy(fine_logits_all[idx, y_coarse, :], y_fine)
    L_hard = L_hard_coarse + L_hard_fine

    # ---- distillation soft targets from q_table ----
    q_stage_6 = q_table[y_stage].to(device)           # (B,6)
    q_coarse, q_fine = stage_to_coarse_fine_soft(q_stage_6)  # (B,3), (B,3,2)

    # coarse distill
    Ld_coarse = kl_distill(coarse_logits, q_coarse, T=T)

    # fine distill: distill all 3 branches but weight by teacher coarse prob
    Ld_fine = 0.0
    for g in range(3):
        w = q_coarse[:, g].detach()  # (B,)
        # per-sample KL
        log_p_s = F.log_softmax(fine_logits_all[:, g, :] / T, dim=-1)
        kl_per = F.kl_div(log_p_s, q_fine[:, g, :], reduction="none").sum(dim=1) * (T*T)  # (B,)
        Ld_fine = Ld_fine + (w * kl_per).mean()

    L_distill = Ld_coarse + Ld_fine

    return L_hard + alpha * L_distill
