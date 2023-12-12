import torch
import torch.nn as nn

class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num = targets.size(0)

        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = m1 * m2

        score = (
            2.0
            * (intersection.sum(1) + self.smooth)
            / (m1.sum(1) + m2.sum(1) + self.smooth)
        )
        score = 1 - score.sum() / num
        return score

import torch.nn.functional as F

class KLTDivergence(nn.Module):
    def __init__(self, temperature=1):
        super(KLTDivergence, self).__init__()
        self.temperature = temperature

    def forward(self, student_output_logits, teacher_output_logits):
        student_sig = torch.sigmoid(student_output_logits / self.temperature)
        teacher_sig = torch.sigmoid(teacher_output_logits / self.temperature)

        # Compute Binary Cross-Entropy Loss
        loss = F.binary_cross_entropy(student_sig, teacher_sig, reduction='mean')

        return (self.temperature ** 2) * loss


# segmentation_loss = dice_loss + bce_loss
# kd_loss = T^2 * kd_loss
# loss = (1 - alpha) * kd_loss + alpha * segmentation_loss