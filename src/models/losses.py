import torch
import torch.nn as nn


class SoftCrossEntropyLoss(nn.Module):

    def __init__(self, red="mean"):
        super(SoftCrossEntropyLoss, self).__init__()
        if red not in ["mean", "sum"]:
            self.reduction = "mean"
        else:
            self.reduction = red
        

    def forward(self, inputs, target):
        """
        :param inputs: predictions
        :param target: target labels
        :return: loss
        """
        log_likelihood = - nn.functional.log_softmax(inputs, dim=1)
        sample_num, _ = target.shape
        if self.reduction == "mean":
            loss_value = torch.sum(torch.mul(log_likelihood, target)) / sample_num
        elif self.reduction == "sum":
            loss_value = torch.sum(torch.mul(log_likelihood, target))
        else:
            raise NotImplementedError("Invalid reduction type for SCE. Only 'mean' or 'sum' are valid.")
        return loss_value


def build_target_tensor(valid_answers, ans_ids):
    """
    Given valid answers for each instance of the batch, it returns the target tensor
    :param valid_answers: Valid answers of this batch, a list of answers per instance.
    :param ans_ids: Id of each answer in our answer vocabulary.
    :return: Tensor of shape (bs, n_class)
    """
    target = torch.zeros((len(valid_answers), len(ans_ids.keys())))

    for i, answers in enumerate(valid_answers):
        
        # SCE Loss
        n = 0
        for ans in answers:
            try:
                target[i, ans_ids[ans]] += 0.1
            except KeyError:
                n += 1
        if n < 10:
            target[i] = target[i] * 10 / (10 - n)
        elif n == 10:
            target[i, ans_ids["UNK"]] = 1.0

    return target
