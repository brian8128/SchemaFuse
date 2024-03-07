import torch

def multiple_choice_cross_entropy(preds, labels):
  """
  A replacement for torch.nn.functional.cross_entropy_loss where the labels can 
  contain more than one possibly correct answer. These labels are combined with 
  a logical OR instead of a logical AND so either answer is regarded as correct
  even if zero probability is assigned to the/an other possible correct answer. 

  preds - The same as in F.cross_entropy_loss
  """
  num = -torch.log(torch.sum(torch.exp(preds) * labels, axis=1))
  denom = torch.log(torch.sum(torch.exp(preds), axis=1))
  losses = num + denom
  return torch.mean(losses)