import torch
import torch.nn as nn

class LabelSmoothingLoss(nn.BCEWithLogitsLoss):
    def __init__(self, nClass, ε=0.01):
        super().__init__()
        self.nClass = nClass
        self.ε      = ε

    def forward(self, mScore, vY):
        ε      = self.ε
        nClass = self.nClass
        mScore = mScore.log_softmax(dim=-1)

        with torch.no_grad():
            mSmoothY = torch.empty_like(mScore).fill_(ε / (nClass - 1))
            mSmoothY.scatter_(1, vY.unsqueeze(1), 1-ε)

        return torch.mean( torch.sum(-mSmoothY * mScore, dim=-1) )
        
class FloatConvertBCEWithLogitsLoss():
    def __init__(self, 
                 weight: Optional[Tensor] = None,
                 size_average=None, 
                 reduce=None, 
                 reduction: str = 'mean',
                 pos_weight: Optional[Tensor] = None) -> None:
        super(FloatConvertBCEWithLogitsLoss, self).__init__(
        weight,
        size_average, 
        reduce, 
        reduction, 
        pos_weight)
        
    def forward(self, input: Tensor, target: Tensor) -> Tensor: 
        return super(FloatConvertBCEWithLogitsLoss, self).forward(input, target.float())