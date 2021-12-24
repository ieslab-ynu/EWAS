import torch.nn as nn
import torch

class UMask(nn.Module):
    '''
    Unit Mask
    with mask shape (c,h,w) same as input tensor t (batch,c,h,w)
    '''
    def __init__(self,t,num_classes):
        super(UMask, self).__init__()
        N,C,H,W = t.size()
        # nn.Conv2d(in_channels=t.size(1),out_channels=t.size(1),kernel_size=3,padding=1)
        # self.conv = nn.Conv2d(in_channels=C,out_channels=C,kernel_size=1)

        # self.aux_conv = nn.Conv2d(in_channels=C,out_channels=C,kernel_size=3)
        self.aux_linear = nn.Linear(C*H*W,num_classes)



    def forward(self,x,label=None):
        # mask_score = self.conv(x)
        # mask = torch.relu(mask_score)
        #fc_in = self.aux_conv(mask)
        N,C,H,W = x.size()
        fc_out = self.aux_linear(x.view(N,-1))
        if label is not None:
            mask = self.aux_linear.weight[label,:]
            out = x * mask.view(-1,C,H,W)
        else:
            pred_label = fc_out.max(1)[1]
            mask = self.aux_linear.weight[pred_label,:]
            out = x * mask.view(-1,C,H,W)
        return out ,fc_out
