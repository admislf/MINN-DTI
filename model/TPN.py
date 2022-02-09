import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)
def conv5x5(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=5,
                     stride=stride, padding=2, bias=False)
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=False)
# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv5x5(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        if self.downsample:
            return self.elu(self.bn2(self.conv2(self.elu(self.bn1(self.conv1(x)))))+self.downsample(x))
        else:
            return self.elu(self.bn2(self.conv2(self.elu(self.bn1(self.conv1(x)))))+x)
class DrugVQA(torch.nn.Module):
    def __init__(self,d_a,r,in_channels,cnn_channels,cnn_layers):
        """
        Initializes parameters suggested in paper
 
        args:
            d_a            : {int} hidden dimension for the dense layer
            r              : {int} attention-hops or attention heads
            in_channels    : {int} channels of CNN block input
            cnn_channels   : {int} channels of CNN block
            cnn_layers     : {int} num of layers of each CNN block
            node_feat_size : {int} Size for the input node features.
            edge_feat_size : {int} Size for the input edge features.
            num_layers     : {int} Number of GNN layers. Default to 2.
            graph_feat_size: {int} Size for the graph representations to be computed. Default to 200.
        Returns:
            self
        """
        super(DrugVQA,self).__init__()
        #cnn
        self.block = ResidualBlock
        self.r = r
        self.in_channels = in_channels
        self.linear_first_seq = torch.nn.Linear(cnn_channels,d_a)
        self.linear_second_seq = torch.nn.Linear(d_a,self.r)
        self.conv = conv3x3(1, self.in_channels)
        self.bn = nn.BatchNorm2d(self.in_channels)
        self.elu = nn.ELU(inplace=False)
        self.layer1 = self.make_layer(cnn_channels, cnn_layers)
        self.layer2 = self.make_layer(cnn_channels, cnn_layers)
        self.linear_first_seq = torch.nn.Linear(cnn_channels,d_a)
    def softmax(self,input, axis=1):
        """
        Softmax applied to axis=n
        Args:
           input: {Tensor,Variable} input on which softmax is to be applied
           axis : {int} axis on which softmax is to be applied

        Returns:
            softmaxed tensors
        """
        input_size = input.size()
        input = input.transpose(axis, len(input_size)-1)
        trans_size = input.size()
        return F.softmax((input.contiguous().view(-1, trans_size[-1])),dim=1).view(*trans_size).transpose(axis, len(input_size)-1)

    def make_layer(self,out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(self.block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(self.block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self,x):
        x= torch.mean(self.layer2(self.layer1(self.elu(self.bn(self.conv(x))))),2).permute(0,2,1)
        seq_att = self.softmax(self.linear_second_seq(torch.tanh(self.linear_first_seq(x))),1).transpose(1,2)
        #print(1,seq_att.size())
        return seq_att@x,seq_att
        