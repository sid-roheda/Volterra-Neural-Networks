import torch
import torch.nn as nn
from mypath import Path

class VNN_F(nn.Module):
    def __init__(self, num_classes, num_ch = 3, pretrained=False):
        super(VNN_F, self).__init__()
#         Q0 = 2
#         nch_out0 = 96 
#         self.conv10 = nn.Conv3d(num_ch, nch_out0, kernel_size=(1, 1, 1), padding=(0, 0, 0))
#         self.bn10 = nn.BatchNorm3d(nch_out0)
#         self.conv20 = nn.Conv3d(num_ch, 2*Q0*nch_out0, kernel_size=(1, 1, 1), padding=(0, 0, 0))
# #         self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
#         self.bn20 = nn.BatchNorm3d(nch_out0)

        Q1 = 2
        nch_out1 = 256 
        self.conv11 = nn.Conv3d(num_ch, nch_out1, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn11 = nn.BatchNorm3d(nch_out1)
        self.conv21 = nn.Conv3d(num_ch, 2*Q1*nch_out1, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.bn21 = nn.BatchNorm3d(nch_out1)

#         Q1_red = 2
#         nch_out1_red = 96
#         self.conv11_red = nn.Conv3d(nch_out1, nch_out1_red, kernel_size=(1, 1, 1), padding=(0, 0, 0))
#         self.bn11_red = nn.BatchNorm3d(nch_out1_red)
#         self.conv21_red = nn.Conv3d(nch_out1, 2*Q1_red*nch_out1_red, kernel_size=(1, 1, 1), padding=(0, 0, 0))
# #         self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
#         self.bn21_red = nn.BatchNorm3d(nch_out1_red)

#         Q2 = 2
#         nch_out2 = 512 
#         self.conv12 = nn.Conv3d(nch_out1_red, nch_out2, kernel_size=(3, 3, 3), padding=(1, 1, 1))
#         self.bn12 = nn.BatchNorm3d(nch_out2)
#         self.conv22 = nn.Conv3d(nch_out1_red, 2*Q2*nch_out2, kernel_size=(3, 3, 3), padding=(1, 1, 1))
#         self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
#         self.bn22 = nn.BatchNorm3d(nch_out2)
        
        self.fc8 = nn.Linear(12544, num_classes)

        self.dropout = nn.Dropout(p=0.5)

        

        self.__init_weight()
        
    def forward(self, x, activation = False):
        # Q0=2
        # nch_out0 = 96
        # x10 = self.conv10(x)
        # x10 = self.bn10(x10)

        # x20 = self.conv20(x)

        # x20_mul = torch.mul(x20[:,0:Q0*nch_out0,:,:,:],x20[:,Q0*nch_out0:2*Q0*nch_out0,:,:,:])
        # x20_add = torch.zeros_like(x10)
        # for q in range(Q0):
        #     x20_add = torch.add(x20_add, x20_mul[:,(q*nch_out0):((q*nch_out0)+(nch_out0)),:,:,:])
        # x20_add = self.bn20(x20_add)
        # x = torch.add(x10, x20_add)

        Q1=2
        nch_out1 = 256

        x11 = self.conv11(x)
        x11 = self.bn11(x11)
        
        x21 = self.conv21(x)
        
 
        x21_mul = torch.mul(x21[:,0:Q1*nch_out1,:,:,:],x21[:,Q1*nch_out1:2*Q1*nch_out1,:,:,:])
        x21_add = torch.zeros_like(x11)
        for q in range(Q1):
            x21_add = torch.add(x21_add, x21_mul[:,(q*nch_out1):((q*nch_out1)+(nch_out1)),:,:,:])
        x21_add = self.bn21(x21_add)
        x = self.pool1(torch.add(x11, x21_add))

        # Q1_red=2
        # nch_out1_red = 96
        
        # x11_red = self.conv11_red(x)
        # x11_red = self.bn11_red(x11_red)
        
        # x21_red = self.conv21_red(x)
        
 
        # x21_red_mul = torch.mul(x21_red[:,0:Q1_red*nch_out1_red,:,:,:],x21_red[:,Q1_red*nch_out1_red:2*Q1_red*nch_out1_red,:,:,:])
        # x21_red_add = torch.zeros_like(x11_red)
        # for q in range(Q1_red):
        #     x21_red_add = torch.add(x21_red_add, x21_red_mul[:,(q*nch_out1_red):((q*nch_out1_red)+(nch_out1_red)),:,:,:])
        # x21_red_add = self.bn21_red(x21_red_add)
        # x = torch.add(x11_red, x21_red_add)

        # Q2=2
        # nch_out2 = 512

        # x12 = self.conv12(x)
        # x12 = self.bn12(x12)
        
        # x22 = self.conv22(x)
        
 
        # x22_mul = torch.mul(x22[:,0:Q2*nch_out2,:,:,:],x22[:,Q2*nch_out2:2*Q2*nch_out2,:,:,:])
        # x22_add = torch.zeros_like(x12)
        # for q in range(Q2):
        #     x22_add = torch.add(x22_add, x22_mul[:,(q*nch_out2):((q*nch_out2)+(nch_out2)),:,:,:])
        # x22_add = self.bn22(x22_add)
        # x = self.pool2(torch.add(x12, x22_add))
        
        
        
#         print(x.shape)
        x = x.view(-1, 12544)
        
        
        x = self.dropout(x)
     
        logits = self.fc8(x)

        return logits
    
    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
def get_1x_lr_params(model):
    """
    This generator returns all the parameters for conv and two fc layers of the net.
    """
    b = [model.conv11, model.bn11, model.conv21, model.bn21] #, model.conv11_red, model.bn11_red, model.conv21_red, model.bn21_red, model.conv11, model.bn12, model.conv22, model.bn22] # model.conv10, model.bn10, model.conv20, model.bn20, 
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k  

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last fc layer of the net.
    """
    b = [model.fc8]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k



if __name__ == "__main__":
    inputs = torch.rand(1, 3, 16, 112, 112)
    net = C3D(num_classes=101, pretrained=True)

    outputs = net.forward(inputs)
    print(outputs.size())