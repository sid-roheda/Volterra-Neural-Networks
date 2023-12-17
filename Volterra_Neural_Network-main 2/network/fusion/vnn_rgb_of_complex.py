import torch
import torch.nn as nn
# from mypath import Path

class VNN(nn.Module):
    def __init__(self, num_classes = 400, num_ch = 3, pretrained=False):
        super(VNN, self).__init__()
        Q1 = 4
        nch_out1_5 = 8; nch_out1_3 = 8; nch_out1_1 = 8;
        sum_chans = nch_out1_5+nch_out1_3+nch_out1_1

        self.conv11_5 = nn.Conv3d(num_ch, nch_out1_5, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv11_3 = nn.Conv3d(num_ch, nch_out1_3, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv11_1 = nn.Conv3d(num_ch, nch_out1_1, kernel_size=(1, 1, 1), padding=(0, 0, 0))

        self.bn11 = nn.BatchNorm3d(sum_chans)

        self.conv21_5 = nn.Conv3d(num_ch, 2*Q1*nch_out1_5, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv21_3 = nn.Conv3d(num_ch, 2*Q1*nch_out1_3, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv21_1 = nn.Conv3d(num_ch, 2*Q1*nch_out1_1, kernel_size=(1, 1, 1), padding=(0, 0, 0))

        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.bn21 = nn.BatchNorm3d(sum_chans)

        
        Q2 = 4
        nch_out2 = 32
        self.conv12 = nn.Conv3d(nch_out1_5+nch_out1_3+nch_out1_1, nch_out2, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn12 = nn.BatchNorm3d(nch_out2)
        self.conv22 = nn.Conv3d(nch_out1_5+nch_out1_3+nch_out1_1, 2*Q2*nch_out2, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.bn22 = nn.BatchNorm3d(nch_out2)

        Q3 = 4
        nch_out3 = 64
        self.conv13 = nn.Conv3d(nch_out2, nch_out3, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn13 = nn.BatchNorm3d(nch_out3)
        self.conv23 = nn.Conv3d(nch_out2, 2*Q3*nch_out3, kernel_size=(3, 3, 3), padding=(1, 1, 1))
#         self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.bn23 = nn.BatchNorm3d(nch_out3)


        Q4 = 4
        nch_out4 = 96
        self.conv14 = nn.Conv3d(nch_out3, nch_out4, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn14 = nn.BatchNorm3d(nch_out4)
        self.conv24 = nn.Conv3d(nch_out3, 2*Q4*nch_out4, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.bn24 = nn.BatchNorm3d(nch_out4)
        
        Q6 = 4
        nch_out6 = 128
        self.conv16 = nn.Conv3d(nch_out4, nch_out6, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn16 = nn.BatchNorm3d(nch_out6)
        self.conv26 = nn.Conv3d(nch_out4, 2*Q4*nch_out6, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool6 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.bn26 = nn.BatchNorm3d(nch_out6)
        
        Q7 = 4
        nch_out7 = 256
        self.conv17 = nn.Conv3d(nch_out6, nch_out7, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn17 = nn.BatchNorm3d(nch_out7)
        self.conv27 = nn.Conv3d(nch_out6, 2*Q4*nch_out7, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool7 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.bn27 = nn.BatchNorm3d(nch_out7)
        
        Q5 = 2
        nch_out5 = 256
        self.conv15 = nn.Conv3d(nch_out7, nch_out5, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn15 = nn.BatchNorm3d(nch_out5)
        self.conv25 = nn.Conv3d(nch_out7, 2*Q5*nch_out5, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.bn25 = nn.BatchNorm3d(nch_out5)

#         self.fc6 = nn.Linear(100352, 4096)
#         self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(12544, num_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

        self.__init_weight()

        # if pretrained:
        #     self.__load_pretrained_weights()


    def forward(self, x, activation = False):
        
        Q1=4
        nch_out1_5 = 8; nch_out1_3 = 8; nch_out1_1 = 8;

        x11_5 = self.conv11_5(x) 
        x11_3 = self.conv11_3(x) 
        x11_1 = self.conv11_1(x) 

        x11 = torch.cat((x11_5, x11_3, x11_1), 1)

        # print('CONCAT SHAPE: ', x11.shape)

        x11 = self.bn11(x11)
        
        x21_5 = self.conv21_5(x)
        
        x21_5mul = torch.mul(x21_5[:,0:Q1*nch_out1_5,:,:,:],x21_5[:,Q1*nch_out1_5:2*Q1*nch_out1_5,:,:,:])
        x21_5add = torch.zeros_like(x11_5)
        for q in range(Q1):
            x21_5add = torch.add(x21_5add, x21_5mul[:,(q*nch_out1_5):((q*nch_out1_5)+(nch_out1_5)),:,:,:])

        x21_3 = self.conv21_3(x)
        
        x21_3mul = torch.mul(x21_3[:,0:Q1*nch_out1_3,:,:,:],x21_3[:,Q1*nch_out1_3:2*Q1*nch_out1_3,:,:,:])
        x21_3add = torch.zeros_like(x11_3)
        for q in range(Q1):
            x21_3add = torch.add(x21_3add, x21_3mul[:,(q*nch_out1_3):((q*nch_out1_3)+(nch_out1_3)),:,:,:])

        x21_1 = self.conv21_1(x)
        
        x21_1mul = torch.mul(x21_1[:,0:Q1*nch_out1_1,:,:,:],x21_1[:,Q1*nch_out1_1:2*Q1*nch_out1_1,:,:,:])
        x21_1add = torch.zeros_like(x11_1)
        for q in range(Q1):
            x21_1add = torch.add(x21_1add, x21_1mul[:,(q*nch_out1_1):((q*nch_out1_1)+(nch_out1_1)),:,:,:])
        

        x21_add = torch.cat((x21_5add, x21_3add, x21_1add), 1)

        x21_add = self.bn21(x21_add)

        x = self.pool1(torch.add(x11, x21_add))
#         x = torch.add(x11, x21_add)

        # print('x: ', x.shape) 

        Q2=4
        nch_out2 = 32 

        x12 = self.conv12(x)
        x12 = self.bn12(x12)
        # print('x11: ', x11.shape)
        x22 = self.conv22(x)
        # print('x21: ', x21.shape)
        x22_mul = torch.mul(x22[:,0:Q2*nch_out2,:,:,:],x22[:,Q2*nch_out2:2*Q2*nch_out2,:,:,:])
        x22_add = torch.zeros_like(x12)
        for q in range(Q2):
            x22_add = torch.add(x22_add, x22_mul[:,(q*nch_out2):((q*nch_out2)+(nch_out2)),:,:,:])
        x22_add = self.bn22(x22_add)
        # x = torch.add(x12, x22_add)
        x = self.pool2(torch.add(x12, x22_add))

        # print('x: ', x.shape) 

        Q3=4
        nch_out3 = 64

        x13 = self.conv13(x)
        x13 = self.bn13(x13)
        # print('x11: ', x11.shape)
        x23 = self.conv23(x)
        # print('x21: ', x21.shape)
        x23_mul = torch.mul(x23[:,0:Q3*nch_out3,:,:,:],x23[:,Q3*nch_out3:2*Q3*nch_out3,:,:,:])
        x23_add = torch.zeros_like(x13)
        for q in range(Q3):
            x23_add = torch.add(x23_add, x23_mul[:,(q*nch_out3):((q*nch_out3)+(nch_out3)),:,:,:])
        x23_add = self.bn23(x23_add)
#         x = self.pool3(torch.add(x13, x23_add))
        x = torch.add(x13, x23_add)

        # # print('x: ', x.shape) 

        Q4=4
        nch_out4 = 96

        x14 = self.conv14(x)
        x14 = self.bn14(x14)
        # print('x11: ', x11.shape)
        x24 = self.conv24(x)
        # print('x21: ', x21.shape)
        x24_mul = torch.mul(x24[:,0:Q4*nch_out4,:,:,:],x24[:,Q4*nch_out4:2*Q4*nch_out4,:,:,:])
        x24_add = torch.zeros_like(x14)
        for q in range(Q4):
            x24_add = torch.add(x24_add, x24_mul[:,(q*nch_out4):((q*nch_out4)+(nch_out4)),:,:,:])
        x24_add = self.bn24(x24_add)
#         x = self.pool4(torch.add(x14, x24_add))
        x = torch.add(x14, x24_add)
    
        Q6=4
        nch_out6 = 128

        x16 = self.conv16(x)
        x16 = self.bn16(x16)
        # print('x11: ', x11.shape)
        x26 = self.conv26(x)
        # print('x21: ', x21.shape)
        x26_mul = torch.mul(x26[:,0:Q6*nch_out6,:,:,:],x26[:,Q6*nch_out6:2*Q6*nch_out6,:,:,:])
        x26_add = torch.zeros_like(x16)
        for q in range(Q6):
            x26_add = torch.add(x26_add, x26_mul[:,(q*nch_out6):((q*nch_out6)+(nch_out6)),:,:,:])
        x26_add = self.bn26(x26_add)
        x = self.pool6(torch.add(x16, x26_add))
#         x = torch.add(x16, x26_add)
    
        Q7=4
        nch_out7 = 256

        x17 = self.conv17(x)
        x17 = self.bn17(x17)
        # print('x11: ', x11.shape)
        x27 = self.conv27(x)
        # print('x21: ', x21.shape)
        x27_mul = torch.mul(x27[:,0:Q7*nch_out7,:,:,:],x27[:,Q7*nch_out7:2*Q7*nch_out7,:,:,:])
        x27_add = torch.zeros_like(x17)
        for q in range(Q7):
            x27_add = torch.add(x27_add, x27_mul[:,(q*nch_out7):((q*nch_out7)+(nch_out7)),:,:,:])
        x27_add = self.bn27(x27_add)
#         x = self.pool7(torch.add(x17, x27_add))
        x = torch.add(x17, x27_add)

        # print('x: ', x.shape) 

        Q5 = 2
        nch_out5 = 256
        x15 = self.conv15(x)
        x15 = self.bn15(x15)
        # print('x11: ', x11.shape)
        x25 = self.conv25(x)
        # print('x21: ', x21.shape)
        x25_mul = torch.mul(x25[:,0:Q5*nch_out5,:,:,:],x25[:,Q5*nch_out5:2*Q5*nch_out5,:,:,:])
        x25_add = torch.zeros_like(x15)
        for q in range(Q5):
            x25_add = torch.add(x25_add, x25_mul[:,(q*nch_out5):((q*nch_out5)+(nch_out5)),:,:,:])
        x25_add = self.bn25(x25_add)
        x = self.pool5(torch.add(x15, x25_add))
        # x = torch.add(x15, x25_add)




        print('x: ', x.shape) 

        x = x.view(-1, 12544)
#         x = self.relu(self.fc6(x))
#         x = self.dropout(x)
#         x = self.relu(self.fc7(x))
#         x = self.dropout(x)

        logits = self.fc8(x)

        return x
 
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
    b = [model.conv11_5, model.conv11_3, model.conv11_1, model.bn11, model.conv21_5, model.conv21_3, model.conv21_1, model.bn21, model.conv12, model.bn12, model.conv22, model.bn22, model.conv13, model.bn13, model.conv23, model.bn23, model.conv14, model.bn14, model.conv24, model.bn24] #, model.fc6, model.fc7]
    
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k  

# def get_10x_lr_params(model):
#     """
#     This generator returns all the parameters for the last fc layer of the net.
#     """
#     b = [model.fc8]
#     for j in range(len(b)):
#         for k in b[j].parameters():
#             if k.requires_grad:
#                 yield k