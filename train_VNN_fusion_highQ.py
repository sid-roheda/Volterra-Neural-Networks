import timeit
from datetime import datetime
import socket
import os
import glob
from tqdm import tqdm
import cv2
import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np

from dataloaders.dataset import VideoDataset
from network.fusion import vnn_rgb_of_highQ, vnn_fusion_highQ
from joblib import Parallel, delayed

 
def flow(X, Ht, Wd, of_skip=1, polar=False):
    
    X_of = np.zeros([int(X.shape[0]/of_skip), Ht, Wd, 2])
    
    of_ctr=-1
    for j in range(0,X.shape[0]-of_skip,of_skip):
        of_ctr+=1
        flow =  cv2.normalize(cv2.calcOpticalFlowFarneback(cv2.cvtColor(np.array(X[j+of_skip,:,:,:], dtype=np.uint8), cv2.COLOR_BGR2GRAY), cv2.cvtColor(np.array(X[j,:,:,:], dtype=np.uint8), cv2.COLOR_BGR2GRAY),None, 0.5, 3, 15, 3, 5, 1.2, 0), None, alpha=0, beta=1,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#             flow =  cv2.calcOpticalFlowFarneback(cv2.cvtColor(np.array(X[i,j+of_skip,:,:,:], dtype=np.uint8), cv2.COLOR_BGR2GRAY), cv2.cvtColor(np.array(X[i,j,:,:,:], dtype=np.uint8), cv2.COLOR_BGR2GRAY),None, 0.5, 3, 15, 3, 5, 1.2, 0)
        if polar:
            mag, ang = cv2.cartToPolar(flow[:,:,0], flow[:,:,1])
            X_of[of_ctr,:,:,:] = np.concatenate([np.expand_dims(mag, axis=2), np.expand_dims(ang, axis=2)], axis=2)
        else:
            X_of[of_ctr,:,:,:] = flow
    # print('X_of_Shape: ', X_of.shape)

    return X_of

def compute_optical_flow(X, Ht, Wd, num_proc = 4, of_skip = 1, polar = False):
    # print('here')
    X = (X.permute(0,2,3,4,1)).detach().cpu().numpy()
    optical_flow = Parallel(n_jobs=num_proc)(delayed(flow)(X[i], Ht, Wd, of_skip, polar) for i in range(X.shape[0]))
    X_of = torch.tensor(np.asarray(optical_flow)).float()
    # print(X_of.shape)
    return X_of.permute(0,4,1,2,3)


# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

nEpochs = 100  # Number of epochs for training
resume_epoch = 0  # Default is 0, change if want to resume
useTest = True # See evolution of the test set when training
nTestInterval = 10 # Run on test set every nTestInterval epochs
snapshot = 5 # Store a model every snapshot epochs
lr = 1e-4 # Learning rate

dataset = 'ucf101' # Options: hmdb51 or ucf101

if dataset == 'hmdb51':
    num_classes=51
elif dataset == 'ucf101':
    num_classes = 101
else:
    print('We only implemented hmdb and ucf datasets.')
    raise NotImplementedError

save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]

if resume_epoch != 0:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) if runs else 0
else:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))
modelName = 'VNN_Fusion' # Options: C3D or R2Plus1D or R3D
saveName = modelName + '-' + dataset

def train_model(dataset=dataset, save_dir=save_dir, num_classes=num_classes, lr=lr,
                num_epochs=nEpochs, save_epoch=snapshot, useTest=useTest, test_interval=nTestInterval):
    """
        Args:
            num_classes (int): Number of classes in the data
            num_epochs (int, optional): Number of epochs to train for.
    """

    if modelName == 'C3D':
        model = C3D_model.C3D(num_classes=num_classes, pretrained=True)
        train_params = [{'params': C3D_model.get_1x_lr_params(model), 'lr': lr},
                        {'params': C3D_model.get_10x_lr_params(model), 'lr': lr * 10}]
    
    elif modelName == 'VNN':
        model = VNN_model.VNN(num_classes=num_classes, pretrained=False)
        train_params = [{'params': VNN_model.get_1x_lr_params(model), 'lr': lr},
                        {'params': VNN_model.get_10x_lr_params(model), 'lr': lr * 10}]
    elif modelName == 'VNN_2021':
        model = VNN_model_2021.VNN(num_classes=num_classes, pretrained=False)
        train_params = [{'params': VNN_model_2021.get_1x_lr_params(model), 'lr': lr},
                        {'params': VNN_model_2021.get_10x_lr_params(model), 'lr': lr * 10}]
    elif modelName == 'VNN_Fusion':
        model_RGB = vnn_rgb_of_highQ.VNN(num_classes=num_classes, num_ch=3, pretrained=False)
        model_OF = vnn_rgb_of_highQ.VNN(num_classes=num_classes, num_ch=2, pretrained=False)
        model_fuse = vnn_fusion_highQ.VNN_F(num_classes=num_classes, num_ch=192, pretrained=False)
        train_params = [{'params': vnn_rgb_of_highQ.get_1x_lr_params(model_RGB), 'lr': lr}, {'params': vnn_rgb_of_highQ.get_1x_lr_params(model_OF), 'lr': lr}, {'params': vnn_fusion_highQ.get_1x_lr_params(model_fuse), 'lr': lr},
                        {'params': vnn_fusion_highQ.get_10x_lr_params(model_fuse), 'lr': lr}]
    elif modelName == 'R2Plus1D':
        model = R2Plus1D_model.R2Plus1DClassifier(num_classes=num_classes, layer_sizes=(2, 2, 2, 2))
        train_params = [{'params': R2Plus1D_model.get_1x_lr_params(model), 'lr': lr},
                        {'params': R2Plus1D_model.get_10x_lr_params(model), 'lr': lr * 10}]
    elif modelName == 'R3D':
        model = R3D_model.R3DClassifier(num_classes=num_classes, layer_sizes=(2, 2, 2, 2))
        train_params = model.parameters()
    else:
        print('We only implemented C3D, R2Plus1D, and VNN models.')
        raise NotImplementedError
    criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
#     optimizer = optim.SGD(train_params, lr=lr, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.Adam(train_params, lr=lr, weight_decay=5e-4)    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5,
                                          gamma=0.9)  # the scheduler divides the lr by 10 every 10 epochs

    if resume_epoch == 0:
        print("Training {} from scratch...".format(modelName))
    else:
        checkpoint = torch.load(os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar'),
                       map_location=lambda storage, loc: storage)   # Load all tensors onto the CPU
        print("Initializing weights from: {}...".format(
            os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar')))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])

    print('Total params: %.2fM' % ((sum(p.numel() for p in model_RGB.parameters()) + sum(p.numel() for p in model_OF.parameters()) + sum(p.numel() for p in model_fuse.parameters())) / 1000000.0))
    model_RGB.to(device)
    model_OF.to(device)
    model_fuse.to(device)
    criterion.to(device)

    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    print('Training model on {} dataset...'.format(dataset))
    train_dataloader = DataLoader(VideoDataset(dataset=dataset, split='train',clip_len=16), batch_size=16, shuffle=True, num_workers=4) #batch_size=16
    val_dataloader   = DataLoader(VideoDataset(dataset=dataset, split='val',  clip_len=16), batch_size=16, num_workers=4) #batch_size=16
    test_dataloader  = DataLoader(VideoDataset(dataset=dataset, split='test', clip_len=16), batch_size=16, num_workers=4) #batch_size=16

    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}
    test_size = len(test_dataloader.dataset)

    for epoch in range(resume_epoch, num_epochs):
        # each epoch has a training and validation step
        for phase in ['train', 'val']:
            start_time = timeit.default_timer()

            # reset the running loss and corrects
            running_loss = 0.0
            running_corrects = 0.0

            # set model to train() or eval() mode depending on whether it is trained
            # or being validated. Primarily affects layers such as BatchNorm or Dropout.
            if phase == 'train':
                # scheduler.step() is to be called once every epoch during training
                scheduler.step()
                model_RGB.train()
                model_OF.train()
                model_fuse.train()
            else:
                model_RGB.eval()
                model_OF.eval()
                model_fuse.eval()

            for inputs, labels in tqdm(trainval_loaders[phase]):
                # move inputs and labels to the device the training is taking place on
                
                inputs_of = compute_optical_flow(inputs, 112, 112)
                
#                 inputs = torch.cat((inputs, inputs_of), 1)
#                 print('Input_Shape: ', inputs.shape)
                inputs = Variable(inputs, requires_grad=True).to(device)
                inputs_of = Variable(inputs_of, requires_grad=True).to(device)
                labels = Variable(labels).to(device)
                optimizer.zero_grad()

                if phase == 'train':
                    outputs_rgb = model_RGB(inputs)
                    outputs_of = model_OF(inputs_of)
                    outputs = model_fuse(torch.cat((outputs_rgb, outputs_of), 1))
                else:
                    with torch.no_grad():
                        outputs_rgb = model_RGB(inputs)
                        outputs_of = model_OF(inputs_of)
                        outputs = model_fuse(torch.cat((outputs_rgb, outputs_of), 1))

                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / trainval_sizes[phase]
            epoch_acc = running_corrects.double() / trainval_sizes[phase]

            if phase == 'train':
                writer.add_scalar('data/train_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/train_acc_epoch', epoch_acc, epoch)
            else:
                writer.add_scalar('data/val_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/val_acc_epoch', epoch_acc, epoch)

            print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format(phase, epoch+1, nEpochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

        if epoch % save_epoch == (save_epoch - 1):
            torch.save({
                'epoch': epoch + 1,
                'state_dict_rgb': model_RGB.state_dict(),
                'state_dict_of': model_OF.state_dict(),
                'state_dict_fuse': model_fuse.state_dict(),
                'opt_dict': optimizer.state_dict(),
            }, os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar'))
            print("Save model at {}\n".format(os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar')))

        if useTest and epoch % test_interval == (test_interval - 1):
            model_RGB.eval()
            model_OF.eval()
            model_fuse.eval()
            start_time = timeit.default_timer()

            running_loss = 0.0
            running_corrects = 0.0

            for inputs, labels in tqdm(test_dataloader):
                inputs_of = compute_optical_flow(inputs, 112, 112)
#                 inputs = torch.cat((inputs, inputs_of), 1)
                inputs = inputs.to(device)
                inputs_of = inputs_of.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    outputs_rgb = model_RGB(inputs)
                    outputs_of = model_OF(inputs_of)
                    outputs = model_fuse(torch.cat((outputs_rgb, outputs_of), 1))
                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / test_size
            epoch_acc = running_corrects.double() / test_size

            writer.add_scalar('data/test_loss_epoch', epoch_loss, epoch)
            writer.add_scalar('data/test_acc_epoch', epoch_acc, epoch)

            print("[test] Epoch: {}/{} Loss: {} Acc: {}".format(epoch+1, nEpochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

    writer.close()


if __name__ == "__main__":
    train_model()