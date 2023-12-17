import torch
import numpy as np
from network.fusion import vnn_rgb_of_highQ, vnn_fusion_highQ
import cv2
from joblib import Parallel, delayed
torch.backends.cudnn.benchmark = True

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

def CenterCrop(frame, size):
    h, w = np.shape(frame)[0:2]
    th, tw = size
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))

    frame = frame[y1:y1 + th, x1:x1 + tw, :]
    return np.array(frame).astype(np.uint8)


def center_crop(frame):
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    with open('./dataloaders/ucf_labels.txt', 'r') as f:
        class_names = f.readlines()
        f.close()
    # init model
    model_RGB = vnn_rgb_of_highQ.VNN(num_classes=101)
    model_OF = vnn_rgb_of_highQ.VNN(num_classes=101, num_ch=2)
    model_fuse = vnn_fusion_highQ.VNN_F(num_classes=101, num_ch=192)
    checkpoint = torch.load('/Users/sid.roheda/Downloads/VNN_Code/models/VNN_Fusion-ucf101_epoch-99.pth.tar', map_location=lambda storage, loc: storage)
    model_RGB.load_state_dict(checkpoint['state_dict_rgb'])
    model_OF.load_state_dict(checkpoint['state_dict_of'])
    model_fuse.load_state_dict(checkpoint['state_dict_fuse'])
    model_RGB.to(device)
    model_OF.to(device)
    model_fuse.to(device)
    
    model_RGB.eval()
    model_OF.eval()
    model_fuse.eval()

    # read video
    # video = '/Users/sid.roheda/Downloads/VNN_Code/UCF-101/IceDancing/v_IceDancing_g05_c01.avi'
    video = '/Users/sid.roheda/Downloads/VNN_Code/UCF-101/TennisSwing/v_TennisSwing_g02_c06.avi'
    cap = cv2.VideoCapture(video)
    retaining = True

    clip = []
    result = []
    while retaining:
        retaining, frame = cap.read()
        if not retaining and frame is None:
            continue
        tmp_ = center_crop(cv2.resize(frame, (171, 128)))
        tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])
        clip.append(tmp)
        if len(clip) == 16:
            inputs = np.array(clip).astype(np.float32)
            inputs = np.expand_dims(inputs, axis=0)
            inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
            inputs = torch.from_numpy(inputs)
            inputs_of = compute_optical_flow(inputs, 112, 112)
            inputs = torch.autograd.Variable(inputs, requires_grad=False).to(device)
            inputs_of = torch.autograd.Variable(inputs_of, requires_grad=False).to(device)
            with torch.no_grad():
                outputs_rgb = model_RGB(inputs)
                outputs_of = model_OF(inputs_of)
                outputs = model_fuse(torch.cat((outputs_rgb, outputs_of), 1))

            probs = torch.nn.Softmax(dim=1)(outputs)
            label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]

            clip = []

        if "label" in locals(): 

            cv2.putText(frame, class_names[label].split(' ')[-1].strip(), (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 1)
            cv2.putText(frame, "prob: %.4f" % probs[0][label], (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 1)

            # result.append(frame)
            # clip.pop(0)

        cv2.imshow('result', frame)
        cv2.waitKey(100)

    # for i, frame in enumerate(result):
    #     if i < len(result)-1:
    #         cv2.imshow('result', frame)
    #         cv2.waitKey(60)
    #     else:
    #         cv2.imshow('result', frame)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()