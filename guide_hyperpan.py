
import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from guided_diffusion import utils
import torch.optim as optim
from pathlib import Path
import torch.nn.functional as F
from metrics import rmse1
from model1 import *
import cv2
import torch.utils.data as datamodel
from models.cnn import *


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    


class MyLoss(torch.nn.Module): 
    def __init__(self,PSF,R,device):
        super(MyLoss, self).__init__()
        self.R=R
        self.PSF=PSF
        self.device=device
    def forward(self, output, target,MSI,sf=4):
        coeff = torch.unsqueeze(self.PSF, 0)
        coeff = torch.unsqueeze(coeff, 0)
        coeff = torch.repeat_interleave(coeff, output.shape[1],0)
        _,c, h, w= output.shape
        w1,h1=self.PSF.shape
        outs = functional.conv2d(output, coeff.to(self.device), bias=None, stride=sf, padding=int((w1-1)/2),  groups=c)
        target_HSI = functional.conv2d(target, coeff.to(self.device), bias=None, stride=sf, padding=int((w1-1)/2),  groups=c)
        RTZ = torch.tensordot(output, self.R, dims=([1], [1])) 
        RTZ=torch.Tensor.permute(RTZ,(0,3,1,2))
        MSILoss=torch.mean(torch.abs(RTZ-MSI))
        tragetloss=torch.mean(torch.abs(output-target))
        HSILoss=torch.mean(torch.abs(outs[:,:,1:-1,1:-1]-target_HSI[:,:,1:-1,1:-1]))
        loss_total=tragetloss+0.1*HSILoss+MSILoss
        return loss_total








if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--baseconfig', type=str, default='base.json',
                        help='JSON file for creating model and diffusion')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default="0")
    parser.add_argument('-sr', '--savedir', type=str, default='./results')   # where to save the restored images
    parser.add_argument('-eta1', '--eta1', type=float, default=2)            # parameter eta_1
    parser.add_argument('-eta2', '--eta2', type=float, default=2)            # parameter eta_2
    parser.add_argument('-rank', '--rank', type=int, default=4)              # subspace dimension; low rank parameter s
    parser.add_argument('-seed', '--seed', type=int, default=0)          
    parser.add_argument('-dn', '--dataname', type=str, default="Chikusei_test")   
    parser.add_argument('-step', '--step', type=int, default=500)            # Original total sampling step (divisible by accstep)
    parser.add_argument('-accstep', '--accstep', type=int, default=500)      # Actual sampling step (less than step)
    parser.add_argument('-krtype', '--krtype', type=int, default=0)          # how to get the kernel and srf: '0' for estimate, '1' for download
    parser.add_argument('-sn', '--samplenum', type=int, default=1)  
    parser.add_argument('-scale', '--scale', type=int, default=4)            # downsampling scale
    parser.add_argument('-ks', '--ks', type=int, default=5)                 # kernel size
    parser.add_argument('-res', '--res', type=str, default="no")             # how to set residual: 'no' for no residual, 'opt' for estimating residual
    parser.add_argument('-sample_method', '--sample_method', type=str, default='ddpm')

    parser.add_argument('-rs', '--resume_state', type=str, default='./dm_weight/I190000_E97')  # where you put the loaded diffusion model

    ## parse configs
    args = parser.parse_args()

    opt = utils.parse(args)

    device = torch.device("cuda")
    print('device is {}'.format(device))
    # exit()

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # Buliding model
    print('===> Building model')


    # 加载数据集
    class HSIDataset:
        """自定义数据集类"""

        def __init__(self, train_hrhs_all, train_hrms_all, train_lrhs_all):
            self.train_hrhs_all = train_hrhs_all
            self.train_hrms_all = train_hrms_all
            self.train_lrhs_all = train_lrhs_all

        def __getitem__(self, index):
            train_hrhs = self.train_hrhs_all[index, :, :, :]
            train_hrms = self.train_hrms_all[index, :, :, :]
            train_lrhs = self.train_lrhs_all[index, :, :, :]
            return train_hrhs, train_hrms, train_lrhs

        def __len__(self):
            return self.train_hrhs_all.shape[0]
        
    downsample_factor=4
    PSF = fspecial('gaussian', 5, 3)
    # 读取数据
    training_size=48
    stride=1
    file_path = "./data/hypan_pavia_data.mat"
    data = sio.loadmat(file_path)
    HRHS = data['gt']
    HRHS = HRHS.transpose(2,0,1)

    Ch=93

    R=np.full((1,HRHS.shape[0]),1/Ch)

    MSI0 = np.tensordot(R,  HRHS, axes=([1], [0]))
    HSI0=Gaussian_downsample(HRHS,PSF,downsample_factor)
    Rr=3
    inters = int((Ch+1)/(Rr+1)) # interval
    selected_bands = [(t+1)*inters-1 for t in range(Rr)]
    Eband = torch.Tensor(selected_bands).type(torch.int).to(device)  

    AE_model = CNN(Ch,1).to(device)


    PSF_T=torch.Tensor(PSF).to(device)
    R_T = torch.Tensor(R).to(device)
    loss_zeroshot = MyLoss(PSF_T,R_T,device).to(device)

    optimizer_AE = optim.Adam(AE_model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_AE, 1000, gamma=0.8, last_epoch=-1)


    augument=[0]
    HSI_aug=[]
    HSI_aug.append(HSI0)
    MSI_aug=[]
    MSI_aug.append(MSI0)
    train_hrhs = []
    train_hrms = []
    train_lrhs= []
    for j in augument:       
        HSI = cv2.flip(HSI0, j)
        HSI_aug.append(HSI)
    for j in range(len(HSI_aug)):
        HSI = HSI_aug[j]
        HSI_Abun=HSI
        HSI_LR_Abun=Gaussian_downsample(HSI_Abun,PSF,downsample_factor)
        MSI_LR=np.tensordot(R,  HSI, axes=([1], [0])) 
        for j in range(0, HSI_Abun.shape[1]-training_size+1, stride):
            for k in range(0, HSI_Abun.shape[2]-training_size+1, stride):
                
                temp_hrhs = HSI[:,j:j+training_size, k:k+training_size]
                temp_hrms = MSI_LR[:,j:j+training_size, k:k+training_size]
                temp_lrhs = HSI_LR_Abun[:,int(j/downsample_factor):int((j+training_size)/downsample_factor), int(k/downsample_factor):int((k+training_size)/downsample_factor)]


                train_hrhs.append(temp_hrhs)
                train_hrms.append(temp_hrms)
                train_lrhs.append(temp_lrhs)

    train_hrhs=torch.Tensor(train_hrhs)
    train_lrhs=torch.Tensor(train_lrhs)
    train_hrms=torch.Tensor(train_hrms)

    print(train_hrhs.shape, train_hrms.shape, train_lrhs.shape)

    #########
    batchsz=64
    print(batchsz)
    train_data=HSIDataset(train_hrhs,train_hrms,train_lrhs)
    train_loader = datamodel.DataLoader(dataset=train_data, batch_size=batchsz, shuffle=True)



    AE_model.eval()
    for epoch in range(2001):
        count = 0
        for step1, (a1, a2,a3) in enumerate(train_loader):


            im_out = AE_model(a3.to(device),a2.to(device))


            loss = loss_zeroshot(im_out.to(device),a1.to(device),a2.to(device))


            optimizer_AE.zero_grad()
            loss.backward(retain_graph=True)
            optimizer_AE.step()
            scheduler.step()
            count = count + 1


        if (epoch) % 50 == 0:
            print("!!! 第{}个Epoch的第{}轮 loss1 = {} " .format(epoch,count,loss))

    OUT_DIR = Path('./guide_weights/')
    # torch.save(AE_model.state_dict(), OUT_DIR.joinpath('Pavia_hyperpan.pth'))

