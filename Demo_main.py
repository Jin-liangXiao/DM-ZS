import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch as th
import torch.nn.functional as nF
from pathlib import Path
import torch.optim as optim
from guided_diffusion import utils
from guided_diffusion.create import create_model_and_diffusion_RS
from collections import OrderedDict
from metrics import *
from models.cnn import *
from models.myunet import skip
from models.mlp import Spe_net
import torch.backends.cudnn as cudnn
import random
import scipy.io as sio
import time



class SpeLoss(torch.nn.Module):
    def __init__(self,psf,HSI,device):
        super(SpeLoss, self).__init__()
        # self.psf = psf
        self.device=device
        self.weight= psf
        self.HSI= HSI
    def forward(self, output):
        outs = nF.conv2d(output, self.weight.to(self.device), padding=int((5 - 1)/2),  groups=(self.HSI).shape[1])
        outs=outs[:,:,::4, ::4]
        HSILoss=torch.norm(outs-HSI)
        return HSILoss
    
class SpaLoss(torch.nn.Module):
    def __init__(self,R,MSI,device):
        super(SpaLoss, self).__init__()
        self.R=R
        self.device=device
        self.MSI= MSI
    def forward(self, output):
        RTZ = torch.tensordot(output, self.R.to(self.device), dims=([1], [1])) 
        RTZ=torch.Tensor.permute(RTZ,(0,3,1,2))
        MSILoss=torch.norm(RTZ-self.MSI.to(self.device))
        return MSILoss


if __name__ == "__main__":


        # ================== Pre-Define =================== #
    SEED = 42
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # cudnn.benchmark = True  ###自动寻找最优算法
    torch.backends.cudnn.benchmark = False
    cudnn.deterministic = True

    torch.set_default_dtype(torch.float32)


    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--baseconfig', type=str, default='base.json',
                        help='JSON file for creating model and diffusion')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default="0")
    parser.add_argument('-sr', '--savedir', type=str, default='./results')   # where to save the restored images
    parser.add_argument('-rank', '--rank', type=int, default=3)              # subspace dimension; low rank parameter s
    parser.add_argument('-seed', '--seed', type=int, default=0)          
    parser.add_argument('-dn', '--dataname', type=str, default="Pavia_test")
    parser.add_argument('-step', '--step', type=int, default=100)            # Original total sampling step (divisible by accstep)
    parser.add_argument('-accstep', '--accstep', type=int, default=100)      # Actual sampling step (less than step)
    parser.add_argument('-scale', '--scale', type=int, default=4)            # downsampling scale
    parser.add_argument('-ks', '--ks', type=int, default=5)                 # kernel size
    parser.add_argument('-res', '--res', type=str, default="no")
    parser.add_argument('-sample_method', '--sample_method', type=str, default='ddpm')

    parser.add_argument('-rs', '--resume_state', type=str, default='./dm_weight/I190000_E97')

    ## parse configs
    args = parser.parse_args()

    opt = utils.parse(args)
    opt = utils.dict_to_nonedict(opt)
    opt['diffusion']['diffusion_steps'] = args.step
    opt['diffusion']['acce_steps'] = args.accstep
    
    device = th.device("cuda")
    dname = opt['dataname']
    
    ## create model and diffusion process
    model, diffusion = create_model_and_diffusion_RS(opt)



    print('===> Building model')

    Ch = 93
    AE_guide = CNN(Ch,1).to(device)
    ckpt = "./guide_weights/Pavia_hyperpan.pth"

    
    weight = torch.load(ckpt)
    AE_guide.load_state_dict(weight, strict=False)

    input_depth = Ch
    output_depth = Ch
    rank = 20 #adjust for better performance WDC, Chikusei:40
    tt = 1
    AE_model2 = Spe_net(64*64, rank).to(device)

    AE_model1 = skip(3, rank,
                num_channels_down=[128] * tt,
                num_channels_up=[128] * tt,
                num_channels_skip=[4] * tt,
                filter_size_up=3, filter_size_down=3, filter_skip_size=1,
                upsample_mode='bilinear',
                need_sigmoid=False, need_bias=True, act_fun='LeakyReLU').to(device)

    ## seed
    seeed = opt['seed']
    np.random.seed(seeed)
    th.manual_seed(seeed)
    th.cuda.manual_seed(seeed)

    ## Load diffusion model
    load_path = opt['resume_state']
    gen_path = '{}_gen.pth'.format(load_path)
    cks = th.load(gen_path)
    new_cks = OrderedDict()
    for k, v in cks.items():
        newkey = k[11:] if k.startswith('denoise_fn.') else k
        new_cks[newkey] = v
    model.load_state_dict(new_cks, strict=False)
    for param in model.parameters():
        param.requires_grad=False
    model.to(device)
    model.eval()
 
    ## params
    param = dict()
    param['scale'] = opt['scale'] # downsampling scale
    param['k_s'] = opt['ks']      # kernel size



    data2='pavia'
    downsample_factor=4
    file_path = "./data/hypan_pavia_data.mat"
    data = sio.loadmat(file_path)
    HRHS = data['gt']  # 64x64x4
    HRHS = HRHS.transpose(2,0,1)
    R = data['F']
    PSF = data['psf']


    PH = th.from_numpy(R).to(th.float32).to(device)
    param['PH'] = PH.to(device)


    MSI = np.tensordot(R,  HRHS, axes=([1], [0]))
    HRHS = th.from_numpy(HRHS)
    MSI = th.from_numpy(MSI)
    ms, Ch = HRHS.shape[1], HRHS.shape[0]
    HRHS = HRHS.unsqueeze(0).to(device) # [1, Ch, ms, ms]
    MSI = MSI.unsqueeze(0)
    kernel = th.from_numpy(PSF).repeat(Ch,1,1,1)
    kernel = kernel.to(th.float32)
    kernel = kernel.to(device)
    param['kernel'] = kernel # kernel
    from functools import partial
    blur = partial(nF.conv2d, weight=param['kernel'], padding=int((param['k_s'] - 1)/2), groups=Ch) 
    down = lambda x : x[:,:,::param['scale'], ::param['scale']]
    HSI=blur(HRHS.type(torch.float32))
    HSI=down(HSI)
    PSF_T=th.Tensor(PSF)



    loss_spe = SpeLoss(kernel, HSI.to(th.float32).to(device),device)
    R = th.from_numpy(R)
    loss_spa = SpaLoss(R.to(th.float32), MSI.to(th.float32).to(device),device)


    
    Rr = opt['rank']

    # select bands
    inters = int((Ch+1)/(Rr+1)) # interval
    selected_bands = [(t+1)*inters-1 for t in range(Rr)]
    param['Band'] = th.Tensor(selected_bands).type(th.int).to(device)  
    
    model_condition = {'LRHS': HSI.to(device), 'PAN': MSI.to(device)}

    out_path = Path(opt['savedir'])
    out_path.mkdir(parents=True, exist_ok=True)
    ###
    diff_time_1 = time.time()
    label = False
    print(HSI.shape)
    print(MSI.shape)
    sample,E,add_res = diffusion.sample_loop(
        model,
        AE_model1,
        AE_model2,
        AE_guide,
        label,
        (1, Ch, ms, ms),
        Rr = Rr,
        noise = None,
        clip_denoised=True,
        model_condition=model_condition,
        param=param,
        save_root=out_path,
        sample_method=args.sample_method,
        res = args.res  # opt, itp
    )   
    sample = (sample + 1)/2
    diff_time_2 = time.time()
    diff_time=diff_time_1-diff_time_2



    HRHS = HRHS.cpu().numpy().squeeze()


    p = []
    p = p + [x for x in AE_model1.parameters()]
    p = p + [x for x in AE_model2.parameters()]
    optimizer_AE = optim.Adam(p, lr=0.001)
    start_time = time.time()
    for epoch in range(20000):
        count = 0

        if epoch == 500:#this can be adjusted for more alternative iteration
            label = True
            sample,E,add_res = diffusion.sample_loop(
                model,
                AE_model1,
                AE_model2,
                AE_guide,
                label,
                (1, Ch, ms, ms),
                Rr = Rr,
                noise = None,
                clip_denoised=True,
                model_condition=model_condition,
                param=param,
                save_root=out_path,
                sample_method=args.sample_method,
                res = args.res  # opt, itp
            )
            
            sample = (sample + 1)/2

        net_input= model_condition["LRHS"].reshape(1,Ch,64*64)
        im_out1 = AE_model1(sample).squeeze()
        im_out2 = AE_model2(net_input.unsqueeze(0)).squeeze(0)
        im_out = th.tensordot(im_out2,im_out1, dims=([2], [0]))



        loss = loss_spa(im_out)+1*loss_spe(im_out)


        optimizer_AE.zero_grad()
        loss.backward(retain_graph=True)
        optimizer_AE.step()

        count = count + 1



        if (epoch) % 1000 == 0:
            out = im_out.detach().cpu().numpy().squeeze()
            HRHS0 = HRHS.squeeze()
            a, psnr_x = rmse1(np.clip(out, 0, 1), HRHS0)
            print("第{}个Epoch的第{}轮 loss1 = {}  psnr = {}" .format(epoch,count,loss,psnr_x))

    out = im_out.detach().cpu().numpy().squeeze()
    rmse_x, psnr = rmse1(np.clip(out, 0, 1), HRHS)
    end_time = time.time()
    Ours_time = end_time - start_time
    print(psnr)
    print(Ours_time)




