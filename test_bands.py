import argparse
import os
import torch
import numpy as np
import cv2
import hdf5storage as hdf5

import utils
import dataset
from EvalMetrics import computeMRAE,computeRMSE,computeMAE,calc_psnr, calc_rmse, calc_ergas, calc_sam
from SpectralUtils import savePNG
from scipy.interpolate import interp1d
from scipy.io import loadmat,savemat
if __name__ == "__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # Pre-train, saving, and loading parameters
    parser.add_argument('--pre_train', type=bool, default=True, help='pre_train or not')
    parser.add_argument('--test_batch_size', type=int, default=1, help='size of the testing batches for single GPU')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='number of cpu threads to use during batch generation')
    parser.add_argument('--val_path', type=str, default='./val', help='saving path that is a folder')
    parser.add_argument('--test_path', type=str, default='./test', help='saving path that is a folder')
    parser.add_argument('--task_name', type=str, default='track1',
                        help='task name for loading networks, saving, and log')
    # Network initialization parameters
    parser.add_argument('--pad', type=str, default='reflect', help='pad type of networks')
    parser.add_argument('--activ', type=str, default='lrelu', help='activation type of networks')
    parser.add_argument('--norm', type=str, default='none', help='normalization type of networks')
    parser.add_argument('--crop_size', type=int, default=512, help='input channels for generator')
    parser.add_argument('--in_channels', type=int, default=204, help='input channels for generator')
    parser.add_argument('--out_channels', type=int, default=204, help='output channels for generator')
    parser.add_argument('--start_channels', type=int, default=64, help='start channels for generator')
    parser.add_argument('--init_type', type=str, default='xavier', help='initialization type of generator')
    parser.add_argument('--init_gain', type=float, default=0.02, help='initialization gain of generator')
    # Dataset parameters
    parser.add_argument('--baseroot', type=str, default='/amax/home/dzr/dzr/dzr/autoCalib/hsi_f16/', help='baseroot')
    # NTIRE2020_Validation_Clean    NTIRE2020_Validation_RealWorld
    opt = parser.parse_args()

    # ----------------------------------------
    #                   Test
    # ----------------------------------------
    method_name='sit'
    import os
    def write_result(res):
        model_dir = '/amax/home/dzr/dzr/dzr/autoCalib/Mine/visual-paper/test_hsi_real/test_eval/'
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        save_name = method_name+".txt"
        with open(os.path.join(model_dir, save_name), 'a+') as f:
            # f.write(str(datetime.datetime.now()))
            # f.write('\r\n')
            f.write(res)
            f.write('\r\n')
    load_net_name_list = []
    # load_net_name_list = os.listdir('./checkpoints/HRN/track1/')
    load_net_name_list.append('./track1/G_epoch1200_bs4.pth')
    import json
    for i, name in enumerate(load_net_name_list):
        # Initialize
        epoch_num = name.split('_')[1]
        print(epoch_num)
        # name = './checkpoints/HRN/track1/'+name
        model_save_dir = '/amax/home/dzr/dzr/dzr/autoCalib/Mine/visual-paper/test_hsi_real/'+method_name+'/'
        if not os.path.isdir(model_save_dir):
            os.makedirs(model_save_dir, exist_ok=True)
        generator = utils.create_generator_val1(opt,name).cuda()
        demo_dataset = dataset.HS_multiscale_DSet(opt,'test')
        demo_loader = torch.utils.data.DataLoader(demo_dataset, batch_size=opt.test_batch_size, shuffle=False,
                                                  num_workers=opt.num_workers, pin_memory=True)
        # sample_folder = os.path.join(opt.val_path, opt.task_name, str(i))
        # utils.check_path(sample_folder)
        # sample_folder = '/amax/storage/nfs/dzr/dzr/flower-msi-sel/hscnn_10k_pred/pred_val'
        # sample_folder = '/amax/storage/nfs/dzr/dzr/flower-msi-sel/hscnn_10k_pred/pred_train'
        print('Network name: %s; The %d-th iteration' % (name, i))
        val_txt = ''
        # psnr,rmse,ergas,sam = 0.0,0.0,0.0,0.0
        # psnr_v,rmse_v,ergas_v,sam_v = 0.0,0.0,0.0,0.0
        # psnr_o,rmse_o,ergas_o,sam_o = 0.0,0.0,0.0,0.0
        psnr_array = np.zeros(204)
        for j,(hsi_sc, hsi_gt,img_name) in enumerate(demo_loader):            # To device
            hsi_sc = hsi_sc.cuda()
            img_name = img_name[0]
            print(j, img_name)
            # imgid = img_name[:-4]
            # img_gtid = img_name.split("_")[1]
            # gt_rgb_path = gt_rgb_root + 'gtRef_'+img_gtid+'.png'
            # image_gt = cv2.imread(gt_rgb_path,-1)
            # Forward propagation
            with torch.no_grad():
                out = generator(hsi_sc)  # [0:480, 0:512, :], [1, 31, 480, 512]
                # import pdb;pdb.set_trace()
            # Save
            hsi_gt = hsi_gt.clone().data.permute(0, 2, 3, 1).cpu().numpy()[0, :, :, :].astype(np.float64)
            out = out.clone().data.permute(0, 2, 3, 1).cpu().numpy()[0, :, :, :].astype(np.float64)
            # np.save(model_save_dir+'pred_'+img_name+'.npy',out.astype(np.float16))
            for b in range(204):
                psnr_array[b] += calc_psnr(hsi_gt[:,:,b],out[:,:,b])
            
            # sam += calc_sam(hsi_gt,out)
            # rmse += calc_rmse(hsi_gt,out)
            # ergas += calc_ergas(hsi_gt,out)
            # psnr_v += calc_psnr(hsi_gt[:,:,:102],out[:,:,:102])
            # rmse_v += calc_rmse(hsi_gt[:,:,:102],out[:,:,:102])
            # ergas_v += calc_ergas(hsi_gt[:,:,:102],out[:,:,:102])
            # sam_v += calc_sam(hsi_gt[:,:,:102],out[:,:,:102])
            # psnr_o += calc_psnr(hsi_gt[:,:,103:],out[:,:,103:])
            # rmse_o += calc_rmse(hsi_gt[:,:,103:],out[:,:,103:])
            # ergas_o += calc_ergas(hsi_gt[:,:,103:],out[:,:,103:])
            # sam_o += calc_sam(hsi_gt[:,:,103:],out[:,:,103:])
        
        length = len(demo_loader)
        psnr_array = psnr_array / length
        np.save('/amax/home/dzr/dzr/dzr/autoCalib/Mine/visual-paper/hsi_real/bands_test/'+method_name+'.npy',psnr_array)
        # psnr = psnr / length
        # rmse = rmse / length
        # ergas = ergas / length
        # sam = sam / length
        # psnr_v = psnr_v / length
        # rmse_v = rmse_v / length
        # ergas_v = ergas_v / length
        # sam_v = sam_v / length
        # psnr_o = psnr_o / length
        # rmse_o = rmse_o / length
        # ergas_o = ergas_o / length
        # sam_o = sam_o / length
        # valstr = 'psnr:'+str(psnr) +'\t sam:'+str(sam)+'\t rmse:' + str(rmse)+'\t ergas:'+str(ergas)+'\n'
        # valstr = valstr + "psnr_v:" + str(psnr_v) + '\t'
        # valstr = valstr + "sam_v:" + str(sam_v) + '\t'
        # valstr = valstr + "rmse_v:" + str(rmse_v) + '\t'
        # valstr = valstr + "ergas_v:" + str(ergas_v) + '\n'
        # valstr = valstr + "psnr_o:" + str(psnr_o) + '\t'
        # valstr = valstr + "sam_o:" + str(sam_o) + '\t'
        # valstr = valstr + "rmse_o:" + str(rmse_o) + '\t'
        # valstr = valstr + "ergas_o:" + str(ergas_o) + '\n'
        # write_result(valstr)