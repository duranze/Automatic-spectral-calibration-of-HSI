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
    import os
    load_net_name_list = []
    # load_net_name_list = os.listdir('./checkpoints/HRN/track1/')
    load_net_name_list.append('./track1/G_epoch40_bs2.pth')
    # def write_result(res,method_name='hcaNet',split='val'):
    #     model_dir = split+"_"+method_name
    #     if not os.path.isdir(model_dir):
    #         os.makedirs(model_dir, exist_ok=True)
    #     save_name = split+"_"+method_name+".txt"
    #     with open(os.path.join(model_dir, save_name), 'a+') as f:
    #         # f.write(str(datetime.datetime.now()))
    #         # f.write('\r\n')
    #         f.write(res)
    #         f.write('\r\n')
    def hyper_resample(datacube):
        """
        Resample a hyperspectral data cube to new specified wavelengths.
        
        Args:
        datacube (np.array): The input hyperspectral data cube with shape (height, width, num_bands).
        wavelens (np.array): The original wavelengths corresponding to the third dimension of the datacube.
        samp_waves (np.array): The target wavelengths for resampling.
        
        Returns:
        np.array: The resampled hyperspectral data cube.
        """
        # Initialize the output data cube with the new shape
        samp_waves = np.linspace(400, 700, 31)
        wavelens = np.array([
        397.32,
        400.20,
        403.09,
        405.97,
        408.85,
        411.74,
        414.63,
        417.52,
        420.40,
        423.29,
        426.19,
        429.08,
        431.97,
        434.87,
        437.76,
        440.66,
        443.56,
        446.45,
        449.35,
        452.25,
        455.16,
        458.06,
        460.96,
        463.87,
        466.77,
        469.68,
        472.59,
        475.50,
        478.41,
        481.32,
        484.23,
        487.14,
        490.06,
        492.97,
        495.89,
        498.80,
        501.72,
        504.64,
        507.56,
        510.48,
        513.40,
        516.33,
        519.25,
        522.18,
        525.10,
        528.03,
        530.96,
        533.89,
        536.82,
        539.75,
        542.68,
        545.62,
        548.55,
        551.49,
        554.43,
        557.36,
        560.30,
        563.24,
        566.18,
        569.12,
        572.07,
        575.01,
        577.96,
        580.90,
        583.85,
        586.80,
        589.75,
        592.70,
        595.65,
        598.60,
        601.55,
        604.51,
        607.46,
        610.42,
        613.38,
        616.34,
        619.30,
        622.26,
        625.22,
        628.18,
        631.15,
        634.11,
        637.08,
        640.04,
        643.01,
        645.98,
        648.95,
        651.92,
        654.89,
        657.87,
        660.84,
        663.81,
        666.79,
        669.77,
        672.75,
        675.73,
        678.71,
        681.69,
        684.67,
        687.65,
        690.64,
        693.62,
        696.61,
        699.60,
        702.58,
        705.57,
        708.57,
        711.56,
        714.55,
        717.54,
        720.54,
        723.53,
        726.53,
        729.53,
        732.53,
        735.53,
        738.53,
        741.53,
        744.53,
        747.54,
        750.54,
        753.55,
        756.56,
        759.56,
        762.57,
        765.58,
        768.60,
        771.61,
        774.62,
        777.64,
        780.65,
        783.67,
        786.68,
        789.70,
        792.72,
        795.74,
        798.77,
        801.79,
        804.81,
        807.84,
        810.86,
        813.89,
        816.92,
        819.95,
        822.98,
        826.01,
        829.04,
        832.07,
        835.11,
        838.14,
        841.18,
        844.22,
        847.25,
        850.29,
        853.33,
        856.37,
        859.42,
        862.46,
        865.50,
        868.55,
        871.60,
        874.64,
        877.69,
        880.74,
        883.79,
        886.84,
        889.90,
        892.95,
        896.01,
        899.06,
        902.12,
        905.18,
        908.24,
        911.30,
        914.36,
        917.42,
        920.48,
        923.55,
        926.61,
        929.68,
        932.74,
        935.81,
        938.88,
        941.95,
        945.02,
        948.10,
        951.17,
        954.24,
        957.32,
        960.40,
        963.47,
        966.55,
        969.63,
        972.71,
        975.79,
        978.88,
        981.96,
        985.05,
        988.13,
        991.22,
        994.31,
        997.40,
    1000.49,
    1003.58
    ])
        output_cube = np.zeros((datacube.shape[0], datacube.shape[1], len(samp_waves)))
        
        # Iterate through each pixel in the spatial dimensions
        for i in range(datacube.shape[0]):
            for j in range(datacube.shape[1]):
                # Extract the spectrum at this pixel
                spectrum = datacube[i, j, :]
                
                # Create an interpolation function for the current spectrum
                interp_func = interp1d(wavelens, spectrum, kind='linear', bounds_error=False, fill_value="extrapolate")
                
                # Use the interpolation function to resample the spectrum
                resampled_spectrum = interp_func(samp_waves)
                
                # Place the resampled spectrum in the output data cube
                output_cube[i, j, :] = resampled_spectrum
        cameraResponse = np.load('/amax/home/dzr/dzr/dzr/autoCalib/data_process/cie_1964_w_gain.npz')['filters']
        rgbIm = np.dot(output_cube, cameraResponse)/255.0 *5.0
        rgbIm[rgbIm<0] = 0
        rgbIm[rgbIm>1] = 1
        # rgbIm = np.true_divide(rgbIm,256)
        return rgbIm
    
    def gen_recv_error_map(pred_hs,img_hs,img_rgb,img_name,epo_num,save_path):
        # import pdb; pdb.set_trace()
        
        difference = np.abs(img_hs - pred_hs) / img_hs
        difference = np.mean(difference,axis=2)
        diff_map = np.uint8(difference * 255)
        diff_map = cv2.applyColorMap(diff_map,cv2.COLORMAP_JET)
        final_img = diff_map * 0.9 + img_rgb
        final_path = save_path+epo_num
        tmp_path = final_path
        utils.check_path(final_path)
        final_path = os.path.join(final_path,'recv_'+img_name+'.png')
        cv2.imwrite(final_path,final_img)
        hyper_resample(pred_hs)
        cameraResponse = np.load('/amax/home/dzr/dzr/dzr/autoCalib/data_process/cie_1964_w_gain.npz')['filters']
        rgb_pred = hyper_resample(pred_hs)
        rgb_gt = hyper_resample(img_hs)
        savePNG(rgb_pred,tmp_path+'/'+'pred_'+img_name+'.png')
        savePNG(rgb_gt,tmp_path+'/'+'gt_'+img_name+'.png')

    import json
    gt_rgb_root = '/amax/home/dzr/dzr/dzr/autoCalib/rgb_track/gt_files/'
    for i, name in enumerate(load_net_name_list):
        # Initialize
        epoch_num = name.split('_')[1]
        print(epoch_num)
        # name = './checkpoints/HRN/track1/'+name
        generator = utils.create_generator_val1(opt,name).cuda()
        demo_dataset = dataset.HS_multiscale_DSet(opt,'demo')
        demo_loader = torch.utils.data.DataLoader(demo_dataset, batch_size=opt.test_batch_size, shuffle=False,
                                                  num_workers=opt.num_workers, pin_memory=True)
        # sample_folder = os.path.join(opt.val_path, opt.task_name, str(i))
        # utils.check_path(sample_folder)
        # sample_folder = '/amax/storage/nfs/dzr/dzr/flower-msi-sel/hscnn_10k_pred/pred_val'
        # sample_folder = '/amax/storage/nfs/dzr/dzr/flower-msi-sel/hscnn_10k_pred/pred_train'
        print('Network name: %s; The %d-th iteration' % (name, i))
        demo_vis_dict = {}
        demo_nir_dict = {}
        demo_whole_dice = {}
        dict_list = [demo_whole_dice,demo_vis_dict,demo_nir_dict]
        # recv_list = ['whole','vis','nir']
        recv_list = ['whole']
        for j,(hsi_sc, hsi_gt,img_name) in enumerate(demo_loader):            # To device
            hsi_sc = hsi_sc.cuda()
            img_name = img_name[0]
            print(j, img_name)
            # imgid = img_name[:-4]
            img_gtid = img_name.split("_")[1]
            gt_rgb_path = gt_rgb_root + 'gtRef_'+img_gtid+'.png'
            image_gt = cv2.imread(gt_rgb_path,-1)
            # Forward propagation
            with torch.no_grad():
                out = generator(hsi_sc)  # [0:480, 0:512, :], [1, 31, 480, 512]
                # import pdb;pdb.set_trace()
            # Save
            hsi_gt = hsi_gt.clone().data.permute(0, 2, 3, 1).cpu().numpy()[0, :, :, :].astype(np.float64)
            out = out.clone().data.permute(0, 2, 3, 1).cpu().numpy()[0, :, :, :].astype(np.float64)
            if img_name[:2] == 'cd':
                np.save('./pred_'+img_name+'.npy',out.astype(np.float16))
            # hsi_gt_vis = hsi_gt[:,:,:102]
            # out_vis = out[:,:,:102]
            # hsi_gt_nir = hsi_gt[:,:,102:]
            # out_nir = out[:,:,102:]
            # gt_list = [hsi_gt]
            # out_list = [out]
            # r = 0
            # psnr = calc_psnr(gt_list[r],out_list[r])
            # rmse = calc_rmse(gt_list[r],out_list[r])
            # ergas = calc_ergas(gt_list[r],out_list[r])
            # sam = calc_sam(gt_list[r],out_list[r])
            # dict_list[r][img_name] = {'psnr':psnr,'rmse':rmse,'ergas':ergas,'sam':sam}
            # # val_vis_dict[img_name] = {'psnr':psnr,'rmse':rmse,'ergas':ergas,'sam':sam}
            # # val_nir_dict[img_name] = {'psnr':psnr,'rmse':rmse,'ergas':ergas,'sam':sam}
            # gen_recv_error_map(out_list[r],gt_list[r],image_gt,img_name,epoch_num,'./visual-demo/'+img_name+'/')
        
        # m=0
        # # for m in range(3):
        #     # calulate 10 group results:
        # dict_m = dict_list[m]
        # overall_dict = {}
        # groups = ['bl','gr','re','ye','pu','sd','cd','rd','ev','sh']
        # for g in groups:
        #     overall_dict[g] = {'psnr':0.0,'rmse':0.0,'ergas':0.0,'sam':0.0} 
        # length = len(dict_m.keys())
        # for fn in dict_m.keys():
        #     group = fn[:2]
        #     overall_dict[group]['psnr'] += dict_m[fn]['psnr']
        #     overall_dict[group]['rmse'] += dict_m[fn]['rmse']
        #     overall_dict[group]['ergas'] += dict_m[fn]['ergas']
        #     overall_dict[group]['sam'] += dict_m[fn]['sam']
        # for group in groups:
        #     overall_dict[group]['psnr'] = overall_dict[group]['psnr'] / length *10
        #     overall_dict[group]['rmse'] = overall_dict[group]['rmse'] / length *10
        #     overall_dict[group]['ergas'] = overall_dict[group]['ergas'] / length *10
        #     overall_dict[group]['sam'] = overall_dict[group]['sam'] / length *10
        # dict_m['overall'] = overall_dict
        # with open('val_'+recv_list[m]+'_'+epoch_num+'.json', 'w') as json_file:
        #     json.dump(dict_m, json_file, indent=4)
        # gen_recv_error_map(out_vis,hsi_gt_vis,image_gt,img_name,epoch_num,'./visual-val/vis/')
        # gen_recv_error_map(out_nir,hsi_gt_nir,image_gt,img_name,epoch_num,'./visual-val/nir/')
    # with open('val_vis_'+epoch_num+'.json', 'w') as json_file:
    #     json.dump(val_vis_dict, json_file, indent=4)
    # with open('val_nir_'+epoch_num+'.json', 'w') as json_file:
    #     json.dump(val_nir_dict, json_file, indent=4)

