import argparse
import os
import torch
import numpy as np
import cv2
import hdf5storage as hdf5

import utils
import dataset
from EvalMetrics import computeMRAE,computeRMSE,computeMAE,calc_psnr, calc_rmse, calc_ergas, calc_sam

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
    load_net_name_list.append('./track1/G_epoch80_bs2.pth')
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
    def gen_recv_error_map(pred_hs,img_hs,img_rgb,img_name,epo_num,save_path):
        # import pdb; pdb.set_trace()
        
        difference = np.abs(img_hs - pred_hs) / img_hs
        difference = np.mean(difference,axis=2)
        diff_map = np.uint8(difference * 255)
        diff_map = cv2.applyColorMap(diff_map,cv2.COLORMAP_JET)
        final_img = diff_map * 0.9 + img_rgb
        final_path = save_path+epo_num
        utils.check_path(final_path)
        final_path = os.path.join(final_path,'recv_'+img_name+'.png')
        cv2.imwrite(final_path,final_img)

    import json
    # gt_rgb_root = '/amax/home/dzr/dzr/dzr/autoCalib/rgb_track/gt_files/'
    for i, name in enumerate(load_net_name_list):
        # Initialize
        epoch_num = name.split('_')[1]
        print(epoch_num)
        # name = './checkpoints/HRN/track1/'+name
        generator = utils.create_generator_val1(opt,name).cuda()
        val_dataset = dataset.HS_multiscale_DSet(opt,'val')
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.test_batch_size, shuffle=False,
                                                  num_workers=opt.num_workers, pin_memory=True)
        # sample_folder = os.path.join(opt.val_path, opt.task_name, str(i))
        # utils.check_path(sample_folder)
        # sample_folder = '/amax/storage/nfs/dzr/dzr/flower-msi-sel/hscnn_10k_pred/pred_val'
        # sample_folder = '/amax/storage/nfs/dzr/dzr/flower-msi-sel/hscnn_10k_pred/pred_train'
        print('Network name: %s; The %d-th iteration' % (name, i))
        val_vis_dict = {}
        val_nir_dict = {}
        val_whole_dice = {}
        dict_list = [val_whole_dice,val_vis_dict,val_nir_dict]
        recv_list = ['whole','vis','nir']
        for j,(hsi_sc, hsi_gt,img_name) in enumerate(val_loader):            # To device
            hsi_sc = hsi_sc.cuda()
            img_name = img_name[0]
            print(j, img_name)
            # imgid = img_name[:-4]
            img_gtid = img_name.split("_")[1]
            # gt_rgb_path = gt_rgb_root + 'gtRef_'+img_gtid+'.png'
            # image_gt = cv2.imread(gt_rgb_path,-1)
            # Forward propagation
            with torch.no_grad():
                out = generator(hsi_sc)  # [0:480, 0:512, :], [1, 31, 480, 512]
                # import pdb;pdb.set_trace()
            # Save
            hsi_gt = hsi_gt.clone().data.permute(0, 2, 3, 1).cpu().numpy()[0, :, :, :].astype(np.float64)
            out = out.clone().data.permute(0, 2, 3, 1).cpu().numpy()[0, :, :, :].astype(np.float64)
            hsi_gt_vis = hsi_gt[:,:,:102]
            out_vis = out[:,:,:102]
            hsi_gt_nir = hsi_gt[:,:,102:]
            out_nir = out[:,:,102:]
            gt_list = [hsi_gt,hsi_gt_vis,hsi_gt_nir]
            out_list = [out,out_vis,out_nir]
            for r in range(3):
                psnr = calc_psnr(gt_list[r],out_list[r])
                rmse = calc_rmse(gt_list[r],out_list[r])
                ergas = calc_ergas(gt_list[r],out_list[r])
                sam = calc_sam(gt_list[r],out_list[r])
                dict_list[r][img_name] = {'psnr':psnr,'rmse':rmse,'ergas':ergas,'sam':sam}
                # val_vis_dict[img_name] = {'psnr':psnr,'rmse':rmse,'ergas':ergas,'sam':sam}
                # val_nir_dict[img_name] = {'psnr':psnr,'rmse':rmse,'ergas':ergas,'sam':sam}
                # if r==0:
                #     gen_recv_error_map(out_list[r],gt_list[r],image_gt,img_name,epoch_num,'./visual-val/'+recv_list[r]+'/')
        
        for m in range(3):
            # calulate 10 group results:
            dict_m = dict_list[m]
            overall_dict = {}
            groups = ['rw','bl','gr','re','ye','pu','sd','cd','rd','ev','sh']
            for g in groups:
                overall_dict[g] = {'psnr':0.0,'rmse':0.0,'ergas':0.0,'sam':0.0} 
            length = len(dict_m.keys())
            for fn in dict_m.keys():
                group = fn[:2]
                overall_dict[group]['psnr'] += dict_m[fn]['psnr']
                overall_dict[group]['rmse'] += dict_m[fn]['rmse']
                overall_dict[group]['ergas'] += dict_m[fn]['ergas']
                overall_dict[group]['sam'] += dict_m[fn]['sam']
            for group in groups:
                overall_dict[group]['psnr'] = overall_dict[group]['psnr'] / length *11
                overall_dict[group]['rmse'] = overall_dict[group]['rmse'] / length *11
                overall_dict[group]['ergas'] = overall_dict[group]['ergas'] / length *11
                overall_dict[group]['sam'] = overall_dict[group]['sam'] / length *11
            dict_m['overall'] = overall_dict
            with open('val_'+recv_list[m]+'_'+epoch_num+'.json', 'w') as json_file:
                json.dump(dict_m, json_file, indent=4)
            # gen_recv_error_map(out_vis,hsi_gt_vis,image_gt,img_name,epoch_num,'./visual-val/vis/')
            # gen_recv_error_map(out_nir,hsi_gt_nir,image_gt,img_name,epoch_num,'./visual-val/nir/')
        # with open('val_vis_'+epoch_num+'.json', 'w') as json_file:
        #     json.dump(val_vis_dict, json_file, indent=4)
        # with open('val_nir_'+epoch_num+'.json', 'w') as json_file:
        #     json.dump(val_nir_dict, json_file, indent=4)

