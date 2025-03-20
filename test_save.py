import argparse
import os
import torch
import numpy as np
import utils
import dataset
from EvalMetrics import calc_psnr, calc_rmse, calc_ergas, calc_sam

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pre_train', type=bool, default=True)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--val_path', type=str, default='./val')
    parser.add_argument('--test_path', type=str, default='./test')
    parser.add_argument('--method_name', type=str, default='sit', help='Method name')
    parser.add_argument('--pad', type=str, default='reflect')
    parser.add_argument('--activ', type=str, default='lrelu')
    parser.add_argument('--norm', type=str, default='none')
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--in_channels', type=int, default=204)
    parser.add_argument('--out_channels', type=int, default=204)
    parser.add_argument('--start_channels', type=int, default=64)
    parser.add_argument('--init_type', type=str, default='xavier')
    parser.add_argument('--init_gain', type=float, default=0.02)
    parser.add_argument('--baseroot', type=str, default='/path/to/data')
    opt = parser.parse_args()

    method_name = opt.method_name.lower()
    def write_result(res):
        model_dir = './test_results_eval/'
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        save_name = method_name + ".txt"
        with open(os.path.join(model_dir, save_name), 'a+') as f:
            f.write(res + '\n')
    
    load_net_name_list = ['./track1/G_epoch1200_bs4.pth']
    for i, name in enumerate(load_net_name_list):
        # Extract epoch number from the filename
        epoch_num = name.split('_')[1]
        print("Loaded epoch:", epoch_num)
        model_save_dir = os.path.join('./test_output/', method_name)
        if not os.path.isdir(model_save_dir):
            os.makedirs(model_save_dir, exist_ok=True)
        generator = utils.create_generator_val1(opt, name).cuda()
        demo_dataset = dataset.HS_multiscale_DSet(opt, 'test')
        demo_loader = torch.utils.data.DataLoader(demo_dataset, batch_size=opt.test_batch_size, shuffle=False,
                                                    num_workers=opt.num_workers, pin_memory=True)
        print(f'Model: {name}; Test iteration: {i}')
        psnr, rmse, ergas, sam = 0.0, 0.0, 0.0, 0.0
        psnr_v, rmse_v, ergas_v, sam_v = 0.0, 0.0, 0.0, 0.0
        psnr_o, rmse_o, ergas_o, sam_o = 0.0, 0.0, 0.0, 0.0
        for j, (hsi_sc, hsi_gt, img_name) in enumerate(demo_loader):
            hsi_sc = hsi_sc.cuda()
            img_name = img_name[0]
            print(j, img_name)
            with torch.no_grad():
                out = generator(hsi_sc)
            hsi_gt_np = hsi_gt.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float64)
            out_np = out.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float64)
            psnr += calc_psnr(hsi_gt_np, out_np)
            rmse += calc_rmse(hsi_gt_np, out_np)
            ergas += calc_ergas(hsi_gt_np, out_np)
            sam += calc_sam(hsi_gt_np, out_np)
            psnr_v += calc_psnr(hsi_gt_np[:, :, :102], out_np[:, :, :102])
            rmse_v += calc_rmse(hsi_gt_np[:, :, :102], out_np[:, :, :102])
            ergas_v += calc_ergas(hsi_gt_np[:, :, :102], out_np[:, :, :102])
            sam_v += calc_sam(hsi_gt_np[:, :, :102], out_np[:, :, :102])
            psnr_o += calc_psnr(hsi_gt_np[:, :, 103:], out_np[:, :, 103:])
            rmse_o += calc_rmse(hsi_gt_np[:, :, 103:], out_np[:, :, 103:])
            ergas_o += calc_ergas(hsi_gt_np[:, :, 103:], out_np[:, :, 103:])
            sam_o += calc_sam(hsi_gt_np[:, :, 103:], out_np[:, :, 103:])
        length = len(demo_loader)
        psnr /= length
        rmse /= length
        ergas /= length
        sam /= length
        psnr_v /= length
        rmse_v /= length
        ergas_v /= length
        sam_v /= length
        psnr_o /= length
        rmse_o /= length
        ergas_o /= length
        sam_o /= length
        valstr = f'psnr:{psnr}\tsam:{sam}\trmse:{rmse}\tergas:{ergas}\n'
        valstr += f'psnr_v:{psnr_v}\tsam_v:{sam_v}\trmse_v:{rmse_v}\tergas_v:{ergas_v}\n'
        valstr += f'psnr_o:{psnr_o}\tsam_o:{sam_o}\trmse_o:{rmse_o}\tergas_o:{ergas_o}\n'
        write_result(valstr)
