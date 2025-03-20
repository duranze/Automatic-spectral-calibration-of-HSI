import os
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import dataset
import utils
from EvalMetrics import calc_psnr, calc_rmse, calc_ergas, calc_sam

def Trainer(opt):
    # Process multiple GPUs
    gpu_num = torch.cuda.device_count()
    print("Detected %d GPUs" % gpu_num)
    opt.batch_size *= gpu_num
    opt.num_workers *= gpu_num

    # Create folder for saving models
    utils.check_path(opt.save_path)
    cudnn.benchmark = opt.cudnn_benchmark

    # Define loss function
    criterion_L1 = torch.nn.L1Loss().cuda()

    # Initialize generator
    generator = utils.create_generator(opt)
    if opt.multi_gpu:
        generator = nn.DataParallel(generator).cuda()
    else:
        generator = generator.cuda()

    # Optimizer
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2),
                                   weight_decay=opt.weight_decay)

    # Learning rate adjustment function
    def adjust_learning_rate(opt, epoch, iteration, optimizer):
        if opt.lr_decrease_mode == 'epoch':
            lr = opt.lr * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        elif opt.lr_decrease_mode == 'iter':
            lr = opt.lr * (opt.lr_decrease_factor ** (iteration // opt.lr_decrease_iter))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    # Model saving function
    def save_model(opt, epoch, iteration, len_dataset, generator):
        if opt.save_mode == 'epoch':
            model_name = opt.method_name + '_epoch%d_bs%d.pth' % (epoch, opt.batch_size)
        else:
            model_name = opt.method_name + '_iter%d_bs%d.pth' % (iteration, opt.batch_size)
        save_model_path = os.path.join(opt.save_path, model_name)
        if opt.multi_gpu:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.module.state_dict(), save_model_path)
                    print('Model saved at epoch %d' % epoch)
            else:
                if iteration % opt.save_by_iter == 0:
                    torch.save(generator.module.state_dict(), save_model_path)
                    print('Model saved at iteration %d' % iteration)
        else:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.state_dict(), save_model_path)
                    print('Model saved at epoch %d' % epoch)
            else:
                if iteration % opt.save_by_iter == 0:
                    torch.save(generator.state_dict(), save_model_path)
                    print('Model saved at iteration %d' % iteration)

    # Dataset and DataLoader
    trainset = dataset.HS_multiscale_DSet(opt, 'train')
    print('Total training images:', len(trainset))
    dataloader = DataLoader(trainset, batch_size=opt.batch_size, shuffle=True,
                            num_workers=opt.num_workers, pin_memory=True)

    prev_time = time.time()

    for epoch in range(opt.epochs):
        for i, (img_A, img_B, _) in enumerate(dataloader):
            b, sb, c, h, w = img_A.size()
            img_A = img_A.view(b * sb, c, h, w).cuda()
            hs_ori = img_B.view(b * sb, c, h, w).cuda()

            optimizer_G.zero_grad()
            recon_B = generator(img_A)
            loss_pixel = criterion_L1(recon_B, hs_ori)
            loss_pixel.backward()
            optimizer_G.step()

            iters_done = epoch * len(dataloader) + i
            iters_left = opt.epochs * len(dataloader) - iters_done
            time_left = datetime.timedelta(seconds=iters_left * (time.time() - prev_time))
            prev_time = time.time()

            print("\r[Epoch %d/%d] [Batch %d/%d] [Rec Loss: %.4f] Time left: %s" %
                  (epoch + 1, opt.epochs, i, len(dataloader), loss_pixel.item(), time_left), end='')

            # Validate every specified epoch
            if ((epoch + 1) % opt.save_by_epoch == 0) and ((iters_done + 1) % len(dataloader) == 0):
                val(opt, generator, epoch)

            save_model(opt, epoch + 1, iters_done + 1, len(dataloader), generator)
            adjust_learning_rate(opt, epoch + 1, iters_done + 1, optimizer_G)

def val(opt, generator, epoch):
    test_dataset = dataset.HS_multiscale_DSet(opt, 'val')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False,
                                              num_workers=opt.num_workers, pin_memory=True)
    psnr_res, rmse_res, ergas_res, sam_res = 0.0, 0.0, 0.0, 0.0
    test_num = len(test_loader)
    for j, (img1, img_ori, path) in enumerate(test_loader):
        img1 = img1.cuda()
        img_ori = img_ori.cuda()
        print(opt.method_name, path[0])
        with torch.no_grad():
            out = generator(img1)
        # Convert to numpy with shape (H x W x C)
        img_ori_np = img_ori.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float64)
        out_np = out.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float64)
        psnr_res += calc_psnr(img_ori_np, out_np)
        rmse_res += calc_rmse(img_ori_np, out_np)
        ergas_res += calc_ergas(img_ori_np, out_np)
        sam_res += calc_sam(img_ori_np, out_np)

    psnr_res /= test_num
    rmse_res /= test_num
    ergas_res /= test_num
    sam_res /= test_num

    valstr = "epoch " + str(epoch) + '\t'
    valstr += "rmse:" + str(rmse_res) + '\t'
    valstr += "psnr:" + str(psnr_res) + '\t'
    valstr += "ergas:" + str(ergas_res) + '\t'
    valstr += "sam:" + str(sam_res)
    write_result(valstr)

def write_result(res):
    model_dir = "./val_results/"
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "val_result.txt"), 'a+') as f:
        f.write(res + '\n')
