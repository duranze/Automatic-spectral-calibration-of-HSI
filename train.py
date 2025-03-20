import argparse
import trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Saving and loading parameters
    parser.add_argument('--save_path', type=str, default='save', help='Directory to save the model')
    parser.add_argument('--save_val_folder', type=str, default='val_results', help='Folder to save validation results')
    parser.add_argument('--save_mode', type=str, default='epoch', help='Save mode: "epoch" or "iter"')
    parser.add_argument('--save_by_epoch', type=int, default=1, help='Save model every specified number of epochs')
    parser.add_argument('--save_by_iter', type=int, default=100000, help='Save model every specified number of iterations')
    parser.add_argument('--load_name', type=str, default='', help='Filename of the pre-trained model')
    # GPU parameters
    parser.add_argument('--multi_gpu', type=bool, default=True, help='Whether to use multiple GPUs')
    parser.add_argument('--gpu_ids', type=str, default='0, 1, 2, 3', help='IDs of GPUs to use')
    parser.add_argument('--cudnn_benchmark', type=bool, default=True, help='Enable cudnn benchmark')
    # Training parameters
    parser.add_argument('--epochs', type=int, default=2000, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--b1', type=float, default=0.5, help='Adam beta1 parameter')
    parser.add_argument('--b2', type=float, default=0.999, help='Adam beta2 parameter')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay for the optimizer')
    parser.add_argument('--lr_decrease_mode', type=str, default='epoch', help='Learning rate decrease mode: "epoch" or "iter"')
    parser.add_argument('--lr_decrease_epoch', type=int, default=200, help='Decrease learning rate every specified number of epochs')
    parser.add_argument('--lr_decrease_iter', type=int, default=50000, help='Decrease learning rate every specified number of iterations')
    parser.add_argument('--lr_decrease_factor', type=float, default=0.5, help='Learning rate decrease factor')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of worker threads for data loading')
    parser.add_argument('--lambda_half', type=float, default=0.5, help='Lambda parameter for SubNet')
    # Network initialization parameters
    parser.add_argument('--pad', type=str, default='reflect', help='Padding type used in the network')
    parser.add_argument('--activ', type=str, default='lrelu', help='Activation function type')
    parser.add_argument('--norm', type=str, default='none', help='Normalization type')
    parser.add_argument('--in_channels', type=int, default=204, help='Number of input channels for the generator')
    parser.add_argument('--out_channels', type=int, default=204, help='Number of output channels for the generator')
    parser.add_argument('--start_channels', type=int, default=64, help='Initial number of channels for the generator')
    parser.add_argument('--init_type', type=str, default='xavier', help='Initialization type for the generator weights')
    parser.add_argument('--init_gain', type=float, default=0.02, help='Initialization gain')
    # Dataset parameters (additional)
    parser.add_argument('--baseroot', type=str, default='./data', help='Root directory of the dataset')
    parser.add_argument('--crop_size', type=int, default=256, help='Crop size')
    parser.add_argument('--val_path', type=str, default='./test', help='Validation set path')
    parser.add_argument('--test_batch_size', type=int, default=1, help='Batch size for testing')
    # Method name parameter (replaces task_name)
    parser.add_argument('--method_name', type=str, default='sit', help='Name of the method to use')
    # Data paths (passed via command line)
    parser.add_argument('--scene_path', type=str,
                        default='/amax/home/dzr/dzr/dzr/autoCalib/hsi_f16/raw_scenes/',
                        help='Path to the raw scene data')
    parser.add_argument('--gt_path', type=str,
                        default='/amax/home/dzr/dzr/dzr/autoCalib/hsi_f16/gt_files/',
                        help='Path to the ground truth data')
    parser.add_argument('--json_file', type=str,
                        default='/amax/home/dzr/dzr/dzr/autoCalib/data_process/real_light_split.json',
                        help='Path to the JSON file for data splits')
    opt = parser.parse_args()

    trainer.Trainer(opt)
