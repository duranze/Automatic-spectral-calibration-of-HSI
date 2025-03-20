import os
import torch

def create_generator(opt):
    # Use method_name (default 'sit') to select the generator method
    method = opt.method_name.lower()
    if method == 'sit':
        import sit
        generator = sit.SIT(dim=32, num_blocks=[2, 2, 2, 2]).cuda()
    # elif method == 'anyformer':
    #     # Example: if there is an anyFormer module, import and instantiate its generator here
    #     from anyFormer import MainNet
    #     generator = MainNet(opt).cuda()
    else:
        raise NotImplementedError(f"Generator method '{opt.method_name}' is not implemented.")
    
    if opt.load_name:
        trained_dict = torch.load(opt.load_name)
        load_dict(generator, trained_dict)
        print(f'Generator ({opt.method_name}) is loaded!')
    return generator

def load_dict(process_net, pretrained_net):
    # Load only the parameters that exist in process_net
    pretrained_dict = {k: v for k, v in pretrained_net.items() if k in process_net.state_dict()}
    process_net.load_state_dict({**process_net.state_dict(), **pretrained_dict})
    return process_net

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
