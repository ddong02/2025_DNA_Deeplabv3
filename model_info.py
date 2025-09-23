import network
import utils
import os
import argparse
from torchvision import transforms as T
import torch
import torch.nn as nn

from thop import profile
from pytorch_model_summary import summary

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--input", type=str, required=False,
                        help="path to a single image or image directory")

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )

    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--save_val_results_to", default=None,
                        help="save segmentation results to the specified dir")

    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--crop_size", type=int, default=513)

    
    parser.add_argument("--ckpt", default=None, type=str,
                        help="resume from checkpoint")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    return parser

def main():
    opts = get_argparser().parse_args()
    opts.num_classes = 19

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)
    
    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'), weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        print("Resume model from %s" % opts.ckpt)
        del checkpoint
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    ## deeplabv3 → modify classes num 21 to 19
    last_layers = model.module.classifier.classifier[1:]
    print(f"\n### original model ###\n{last_layers}")
    last_layer_in_channels = model.module.classifier.classifier[4].in_channels
    new_num_classes = 19
    print(f"\n✅ new class num: {new_num_classes}")
    new_last_layer = nn.Conv2d(last_layer_in_channels, new_num_classes, kernel_size=(1, 1), stride=(1, 1))
    new_last_layer = new_last_layer.to(device)
    model.module.classifier.classifier[4] = new_last_layer
    new_last_layers = model.module.classifier.classifier[1:]
    print(f"### modified model ###\n{new_last_layers}")
    print(model)
    return

    # Parameters
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1024 **2:.3f} M")
    # dummy input
    x = torch.randn(1, 3, 1080, 1920, device=device)
    # FLOPs
    flops, params = profile(model.module, inputs=(x,))
    print(f'Parameters: {params/(1024*1024):.2f} M')
    result = f'✅ FLOPs: {flops/(1024*1024*1024):.2f} G'
    print(result)
    # Summary
    print(summary(model.module, x, max_depth=None, show_parent_layers=True, print_summary=True))    

if __name__ == '__main__':
    main()

# python model_info.py --model which_model_to_analyze --ckpt path_to_model_pth_file
# python model_info.py --model deeplabv3_mobilenet --ckpt checkpoints/best_deeplabv3_mobilenet_voc_os16.pth
# python model_info.py --model deeplabv3_mobilenet --ckpt checkpoints/latest_deeplabv3_mobilenet_2025dna_os8.pth