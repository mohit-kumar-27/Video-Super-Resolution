import argparse
import gc
import os
import pandas as pd
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
from data import get_training_set
import logger
from rbpn import Net as RBPN
from rbpn import Model_loss
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import utils

################################################## iSEEBETTER TRAINER KNOBS ############################################
UPSCALE_FACTOR = 4
########################################################################################################################

# Handle command line arguments
parser = argparse.ArgumentParser(description='Train: Super Resolution Models')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=3, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=5, help='testing batch size')
parser.add_argument('--start_epoch', type=int, default=1, help='Starting epoch for continuing training')
parser.add_argument('--nEpochs', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=1, help='Snapshots')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.01')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=2, type=int, help='number of gpu')
parser.add_argument('--data_dir', type=str, default='./Vimeo/sequences')
parser.add_argument('--file_list', type=str, default='sep_trainlist.txt')
parser.add_argument('--other_dataset', type=bool, default=False, help="use other dataset than vimeo-90k")
parser.add_argument('--future_frame', type=bool, default=True, help="use future frame")
parser.add_argument('--nFrames', type=int, default=7)
parser.add_argument('--patch_size', type=int, default=64, help='0 to use original frame size')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--model_type', type=str, default='RBPN')
parser.add_argument('--residual', type=bool, default=False)
parser.add_argument('--pretrained_sr', default='RBPN_4x.pth', help='sr pretrained base model')
parser.add_argument('--pretrained', action='store_true', help='Use pretrained model')
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--prefix', default='F7', help='Location to save checkpoint models')
parser.add_argument('--APITLoss', type=bool,default=True, help='Use APIT Loss')
parser.add_argument('--useDataParallel', action='store_true', help='Use DataParallel')
parser.add_argument('-v', '--debug', default=False, action='store_true', help='Print debug spew.')

def trainModel(epoch, training_data_loader, enhance_netG, optimizerG, generatorCriterion, device, args):
    trainBar = tqdm(training_data_loader,total=len(training_data_loader),ascii=True)
    runningResults = {'batchSize': 0, 'Loss': 0}

    enhance_netG.train()

    # Skip first iteration
    iterTrainBar = iter(trainBar)
    next(iterTrainBar)

    for data in iterTrainBar:
        batchSize = len(data)
        runningResults['batchSize'] += batchSize
        
        if args.APITLoss:
            fakeHRs = []
            targets = []

        input, target, neigbor, flow, bicubic = data[0], data[1], data[2], data[3], data[4]
        input = Variable(input).to(device=device, dtype=torch.float)
        target = Variable(target).to(device=device, dtype=torch.float)
        bicubic = Variable(bicubic).to(device=device, dtype=torch.float)
        neigbor = [Variable(j).to(device=device, dtype=torch.float) for j in neigbor]
        flow = [Variable(j).to(device=device, dtype=torch.float) for j in flow]

        fakeHR = enhance_netG(input, neigbor, flow)
        if args.residual:
            fakeHR = fakeHR + bicubic

        if args.APITLoss:
            fakeHRs.append(fakeHR)
            targets.append(target)

        ################################################################################################################
        # (1) Update Enhance network: minimize Perception Loss + Image Loss + TV Loss
        ################################################################################################################
        net_loss = 0

        # Zero-out gradients, i.e., start afresh
        enhance_netG.zero_grad()

        if args.APITLoss:
            idx = 0
            for fakeHR, HRImg in zip(fakeHRs, targets):
                fakeHR = fakeHR.to(device)
                HRImg = HRImg.to(device)
                net_loss += generatorCriterion(fakeHR, HRImg, idx)
                idx += 1
        else:
            net_loss = generatorCriterion(fakeHR, target)

        net_loss /= len(data)

        # Calculate gradients
        net_loss.backward()

        # Update weights
        optimizerG.step()

        runningResults['Loss'] += net_loss.item() * args.batchSize
        
        trainBar.set_description(desc='[Epoch: %d/%d] Epoch Loss: %.4f' %
                                       (epoch, args.nEpochs,runningResults['Loss'] / runningResults['batchSize']))
        gc.collect()

    enhance_netG.eval()

    # learning rate is decayed by a factor of 10 every half of total epochs
    if (epoch + 1) % (args.nEpochs / 2) == 0:
        for param_group in optimizerG.param_groups:
            param_group['lr'] /= 10.0
        logger.info('Learning rate decay: lr=%s', (optimizerG.param_groups[0]['lr']))

    return runningResults

def saveModelParams(epoch, runningResults, enhancer_netG):
    results = {'Loss': [], 'PSNR': [], 'SSIM': []}

    # Save model parameters
    torch.save(enhancer_netG.state_dict(), 'weights/enhancer_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
    logger.info("Checkpoint saved to {}".format('weights/netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch)))

    # Save Loss\Scores\PSNR\SSIM
    results['Loss'].append(runningResults['Loss'] / runningResults['batchSize'])
    # results['PSNR'].append(validationResults['PSNR'])
    #results['SSIM'].append(validationResults['SSIM'])

    if epoch % 1 == 0 and epoch != 0:
        out_path = 'statistics/'
        data_frame = pd.DataFrame(data={'Loss': results['Loss'], },#, 'PSNR': results['PSNR'], 'SSIM': results['SSIM']},
                                  index=range(1, epoch + 1))
        data_frame.to_csv(out_path + 'iSeeBetter_' + str(UPSCALE_FACTOR) + '_Train_Results.csv', index_label='Epoch')

def main():
    """ Lets begin the training process! """

    args = parser.parse_args()

    # Initialize Logger
    logger.initLogger(args.debug)
    # Load dataset
    logger.info('==> Loading datasets')
    train_set = get_training_set(args.data_dir, args.nFrames, args.upscale_factor, args.data_augmentation,
                                 args.file_list,
                                 args.other_dataset, args.patch_size, args.future_frame)
    training_data_loader = DataLoader(dataset=train_set, num_workers=args.threads, batch_size=args.batchSize,
                                      shuffle=True)
    # Use generator as RBPN
    enhancher_netG = RBPN(num_channels=3, base_filter=256, feat=64, num_stages=3, n_resblock=5, nFrames=args.nFrames,
                scale_factor=args.upscale_factor)
    logger.info('# of Generator parameters: %s', sum(param.numel() for param in enhancher_netG.parameters()))

    # Use DataParallel?
    if args.useDataParallel:
        gpus_list = range(args.gpus)
        print(gpus_list)
        enhancher_netG = torch.nn.DataParallel(enhancher_netG, device_ids=gpus_list)

    # Generator loss
    generatorCriterion = nn.L1Loss() if not args.APITLoss else Model_loss()

    # Specify device
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu_mode else "cpu")

    if args.gpu_mode and torch.cuda.is_available():
        utils.printCUDAStats()
        enhancher_netG.to(device)
        generatorCriterion.to(device)

    # Use Adam optimizer
    optimizerG = optim.Adam(enhancher_netG.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    # optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

    if args.APITLoss:
        logger.info("Enhancer Loss: Adversarial Loss + Perception Loss + Image Loss + TV Loss")
    else:
        logger.info("Enhancer Loss: L1 Loss")

    # print iSeeBetter architecture
    utils.printNetworkArch(enhancher_netG)

    if args.pretrained:
        modelPath = os.path.join(args.save_folder + args.pretrained_sr)
        utils.loadPreTrainedModel(gpuMode=args.gpu_mode, model=enhancher_netG, modelPath=modelPath)

    for epoch in range(args.start_epoch, args.nEpochs + 1):
        runningResults = trainModel(epoch, training_data_loader, enhancher_netG, optimizerG, generatorCriterion, device, args)

        if (epoch + 1) % (args.snapshots) == 0:
            saveModelParams(epoch, runningResults, enhancher_netG)

if __name__ == "__main__":
    main()