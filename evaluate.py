import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from config import configurations
from backbone.model_resnet import ResNet_50, ResNet_101, ResNet_152
from backbone.model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
from head.metrics import ArcFace, CosFace, SphereFace, Am_softmax
from loss.focal import FocalLoss
from util.utils import make_weights_for_balanced_classes, get_val_data, separate_irse_bn_paras, separate_resnet_bn_paras, warm_up_lr, schedule_lr, perform_val, get_time, buffer_val, AverageMeter, accuracy, get_val_pair

import argparse
from tqdm import tqdm
import os
import sys
from pdb import set_trace as bp


####   backbone_ir50_ms1m_epoch120.pth   LFW Acc: 0.9984999999999999
####   backbone_ir50_ms1m_epoch120.pth CALFW Acc: 0.9578333333333331
####   backbone_ir50_ms1m_epoch120.pth CPLFW Acc: 0.9223333333333332
####   backbone_ir50_ms1m_epoch120.pth CFP-FF Acc: 0.9972857142857142
####   backbone_ir50_ms1m_epoch120.pth CFP-FP Acc: 0.9811428571428571
####   backbone_ir50_ms1m_epoch120.pth AgeDB_30 Acc: 0.9773333333333334
####   backbone_ir50_ms1m_epoch120.pth VggFace2_FP Acc: 0.9518000000000001

####   IR_50_MODEL_arcface_casia_epoch56_lfw9925.pth  LFW Acc: 
####   IR_50_MODEL_arcface_casia_epoch56_lfw9925.pth  CALFW Acc: 
####   IR_50_MODEL_arcface_casia_epoch56_lfw9925.pth  CPLFW Acc: 
####   IR_50_MODEL_arcface_casia_epoch56_lfw9925.pth  CFP-FF Acc: 
####   IR_50_MODEL_arcface_casia_epoch56_lfw9925.pth  CFP-FP Acc: 
####   IR_50_MODEL_arcface_casia_epoch56_lfw9925.pth  AgeDB_30 Acc: 
####   IR_50_MODEL_arcface_casia_epoch56_lfw9925.pth  VggFace2_FP Acc: 

"""
EXAMPLE:
python3 validate.py  \
--model_path ./../pytorch-face/pth/IR_50_MODEL_arcface_casia_epoch56_lfw9925.pth \
--data_root ./DATA/data \
--model_type IR_50 \
--num_workers 8 \
--batch_size 100
"""

def main(ARGS):

    if ARGS.model_path == None:
        raise AssertionError("Path should not be None")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ####### Model setup
    print('Model type: %s' % ARGS.model_type)
    if ARGS.model_type == 'ResNet_50':
        model = ResNet_50(ARGS.input_size)
    elif ARGS.model_type == 'ResNet_101':
        model = ResNet_101(ARGS.input_size)
    elif ARGS.model_type == 'ResNet_152':
        model = ResNet_152(ARGS.input_size)
    elif ARGS.model_type == 'IR_50':
        model = IR_50(ARGS.input_size)
    elif ARGS.model_type == 'IR_101':
        model = IR_101(ARGS.input_size)
    elif ARGS.model_type == 'IR_152':
        model = IR_152(ARGS.input_size)
    elif ARGS.model_type == 'IR_SE_50':
        model = IR_SE_50(ARGS.input_size)
    elif ARGS.model_type == 'IR_SE_101':
        model = IR_SE_101(ARGS.input_size)
    elif ARGS.model_type == 'IR_SE_152':
        model = IR_SE_152(ARGS.input_size)
    else:
        raise AssertionError('Unsuported model_type {}. We only support: [\'ResNet_50\', \'ResNet_101\', \'ResNet_152\', \'IR_50\', \'IR_101\', \'IR_152\', \'IR_SE_50\', \'IR_SE_101\', \'IR_SE_152\']'.format(ARGS.model_type))

    if use_cuda:
        model.load_state_dict(torch.load(ARGS.model_path))
    else:
        model.load_state_dict(torch.load(ARGS.model_path, map_location='cpu'))

    model.to(device)
    # embedding_size = 512
    model.eval()    

    # DATA_ROOT = './../evoLVe_data/data' # the parent root where your train/val/test data are stored
    # INPUT_SIZE = [112, 112] # support: [112, 112] and [224, 224]
    # BACKBONE_RESUME_ROOT = './../evoLVe_data/pth/backbone_ir50_ms1m_epoch120.pth' # the root to resume training from a saved checkpoint
    # BACKBONE_RESUME_ROOT = './../pytorch-face/pth/IR_50_MODEL_arcface_casia_epoch56_lfw9925.pth' 

    MULTI_GPU = False# flag to use multiple GPUs; if you choose to train with single GPU, you should first run "export CUDA_VISILE_DEVICES=device_id" to specify the GPU card you want to use
    # DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # EMBEDDING_SIZE = 512 # feature dimension
    # BATCH_SIZE = 512



    # BACKBONE = IR_50(INPUT_SIZE)

    # if os.path.isfile(BACKBONE_RESUME_ROOT):
    #     print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT))
    #     BACKBONE.load_state_dict(torch.load(BACKBONE_RESUME_ROOT, map_location='cpu'))
    # else:
    #     print("No Checkpoint Found at '{}'.".format(BACKBONE_RESUME_ROOT))
    #     sys.exit()

    print("=" * 60)
    print("Performing Evaluation on LFW, CFP_FF, CFP_FP, AgeDB, CALFW, CPLFW and VGG2_FP, and Save Checkpoints...")

    #### LFW
    print("Performing Evaluation on LFW...")
    lfw, lfw_issame = get_val_pair(ARGS.data_root, 'lfw')
    accuracy_lfw, best_threshold_lfw, roc_curve_lfw = perform_val(MULTI_GPU, device, ARGS.embedding_size, ARGS.batch_size, model, lfw, lfw_issame)
    print("Evaluation: LFW Acc: {}".format(accuracy_lfw))

    #### CALFW WORKS
    print("Performing Evaluation on CALFW...")
    calfw, calfw_issame = get_val_pair(ARGS.data_root, 'calfw')
    accuracy_calfw, best_threshold_calfw, roc_curve_calfw = perform_val(MULTI_GPU, device, ARGS.embedding_size, ARGS.batch_size, model, calfw, calfw_issame)
    print("Evaluation: CALFW Acc: {}".format(accuracy_calfw))

    #### CPLFW
    print("Performing Evaluation on CPLFW...")
    cplfw, cplfw_issame = get_val_pair(ARGS.data_root, 'cplfw')
    accuracy_cplfw, best_threshold_calfw, roc_curve_calfw = perform_val(MULTI_GPU, device, ARGS.embedding_size, ARGS.batch_size, model, cplfw, cplfw_issame)
    print("Evaluation: CPLFW Acc: {}".format(accuracy_cplfw))

    #### CFP-FF
    print("Performing Evaluation on CFP-FF...")
    cfp_ff, cfp_ff_issame = get_val_pair(ARGS.data_root, 'cfp_ff')
    accuracy_cfp_ff, best_threshold_cfp_ff, roc_curve_cfp_ff = perform_val(MULTI_GPU, device, ARGS.embedding_size, ARGS.batch_size, model, cfp_ff, cfp_ff_issame)
    print("Evaluation: CFP-FF Acc: {}".format(accuracy_cfp_ff))

    #### CFP-FP
    print("Performing Evaluation on CFP-FP...")
    cfp_fp, cfp_fp_issame = get_val_pair(ARGS.data_root, 'cfp_fp')
    accuracy_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp = perform_val(MULTI_GPU, device, ARGS.embedding_size, ARGS.batch_size, model, cfp_fp, cfp_fp_issame)
    print("Evaluation: CFP-FP Acc: {}".format(accuracy_cfp_fp))

    #### AgeDB_30
    print("Performing Evaluation on AgeDB_30...")
    agedb_30, agedb_30_issame = get_val_pair(ARGS.data_root, 'agedb_30')
    accuracy_agedb, best_threshold_agedb, roc_curve_agedb = perform_val(MULTI_GPU, device, ARGS.embedding_size, ARGS.batch_size, model, agedb_30, agedb_30_issame)
    print("Evaluation: AgeDB_30 Acc: {}".format(accuracy_agedb))

    #### VggFace2_FP
    print("Performing Evaluation on VggFace2_FP...")
    vgg2_fp, vgg2_fp_issame = get_val_pair(ARGS.data_root, 'vgg2_fp')
    accuracy_vgg2_fp, best_threshold_vgg2_fp, roc_curve_vgg2_fp = perform_val(MULTI_GPU, device, ARGS.embedding_size, ARGS.batch_size, model, vgg2_fp, vgg2_fp_issame)
    print("Evaluation: VggFace2_FP Acc: {}".format(accuracy_vgg2_fp))
    
    print("=" * 60)
    print("FINAL RESULTS:")
    print("Epoch {}/{}, Evaluation: LFW Acc: {}, CFP_FF Acc: {}, CFP_FP Acc: {}, AgeDB Acc: {}, CALFW Acc: {}, CPLFW Acc: {}, VGG2_FP Acc: {}".format(epoch + 1, NUM_EPOCH, accuracy_lfw, accuracy_cfp_ff, accuracy_cfp_fp, accuracy_agedb, accuracy_calfw, accuracy_cplfw, accuracy_vgg2_fp))
    print("=" * 60)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='Model weights.', default=None)
    parser.add_argument('--data_root', type=str, help='data root.', default='./../evoLVe_data/data')
    parser.add_argument('--model_type', type=str, help='Model type to use for training.', default='IR_50')# support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
    parser.add_argument('--input_size', type=str, help='support: [112, 112] and [224, 224]', default=[112, 112])
    parser.add_argument('--num_workers', type=int, help='Number of threads to use for data pipeline.', default=8)
    parser.add_argument('--batch_size', type=int, help='Number of batches while validating model.', default=100)
    parser.add_argument('--embedding_size', type=int, help='Number of threads to use for data pipeline.', default=512)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
