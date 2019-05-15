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
from util.utils import make_weights_for_balanced_classes, get_val_data, separate_irse_bn_paras, separate_resnet_bn_paras, warm_up_lr, schedule_lr, perform_val, get_time, buffer_val, AverageMeter, accuracy

from tqdm import tqdm
import os
import sys
from pdb import set_trace as bp

if __name__ == '__main__':

    DATA_ROOT = '/Users/eugene/Documents/Developer/Computer-Vision-Python/evoLVe_data/data' # the parent root where your train/val/test data are stored
    INPUT_SIZE = [112, 112] # support: [112, 112] and [224, 224]
    BACKBONE_RESUME_ROOT = '/Users/eugene/Documents/Developer/Computer-Vision-Python/evoLVe_data/pth/backbone_ir50_ms1m_epoch120.pth' # the root to resume training from a saved checkpoint

    MULTI_GPU = False# flag to use multiple GPUs; if you choose to train with single GPU, you should first run "export CUDA_VISILE_DEVICES=device_id" to specify the GPU card you want to use
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    EMBEDDING_SIZE = 512 # feature dimension
    BATCH_SIZE = 512

    # lfw, cfp_ff, cfp_fp, agedb, calfw, cplfw, vgg2_fp, lfw_issame, cfp_ff_issame, cfp_fp_issame, agedb_issame, calfw_issame, cplfw_issame, vgg2_fp_issame = get_val_data(DATA_ROOT)
    lfw, lfw_issame = get_val_data(DATA_ROOT)
    
    BACKBONE = IR_50(INPUT_SIZE)

    if os.path.isfile(BACKBONE_RESUME_ROOT):
        print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT))
        BACKBONE.load_state_dict(torch.load(BACKBONE_RESUME_ROOT, map_location='cpu'))
    else:
        print("No Checkpoint Found at '{}'.".format(BACKBONE_RESUME_ROOT))
        sys.exit()

    print("=" * 60)
    print("Perform Evaluation on LFW, CFP_FF, CFP_FP, AgeDB, CALFW, CPLFW and VGG2_FP, and Save Checkpoints...")
    accuracy_lfw, best_threshold_lfw, roc_curve_lfw = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, lfw, lfw_issame)
    print("Evaluation: LFW Acc: {}".format(accuracy_lfw))



    # buffer_val(writer, "LFW", accuracy_lfw, best_threshold_lfw, roc_curve_lfw, epoch + 1)
    # accuracy_cfp_ff, best_threshold_cfp_ff, roc_curve_cfp_ff = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, cfp_ff, cfp_ff_issame)
    # buffer_val(writer, "CFP_FF", accuracy_cfp_ff, best_threshold_cfp_ff, roc_curve_cfp_ff, epoch + 1)
    # accuracy_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, cfp_fp, cfp_fp_issame)
    # buffer_val(writer, "CFP_FP", accuracy_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp, epoch + 1)
    # accuracy_agedb, best_threshold_agedb, roc_curve_agedb = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, agedb, agedb_issame)
    # buffer_val(writer, "AgeDB", accuracy_agedb, best_threshold_agedb, roc_curve_agedb, epoch + 1)
    # accuracy_calfw, best_threshold_calfw, roc_curve_calfw = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, calfw, calfw_issame)
    # buffer_val(writer, "CALFW", accuracy_calfw, best_threshold_calfw, roc_curve_calfw, epoch + 1)
    # accuracy_cplfw, best_threshold_cplfw, roc_curve_cplfw = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, cplfw, cplfw_issame)
    # buffer_val(writer, "CPLFW", accuracy_cplfw, best_threshold_cplfw, roc_curve_cplfw, epoch + 1)
    # accuracy_vgg2_fp, best_threshold_vgg2_fp, roc_curve_vgg2_fp = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, vgg2_fp, vgg2_fp_issame)
    # buffer_val(writer, "VGGFace2_FP", accuracy_vgg2_fp, best_threshold_vgg2_fp, roc_curve_vgg2_fp, epoch + 1)
    # print("Epoch {}/{}, Evaluation: LFW Acc: {}, CFP_FF Acc: {}, CFP_FP Acc: {}, AgeDB Acc: {}, CALFW Acc: {}, CPLFW Acc: {}, VGG2_FP Acc: {}".format(epoch + 1, NUM_EPOCH, accuracy_lfw, accuracy_cfp_ff, accuracy_cfp_fp, accuracy_agedb, accuracy_calfw, accuracy_cplfw, accuracy_vgg2_fp))
    print("=" * 60)
