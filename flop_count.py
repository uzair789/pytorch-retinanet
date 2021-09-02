"""Script to compute the flops in the models"""

import numpy as np

STAGE_OUT_CHANNEL = [64] + [64] * 4 + [128] * 4 + [256] * 4 + [512] * 4
STAGE_IN_SIZE = [112] + [56] * 4 + [28] * 4 + [14] * 4 + [7] * 4
stage_repeat = [1,4,4,4,4]
channel_scale = [1]

def cal_flops(stage_out_channel=STAGE_OUT_CHANNEL, stage_in_size=STAGE_IN_SIZE):
    flops_total = 0
    for i in range(len(stage_out_channel)):
        if i == 0 :
            flops = 3 * stage_out_channel[i] * stage_in_size[i] ** 2 * 7 * 7
            print (('7x7conv: {}, {}, {}, {}').format(3, stage_out_channel[i], stage_in_size[i], stage_in_size[i]))
        else:
            flops_conv3x3 = stage_out_channel[i-1] * stage_out_channel[i] * stage_in_size[i]**2 * 3 * 3
            print (('3x3conv: {}, {}, {}, {}').format(stage_out_channel[i-1], stage_out_channel[i-1],stage_in_size[i],stage_in_size[i]))
            flops_downsample = 0
            if stage_out_channel [i-1] != stage_out_channel[i]:
                flops_downsample = stage_out_channel[i-1] * stage_out_channel[i] * stage_in_size[i]**2 * 1 * 1
                print (('downsample_1x1conv: {}, {}, {}, {}').format(stage_out_channel [i-1], stage_out_channel[i], stage_in_size[i], stage_in_size[i]))
            flops = flops_conv3x3 + flops_downsample
            print('flops=', flops)
        flops_total +=flops
    #flops_fc = 2048*1000 + 1000
    #flops_total +=flops_fc
    return flops_total


def cal_flops_binary_backbone(stage_out_channel=STAGE_OUT_CHANNEL, stage_in_size=STAGE_IN_SIZE):
    flops_cal_total = 0
    bops_cal_total = 0

    for i in range(len(stage_out_channel)):
        flops_cal = 0
        bops_cal = 0
        if i == 0 :
            flops_cal = 3 * stage_out_channel[i] * stage_in_size[i] ** 2 * 7 * 7
            #print (('7x7conv: {}, {}, {}, {}').format(3, stage_out_channel[i], stage_in_size[i], stage_in_size[i]))
        else:
            bops_cal = stage_out_channel[i-1] * stage_out_channel[i] * stage_in_size[i]**2 * 3 * 3
            #print (('3x3conv: {}, {}, {}, {}').format(stage_out_channel[i-1], stage_out_channel[i-1],stage_in_size[i],stage_in_size[i]))
            if stage_out_channel [i-1] != stage_out_channel[i]:
                flops_cal = stage_out_channel[i-1] * stage_out_channel[i] * stage_in_size[i]**2 * 1 * 1
                #print (('downsample_1x1conv: {}, {}, {}, {}').format(stage_out_channel [i-1], stage_out_channel[i], stage_in_size[i], stage_in_size[i]))
            #print('flops=', flops)
        flops_cal_total += flops_cal
        bops_cal_total += bops_cal
    #flops_fc = 2048*1000
    #flops_cal_total += flops_fc
    return flops_cal_total, bops_cal_total

def cal_flops_FPN():
    flops_P5_1 = 512 * 256 * 1*1 * 26*20
    flops_P5_2 = 256 * 256 * 3*3 * 26*20

    flops_P4_1 = 256 * 256 * 1*1 * 52*40
    flops_P4_2 = 256 * 256 * 3*3 * 52*40

    flops_P3_1 = 128 * 256 * 1*1 * 104*80
    flops_P3_2 = 256 * 256 * 3*3 * 104*80

    flops_P6 = 512 * 256 * 3*3 * 13*10
    flops_P7_2 = 256 * 256 * 3*3 * 7*5

    # upsample layers are missing
    total_flops = flops_P5_1 + flops_P5_2 + flops_P4_1 + flops_P4_2 + flops_P3_1 + flops_P3_2 + flops_P6 + flops_P7_2
    return total_flops


def cal_flops_regression_head():
    input_feature_tensors = [[1, 256, 104, 80],
                             [1, 256, 52, 40],
                             [1, 256, 26, 20],
                             [1, 256, 13, 10],
                             [1, 256, 7, 5]]
    sum_flops_conv1 = 0
    sum_flops_conv2 = 0
    sum_flops_conv3 = 0
    sum_flops_conv4 = 0
    sum_flops_output = 0

    sum_flops_se2 = 0
    sum_flops_se3 = 0
    sum_flops_se4 = 0

    for [b, in_channels, h, w] in input_feature_tensors:

        flops_conv1 = in_channels * 256 * 3*3 * h*w
        flops_conv2 = 256 * 256 * 3*3* h*w

        #se2
        flops_se2 = (256 * 256/16) + (256/16 * 256) # the two linear layers, skiping avgpool for now

        flops_conv3 = 256 * 256 * 3*3 * h*w

        #se3
        flops_se3 = (256 * 256/16) + (256/16 * 256) # the two linear layers, skiping avgpool for now

        flops_conv4 = 256 * 256 * 3*3 * h*w

        #se4
        flops_se4 = (256 * 256/16) + (256/16 * 256) # the two linear layers, skiping avgpool for now

        flops_output = 256 * 36 * 3*3 * h*w
        sum_flops_conv1 += flops_conv1
        sum_flops_conv2 += flops_conv2
        sum_flops_conv3 += flops_conv3
        sum_flops_conv4 += flops_conv4
        sum_flops_output += flops_output
        sum_flops_se2 += flops_se2
        sum_flops_se3 += flops_se3
        sum_flops_se4 += flops_se4

    return [sum_flops_conv1, sum_flops_conv2, sum_flops_conv3, sum_flops_conv4, sum_flops_output, sum_flops_se2, \
        sum_flops_se3, sum_flops_se4]


def cal_flops_classification_head():

    input_feature_tensors = [[1, 256, 104, 80],
                             [1, 256, 52, 40],
                             [1, 256, 26, 20],
                             [1, 256, 13, 10],
                             [1, 256, 7, 5]]
    sum_flops_conv1 = 0
    sum_flops_conv2 = 0
    sum_flops_conv3 = 0
    sum_flops_conv4 = 0
    sum_flops_output = 0

    sum_flops_se2 = 0
    sum_flops_se3 = 0
    sum_flops_se4 = 0

    for [b, in_channels, h, w] in input_feature_tensors:

        flops_conv1 = in_channels * 256 * 3*3 * h*w
        flops_conv2 = 256 * 256 * 3*3* h*w

        #se2
        flops_se2 = (256 * 256/16) + (256/16 * 256) # the two linear layers, skiping avgpool for now

        flops_conv3 = 256 * 256 * 3*3 * h*w

        #se3
        flops_se3 = (256 * 256/16) + (256/16 * 256) # the two linear layers, skiping avgpool for now

        flops_conv4 = 256 * 256 * 3*3 * h*w

        #se4
        flops_se4 = (256 * 256/16) + (256/16 * 256) # the two linear layers, skiping avgpool for now

        flops_output = 256 * 720 * 3*3 * h*w
        sum_flops_conv1 += flops_conv1
        sum_flops_conv2 += flops_conv2
        sum_flops_conv3 += flops_conv3
        sum_flops_conv4 += flops_conv4
        sum_flops_output += flops_output
        sum_flops_se2 += flops_se2
        sum_flops_se3 += flops_se3
        sum_flops_se4 += flops_se4

    return [sum_flops_conv1, sum_flops_conv2, sum_flops_conv3, sum_flops_conv4, sum_flops_output, sum_flops_se2, \
        sum_flops_se3, sum_flops_se4]





def flops_full_precision():
    backbone_flops = cal_flops()
    fpn_flops = cal_flops_FPN()
    reg_head_flops = sum(cal_flops_regression_head()[0:5])
    class_head_flops = sum(cal_flops_classification_head()[0:5])

    total_flops = backbone_flops + fpn_flops + reg_head_flops + class_head_flops
    print('full precision total flops = {} Gigaflops'.format(total_flops/1000000000))

def flops_binary():

    backbone_flops_fp, backbone_flops_bn = cal_flops_binary_backbone()
    fpn_flops_fp = cal_flops_FPN()

    reg_head_flops = cal_flops_regression_head()
    reg_head_flops_fp = reg_head_flops[0] + reg_head_flops[4]
    reg_head_flops_bn = sum(reg_head_flops[1:4])
    reg_head_flops_fp_se = sum(reg_head_flops[5:])

    class_head_flops = cal_flops_classification_head()
    class_head_flops_fp = class_head_flops[0] + class_head_flops[4]
    class_head_flops_bn = sum(class_head_flops[1:4])
    class_head_flops_fp_se = sum(class_head_flops[5:])

    total_flops_fp = backbone_flops_fp + fpn_flops_fp + reg_head_flops_fp + class_head_flops_fp
    total_flops_bn = (backbone_flops_bn + reg_head_flops_bn + class_head_flops_bn)/64

    total_flops = total_flops_fp + total_flops_bn

    unit = 1000000000

    print('Binary version without se total flops = {} Gigaflops'.format(total_flops/1000000000))

    # with se version
    total_flops_fp += class_head_flops_fp_se + reg_head_flops_fp_se
    total_flops_with_se = total_flops_fp + total_flops_bn
    print('Binary version with se total flops = {} Gigaflops '.format(total_flops_with_se/1000000000))

    print('SE FLOPS reg se flops | fp = {} gflops | bn = {} gflops'.format(reg_head_flops_fp_se/unit, reg_head_flops_fp_se/(64*unit)))
    print('SE FLOPS cls se flops | fp = {} gflops | bn = {} gflops'.format(class_head_flops_fp_se/unit, class_head_flops_fp_se/(64*unit)))

    print('FPN FLOPS | fp = {} gflops | bn = {} gflops'.format(fpn_flops_fp/unit, fpn_flops_fp/(64*unit)))




flops_full_precision()
flops_binary()
