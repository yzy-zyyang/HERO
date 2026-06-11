import torch


# 32通道顺序
# 0  Fp1
# 1  Fz
# 2  F3
# 3  F7
# 4  FC5
# 5  FC1
# 6  C3
# 7  T7
# 8  CP5
# 9  CP1
# 10 Pz
# 11 P3
# 12 P7
# 13 O1
# 14 Oz
# 15 O2
# 16 P4
# 17 P8
# 18 CP6
# 19 CP2
# 20 Cz
# 21 C4
# 22 T8
# 23 FC6
# 24 FC2
# 25 F4
# 26 F8
# 27 Fp2
# 28 F9
# 29 P9
# 30 P10
# 31 F10


def feature_trans(subgraph_num, feature):
    if subgraph_num == 7:
        return feature_trans_7(feature)
    elif subgraph_num == 2:
        return feature_trans_2(feature)
    else:
        raise ValueError(f"Unsupported subgraph_num: {subgraph_num}")


def location_trans(subgraph_num, location):
    if subgraph_num == 7:
        return location_trans_7(location)
    elif subgraph_num == 2:
        return location_trans_2(location)
    else:
        raise ValueError(f"Unsupported subgraph_num: {subgraph_num}")


##############################################################
########################### 半脑分割（32通道）
##############################################################

def feature_trans_2(feature):
    """
    32通道半脑分割
    左半球 + 右半球
    注意：中线通道(Fz, Pz, Cz, Oz)不纳入2区半脑虚拟节点，
    它们仍保留在原始节点中参与全局层。
    """
    reassigned_feature = torch.cat((
        # 左半脑: Fp1, F3, F7, FC5, FC1, C3, T7, CP5, CP1, P3, P7, O1, F9, P9
        feature[:, 0:1], feature[:, 2:4], feature[:, 4:6], feature[:, 6:10],
        feature[:, 11:13], feature[:, 13:14], feature[:, 28:30],

        # 右半脑: Fp2, FC2, F4, F8, FC6, C4, T8, CP6, CP2, P4, P8, O2, P10, F10
        feature[:, 27:28], feature[:, 24:27], feature[:, 23:24], feature[:, 21:23],
        feature[:, 18:20], feature[:, 16:18], feature[:, 15:16], feature[:, 30:32],
    ), dim=1)

    return reassigned_feature


def location_trans_2(location):
    reassigned_location = torch.cat((
        # 左半脑
        location[0:1], location[2:4], location[4:6], location[6:10],
        location[11:13], location[13:14], location[28:30],

        # 右半脑
        location[27:28], location[24:27], location[23:24], location[21:23],
        location[18:20], location[16:18], location[15:16], location[30:32],
    ), dim=0)

    return reassigned_location


##############################################################
########################### ROI七区分割（32通道）
##############################################################

def feature_trans_7(feature):
    """
    按用户指定的7区ROI分组重排特征。

    区域1：前额中线区      Fp1, Fz, Fp2
    区域2：左额-左额中央区  F3, F7, FC5, FC1
    区域3：右额-右额中央区  FC2, F4, F8, FC6
    区域4：左中央-左顶颞区  C3, T7, CP5, CP1, P3, P7, O1, F9, P9
    区域5：中线中央-顶枕区  Cz, Pz, Oz
    区域6：右中央-右顶颞区  C4, T8, CP6, CP2, P4, P8, O2, F10, P10
    区域7：枕后/双侧后部补充区 O1, Oz, O2

    注意：区域7与区域4/5/6存在重叠，这是按你指定的分区方案保留的“后部补充功能区”。
    """
    reassigned_feature = torch.cat((
        # 区域1：Fp1, Fz, Fp2
        feature[:, 0:1], feature[:, 1:2], feature[:, 27:28],

        # 区域2：F3, F7, FC5, FC1
        feature[:, 2:4], feature[:, 4:6],

        # 区域3：FC2, F4, F8, FC6
        feature[:, 24:27], feature[:, 23:24],

        # 区域4：C3, T7, CP5, CP1, P3, P7, O1, F9, P9
        feature[:, 6:10], feature[:, 11:14], feature[:, 28:30],

        # 区域5：Cz, Pz, Oz
        feature[:, 20:21], feature[:, 10:11], feature[:, 14:15],

        # 区域6：C4, T8, CP6, CP2, P4, P8, O2, F10, P10
        feature[:, 21:23], feature[:, 18:20], feature[:, 16:18],
        feature[:, 15:16], feature[:, 31:32], feature[:, 30:31],

        # 区域7：O1, Oz, O2
        feature[:, 13:16],
    ), dim=1)

    return reassigned_feature


def location_trans_7(location):
    reassigned_location = torch.cat((
        # 区域1
        location[0:1], location[1:2], location[27:28],

        # 区域2
        location[2:4], location[4:6],

        # 区域3
        location[24:27], location[23:24],

        # 区域4
        location[6:10], location[11:14], location[28:30],

        # 区域5
        location[20:21], location[10:11], location[14:15],

        # 区域6
        location[21:23], location[18:20], location[16:18],
        location[15:16], location[31:32], location[30:31],

        # 区域7
        location[13:16],
    ), dim=0)

    return reassigned_location
