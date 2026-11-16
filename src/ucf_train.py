import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import random

from model import CLIPVAD
from ucf_test import test
from utils.dataset import UCFDataset
from utils.tools import get_prompt_text, get_batch_label
import ucf_option

def CLASM(logits, labels, lengths, device):
    """
    CLASM (Class-specific Attention-based Subtyping Method) 损失计算函数
    
    该函数实现了一种类别的注意力机制损失计算，通过选取top-k实例进行池化，
    然后计算标签与实例logits之间的交叉熵损失。
    
    Args:
        logits (torch.Tensor): 模型输出的原始logits，形状为(batch_size, sequence_length)
        labels (torch.Tensor): 标签张量，形状为(batch_size, num_classes)，需要被归一化
        lengths (torch.Tensor): 每个序列的实际长度，用于确定有效范围
        device (torch.device): 计算设备，用于将张量移动到指定设备
    
    Returns:
        torch.Tensor: 计算得到的MIL(Multiple Instance Learning)损失值，标量张量
    """
    # 初始化实例logits张量
    instance_logits = torch.zeros(0).to(device)
    
    # 对标签进行归一化处理，使其在每个样本维度上和为1
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)

    # 遍历批次中的每个样本，提取top-k实例并计算平均值
    for i in range(logits.shape[0]):
        # 从当前样本的有效范围内选取top-k个最大值（k为序列长度的1/16+1）
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True, dim=0)
        # 计算选中实例的平均值，并拼接到instance_logits中
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)

    # 计算标签与softmax后的实例logits之间的交叉熵损失
    milloss = -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss

def CLAS2(logits, labels, lengths, device):
    '''
    实现一种弱监督学习策略，通过对视频序列中的关键片段进行采样和聚合，来判断整个视频是否包含异常事件。
    
    :param logits: 模型输出的原始预测分数（未经过激活函数），[batch_size, sequence_length]
    :param labels: 真实标签，表示样本的类别信息，[batch_size, num_classes] 或 [batch_size,]
    :param lengths: 序列长度，指示每个样本的有效序列长度，[batch_size,]
    :param device: Description
    '''
    # 创建空的张量instance_logits，用于存储每个样本的实例级logits
    instance_logits = torch.zeros(0).to(device)
    # 第一列 labels[:, 0] 代表 "normal" 类别的概率或标签值，通过 1 - labels[:, 0] 将其转换为 "anomaly" 类别的概率或标签值
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
    labels = labels.to(device)
    # 对logits应用sigmoid函数，将其转换为[0,1]范围内的概率值
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])

    for i in range(logits.shape[0]):
        # 从有效序列中选取前k个最大值，其中k是根据序列长度动态计算的（长度的1/16加1）
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True)
        # 对选出的top-k值求平均，并转换为单元素张量
        tmp = torch.mean(tmp).view(1)
        # 将当前样本的聚合logit拼接到总张量中
        instance_logits = torch.cat([instance_logits, tmp], dim=0)
    # 计算二分类交叉熵损失，将聚合的logits与标签进行比较
    clsloss = F.binary_cross_entropy(instance_logits, labels)
    return clsloss

def train(model, normal_loader, anomaly_loader, testloader, args, label_map, device):
    model.to(device)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    prompt_text = get_prompt_text(label_map)
    ap_best = 0
    epoch = 0

    if args.use_checkpoint == True:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        ap_best = checkpoint['ap']
        print("checkpoint info:")
        print("epoch:", epoch+1, " ap:", ap_best)

    for e in range(args.max_epoch):
        model.train()
        loss_total1 = 0
        loss_total2 = 0
        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)
        for i in range(min(len(normal_loader), len(anomaly_loader))):
            step = 0
            normal_features, normal_label, normal_lengths = next(normal_iter)
            anomaly_features, anomaly_label, anomaly_lengths = next(anomaly_iter)

            visual_features = torch.cat([normal_features, anomaly_features], dim=0).to(device)
            text_labels = list(normal_label) + list(anomaly_label)
            feat_lengths = torch.cat([normal_lengths, anomaly_lengths], dim=0).to(device)
            text_labels = get_batch_label(text_labels, prompt_text, label_map).to(device)

            text_features, logits1, logits2 = model(visual_features, None, prompt_text, feat_lengths) 
            #loss1
            loss1 = CLAS2(logits1, text_labels, feat_lengths, device) 
            loss_total1 += loss1.item()
            #loss2
            loss2 = CLASM(logits2, text_labels, feat_lengths, device)
            loss_total2 += loss2.item()
            #loss3
            # 通过最小化normal与其他异常类别之间的相似度；抑制异常类别间的相似性：防止不同异常类型之间产生过高的语义相似度
            # 增强特征区分度：让不同异常类型的文本特征更加独立
            loss3 = torch.zeros(1).to(device)
            text_feature_normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
            # 余弦相似度计算：遍历其余文本特征（异常类别），
            for j in range(1, text_features.shape[0]):
                # 对每个异常类别特征也进行归一化，
                text_feature_abr = text_features[j] / text_features[j].norm(dim=-1, keepdim=True)
                # 计算与normal类别的余弦相似度（点积运算，因为已经归一化），并累加到loss3中
                loss3 += torch.abs(text_feature_normal @ text_feature_abr)
            loss3 = loss3 / 13 * 1e-1

            loss = loss1 + loss2 + loss3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += i * normal_loader.batch_size * 2
            # 每处理1280个样本就进行一次模型评估，并保存性能最好的模型检查点
            if step % 1280 == 0 and step != 0:
                print('epoch: ', e+1, '| step: ', step, '| loss1: ', loss_total1 / (i+1), '| loss2: ', loss_total2 / (i+1), '| loss3: ', loss3.item())
                AUC, AP = test(model, testloader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)
                AP = AUC

                if AP > ap_best:
                    ap_best = AP 
                    checkpoint = {
                        'epoch': e,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'ap': ap_best}
                    torch.save(checkpoint, args.checkpoint_path)
                
        scheduler.step()
        
        torch.save(model.state_dict(), 'model/model_cur.pth')
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    checkpoint = torch.load(args.checkpoint_path)
    torch.save(checkpoint['model_state_dict'], args.model_path)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = ucf_option.parser.parse_args()
    setup_seed(args.seed)

    label_map = dict({'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest', 'Arson': 'arson', 'Assault': 'assault', 'Burglary': 'burglary', 'Explosion': 'explosion', 'Fighting': 'fighting', 'RoadAccidents': 'roadAccidents', 'Robbery': 'robbery', 'Shooting': 'shooting', 'Shoplifting': 'shoplifting', 'Stealing': 'stealing', 'Vandalism': 'vandalism'})

    normal_dataset = UCFDataset(args.visual_length, args.train_list, False, label_map, True)
    normal_loader = DataLoader(normal_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    anomaly_dataset = UCFDataset(args.visual_length, args.train_list, False, label_map, False)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    test_dataset = UCFDataset(args.visual_length, args.test_list, True, label_map)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, device)

    train(model, normal_loader, anomaly_loader, test_loader, args, label_map, device)