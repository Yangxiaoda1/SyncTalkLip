"""
Author: Jiadong Wang

Note: class SupConLoss is adapted from 'SupContrast' GitHub repository
https://github.com/HobbitLong/SupContrast
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
def qprint(var,str):
    print("\033[92m"+"{}:{}".format(str,var)+"\033[0m")

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
    
    # def forward(self, anchor_audio, positive_video, negative_audio):#trainL2
    #     #anchor_audio:[100,768],positive_video:[100,768],negative_audio:[50,100,768]
    #     dist_positive=torch.sqrt(torch.pow(anchor_audio - positive_video, 2).sum())
    #     negnum=negative_audio.shape[0]
    #     num,sum=0,0
    #     for i in range(negnum):
    #         sum=sum+torch.sqrt(torch.pow(positive_video - negative_audio[i], 2).sum())
    #         num=num+1
    #     dist_negative=sum/num
    #     return dist_positive-dist_negative
    
    def forward(self, anchor_audio, positive_video, negative_audio):#trainL1
        #anchor_audio:[100,768],positive_video:[100,768],negative_audio:[50,100,768]
        dist_positive=torch.abs(anchor_audio - positive_video).sum()/(anchor_audio.shape[0])
        negnum=negative_audio.shape[0]
        num,sum=0,0
        for i in range(negnum):
            sum=sum+torch.abs(positive_video-negative_audio[i]).sum()/(anchor_audio.shape[0])
            num=num+1
        dist_negative=sum/num
        return dist_positive-dist_negative
        
    # def forward(self, anchor_audio, positive_video, negative_audio):#traincos
    #     """

    #     Args:
    #         anchor_audio: B * C
    #         positive_video: B * C
    #         negative_audio: N * B * C

    #     Returns:

    #     """
    #     #anchor_audio:[100,768],positive_video:[100,768],negative_audio:[50,100,768]，50个负样本,100是时间序列长度
    #     device = anchor_audio.device

    #     positive_video = positive_video.unsqueeze(0)# 将正视频样本增加一个维度，以便与负音频样本进行拼接
    #     contrast_feature = torch.cat([positive_video, negative_audio], dim=0)# 将正视频样本和负音频样本拼接在一起，形成对比特征
        
    #     # compute logits,contrast_feature:[51, 100, 768]
    #     anchor_dot_contrast=torch.sum(anchor_audio.unsqueeze(0)*contrast_feature, dim=-1)/self.temperature#计算锚点音频与对比特征的点积
    #     #anchor_dot_contrast:[51,100]
    #     anchor_dot_contrast = anchor_dot_contrast.permute(1, 0)

    #     mask = torch.zeros_like(anchor_dot_contrast).to(device)# 创建一个与anchor_dot_contrast形状相同的零张量，用于创建掩码
    #     mask[:, 0] = 1.# 将掩码的第一列设置为1，表示正样本
    #     mask = mask.detach()# 从计算图中分离掩码，以避免在反向传播中计算其梯度

    #     loss = self.lossBySimi(anchor_dot_contrast, mask)

    #     return loss

    def lossBySimi(self, similarity, mask):#similarity:[100,51],mask:[100,51],100是一个序列的长度，similartiy里存放的是相似度
        # qprint(similarity[:2,:],'similarity[:2,:]')
        # exit(0)
        # for numerical stability
        logits_max, _ = torch.max(similarity, dim=1, keepdim=True)
        logits = similarity - logits_max.detach()

        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # log_prob=logits-logits.sum(1, keepdim=True)
        
        # compute mean of log-likelihood over positive
        sum_log_prob_pos = (mask * log_prob).sum(1)
        mean_log_prob_pos = sum_log_prob_pos

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        return loss


class av_sync(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(av_sync, self).__init__()

        self.Linearaudio = nn.Linear(in_channel, out_channel)

        self.criterion = SupConLoss()

        self.dropout = nn.Dropout(p=0.25)

        self.maxproportion = 60

    def preprocess_local(self, video, audio, pickedimg):
        video_cl = []
        for i, singlepick in enumerate(pickedimg):
            idimg = pickedimg[i]
            for j in range(len(idimg)):
                video_cl.append(video[idimg[j], i])
        return torch.stack(video_cl, dim=0), audio.view(-1, 512)

    def forward(self, video, audio, pickedimg=None):
        # qprint(audio[0],'audio[0]')
        # qprint(video[0],'video[0]')
        # qprint(torch.norm(audio[0]),'norm(audio[0])')
        # qprint(torch.norm(video[0]),'norm(video[0])')
        # qprint(torch.norm(audio[1]),'norm(audio[0])')
        # qprint(torch.norm(video[1]),'norm(video[0])')
        # qprint(audio.shape,'audio.shape')
        # qprint(video.shape,'video.shape')
        """

        :param video: tensor-> batch*ndim
        :param audio: tensor-> batch*ndim
        :return:
        """
        if pickedimg is not None:
            video, audio = self.preprocess_local(video, audio, pickedimg)

        video_cl_norm = F.normalize(video, dim=1)#标准化

        audio = F.normalize(audio, dim=1)
        audio_cl = self.Linearaudio(audio)
        audio_cl_norm = F.normalize(audio_cl, dim=1)

        n_negatives = min(100, int(video_cl_norm.shape[0]/2)) # 计算负样本的数量，最多100个，或者是视频样本数量的一半
        # generate a audio_cl_norm.shape[0] * n_negatives matrix containing indices.
        neg_idxs = torch.multinomial(torch.ones((audio_cl_norm.shape[0], audio_cl_norm.shape[0] - 1), dtype=torch.float), n_negatives)#生成一个矩阵，包括着音频负样本的索引
        tszs = torch.tensor(list(range(audio_cl_norm.shape[0]))).view(-1, 1) # 创建一个与音频样本数量相等的序列
        neg_idxs[neg_idxs >= tszs] += 1  # 调整负样本索引，确保不与正样本重复

        negs_a = audio_cl_norm[neg_idxs.view(-1)]# 获取负样本的音频数据
        negs_a = negs_a.view(# 重新排列负样本数据的维度
            audio_cl_norm.shape[0], n_negatives, audio_cl_norm.shape[1]
        ).permute(1, 0, 2)  # to NxBxC

        # qprint(audio_cl_norm[0],'audio_cl_norm[0]')
        # qprint(video_cl_norm[0],'video_cl_norm[0]')
        # qprint(torch.norm(audio_cl_norm[0]),'norm(audio_cl_norm[0])')
        # qprint(torch.norm(video_cl_norm[0]),'norm(video_cl_norm[0])')
        # qprint(torch.norm(audio_cl_norm[1]),'norm(audio_cl_norm[0])')
        # qprint(torch.norm(video_cl_norm[1]),'norm(video_cl_norm[0])')
        # qprint(audio_cl_norm.shape,'audio_cl_norm.shape')
        # qprint(video_cl_norm.shape,'video_cl_norm.shape')
        # exit(0)
        dist_loss = self.criterion(audio_cl_norm, video_cl_norm, negs_a)

        return dist_loss

