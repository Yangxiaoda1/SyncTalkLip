# SyncTalkLip
Before you start the project, you need to familiarize yourself with these two projects：  
1.Follow [TalkLip/README.md at main · Sxjdwang/TalkLip (github.com)](https://github.com/Sxjdwang/TalkLip/tree/main), run the project.  
2.Follow https://github.com/facebookresearch/av_hubert.git, run the project  
Running in the Project 2 environment：![image](https://github.com/user-attachments/assets/c5ec93d6-f4ed-4d47-bd75-c3da57061500)
Train:  
python train.py \
--file_dir /root/data/lrs2 \
--video_root /root/data/lrs2/main \
--audio_root /root/data/lrs2/audio/main \
--bbx_root /root/bbox/main \
--word_root /root/data/lrs2/main \
--avhubert_root /root/syncavhubert \
--avhubert_path /root/finetune/checkpoints/checkpoint_best.pt \
--checkpoint_dir /root/syncavhubert \
--log_name logs \
--disc_checkpoint_path /root/checkpoint/vis_dis.pth  
Inference:  
python inf_test.py \
--filelist /root/data/lrs2/test.txt \
--video_root /root/data/lrs2/main \
--audio_root /root/data/lrs2/audio/main \
--bbx_root /root/bbox/main \
--save_root {} \
--ckpt_path {} \
--avhubert_root {}  
You may need to change to the corresponding path and modify the corresponding package according to the actual situation.![image](https://github.com/user-attachments/assets/94b25c23-94c5-4f96-b757-5e8dc6ca41d4)
