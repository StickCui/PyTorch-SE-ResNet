python train.py --trainroot /home/stick/Dataset/place365/train.txt\
    --valroot /home/stick/Dataset/place365/val.txt\
    --num_classes 365\
    --batch_size 64\
    --displayInterval 200\
    --resume_path weights/seresnet101_best.pth.tar\
    --gpu 0,1
