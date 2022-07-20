#conda activate meantjo2

CUDA_VISIBLE_DEVICES=0 python ../main_fer.py --exp 'SOFT 0.3 sym' --noise 0.3 --ls2 --eps2 0.3 --k -1  --function softmax --checkpoint-epochs 0   --epochs 80 --consistency 10  --consistency-rampup 5  --lr 0.0001 --batch-size 64 --dataset raf --pretrained_facedb ms-celeb

