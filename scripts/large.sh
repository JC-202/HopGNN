python train.py --cuda_id 0 --dataset Flickr --lr 0.005 --weight_decay 0 --dropout 0.2 --hidden 256 --epochs 300
python train.py --cuda_id 0 --dataset Reddit --lr 0.005 --weight_decay 0 --dropout 0.2 --hidden 256 --epochs 300
python train.py --cuda_id 0 --dataset products --lr 0.001 --weight_decay 5e-6 --dropout 0.5 --hidden 256 --epochs 500