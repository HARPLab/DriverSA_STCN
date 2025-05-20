Activate anaconda

Befpre running train_awareness.py
```
export LD_LIBRARY_PATH=/home/srkhuran-local/anaconda3/envs/stcn_env/lib/:$LD_LIBRARY_PATH
```

Run train_awareness.py 
```
python train_awareness.py --gaze-fade --batch_size 16
```