# Collaborative Filtering | BUIR

## Run

**Store ml-100k dataset under `data`**

Training:

```bash
python3 train.py
```

Options for Training:
```bash
python3 train.py --exp_name [experiment name] --exp_disc [experiment discription] --model [type of model used]
                 --latent_size [latent embeddings size] --epochs [num of epochs] --lr [learning rate]
                 --weight_decay [weight decay] --batch_size [] --momentum [] --train_ratio [train-test split ratio]
                 --num_workers [workers for dataLoader] --cold_start [flag for performing cold start]
                 --cold_start_clusters [num of cluster in cold start kmeans]
```

Output logs and Plots will be saved in ```./experiments/{exp_name}```
