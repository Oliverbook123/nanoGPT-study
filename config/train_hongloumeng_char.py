# 训练一个迷你字符级《红楼梦》模型（中文，120 回，约 90 万汉字）
#
# 先准备数据：python data/hongloumeng/prepare.py
# 然后：      python train.py config/train_hongloumeng_char.py
#
# 超参照搬 shakespeare_char 那套（char-level 小模型），只改了 dataset / out_dir / wandb。
# 注意汉字字符表比英文大得多（约 4000–5000 vs 65），所以词嵌入和输出层更占显存，
# 显存不够就调小 batch_size 或加 gradient_accumulation_steps。

out_dir = 'out-hongloumeng-char'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'hongloumeng-char'
wandb_run_name = 'mini-gpt'

dataset = 'hongloumeng'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters

# baby GPT :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
# device = 'mps'   # or 'cpu'
# compile = False  # do not torch compile the model
# （script/train.sh 会自动补上这两个参数）
