import weightwatcher as ww
from transformers import GPT2Model

gpt2_model = GPT2Model.from_pretrained('gpt2')
gpt2_model.eval()

watcher = ww.WeightWatcher(model=gpt2_model)
details = watcher.analyze(plot=True)
print(details.head())
#    layer_id       name         D  ...      warning weak_rank_loss         xmax        xmin
# 0         2  Embedding  0.076190  ... over-trained              0  3837.188332    0.003564
# 1         8     Conv1D  0.060738  ...                           0  2002.124419  108.881419
# 2         9     Conv1D  0.037382  ...                           2   712.127195   46.092445
# 3        14     Conv1D  0.042383  ...                           0  1772.850274   95.358278
# 4        15     Conv1D  0.062197  ...                           0   626.655218   23.727908
