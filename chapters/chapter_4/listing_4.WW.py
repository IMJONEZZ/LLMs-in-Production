import weightwatcher as ww
from transformers import GPT2Model

gpt2_model = GPT2Model.from_pretrained("gpt2")
gpt2_model.eval()

watcher = ww.WeightWatcher(model=gpt2_model)
details = watcher.analyze(plot=False)
print(details.head())
#    layer_id       name         D  ...      warning        xmax        xmin
# 0         2  Embedding  0.076190  ... over-trained 3837.188332    0.003564
# 1         8     Conv1D  0.060738  ...              2002.124419  108.881419
# 2         9     Conv1D  0.037382  ...               712.127195   46.092445
# 3        14     Conv1D  0.042383  ...              1772.850274   95.358278
# 4        15     Conv1D  0.062197  ...               626.655218   23.727908
