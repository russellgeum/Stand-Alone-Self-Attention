# Stand-Alone Self-Attention
Implementation Stand Alonge Attention  
# Requirements
```
torch >= 1.8.0
```
# Folder 
```
ㄴmodel_layer
    ㄴattention.py
        class StandAloneSelfAttention
        class StandAloneAttention
``` 
# Usage
```
layer   = StandAloneSelfAttention(
    512, 512, kernel_size = (3, 3), stride = 1, padding = 1, heads = 1, dim_head = 512)
tensor  = torch.ones([1, 512, 6, 20])
outputs = layer(tensor)
print(outputs.shape) # [1, 512, 6, 20]
```
# Acknowledgement  
Base SASA repo from @leaderj1001  
repo: https://github.com/leaderj1001/Stand-Alone-Self-Attention  
