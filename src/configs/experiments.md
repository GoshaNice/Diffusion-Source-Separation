# Experiments Details

There are 3 main parts:

Attention part:
1. No attention
2. Only self-attention
3. Self-attention + post-CNN &larr; __main__

Conditioning-part:
1. No conditioning
2. Local Conditioning
3. Global Conditioning &larr; __main__

Freezing:
1. Freeze diffusion model &larr; __main__
2. Freeze all

During Ablation study we subsequently in each part conduct experiments on how different option influence the final metrics.

| Experiment | Part         | Chosen option          |
| ---------- | ------------ | ---------------------- |
| main       | -            | above                  |
| 1          | Attention    | No attention           |
| 2          | Attention    | Only self-attention    |
| 3          | Conditioning | No conditioning        |
| 4          | Conditioning | Local Conditioning     |
| 5          | Freezing     | Freeze all             |

