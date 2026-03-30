# Parameter Guide

## Common Parameters (Kontext & Qwen)

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `base_alpha` | Base strength for frequency preservation. Higher values lead to stronger preservation. | 1.2 - 2.0 |
| `gamma` | Sensitivity of adaptive weighting. Higher values lead to more aggressive adaptation. | 1.5 - 2.0 |
| `levels` | Wavelet decomposition depth. | 2 |
| `wavelet_mode` | Wavelet basis function. | `"db4"` |
| `Inject_time_scale` | Ratio of timesteps for frequency guidance injection. | 0.3 |
| `backward_steps` | Period for path compensation. | 4 |

## Kontext-Only Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `QG_time_scale` | Ratio of timesteps for quality guidance. | 0.7 |
| `QG_lambda` | Quality guidance strength. | 0.3 |

## Script Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `idx` | Select example image (1-3). | 3 |
| `start_edit_turn` | Starting turn number. | 0 |
| `final_edit_turn` | Final turn number. | 10 |

For more details on the design and effect of each parameter, please refer to our [paper](https://arxiv.org/abs/2512.01755).
