<div align="center">

# FreqEdit: Preserving High-Frequency Features for Robust Multi-Turn Image Editing

### CVPR 2026

[![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://freqedit.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2512.01755-b31b1b.svg)](https://arxiv.org/abs/2512.01755)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

**FreqEdit** is a training-free framework for robust multi-turn image editing built on flow-matching-based models ([FLUX.1-Kontext](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) and [Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit)). It addresses progressive quality deterioration (body deformations, edge over-sharpening, texture collapse) that these base models exhibit during iterative editing. For technical details, please refer to our [paper](https://arxiv.org/abs/2512.01755).

![Teaser](assets/teaser.png)

## News
* **[2026.03]** Code is released.
* **[2026.03]** FreqEdit is accepted by **CVPR 2026**!
* **[2025.12]** Paper is released on [arXiv](https://arxiv.org/abs/2512.01755).

## Framework
![Framework](assets/framework.png)

## Installation

```bash
git clone https://github.com/FreqEdit/FreqEdit.git
cd FreqEdit
conda create -n freqedit python=3.10 -y
conda activate freqedit
pip install -r requirements.txt
```

## Usage

We provide demo scripts for multi-turn editing on 3 example images with 10 sequential edits each. All scripts are located in `src/`.

**FreqEdit (Ours):**

```bash
cd src

# Based on FLUX.1-Kontext
python run_FreqEditKontext.py

# Based on Qwen-Image-Edit
python run_FreqEditQwen.py
```

**Native Baselines (without FreqEdit):**

```bash
cd src

python run_nativeKontext.py   # FLUX.1-Kontext
python run_nativeQwen.py      # Qwen-Image-Edit
```

Editing parameters can be configured directly in the scripts. For recommended values and detailed descriptions, see the [Parameter Guide](docs/parameters.md).

## Citation

If you find this work useful, please cite:

```bibtex
@misc{liao2026freqeditpreservinghighfrequencyfeatures,
      title={FreqEdit: Preserving High-Frequency Features for Robust Multi-Turn Image Editing},
      author={Yucheng Liao and Jiajun Liang and Kaiqian Cui and Baoquan Zhao and Haoran Xie and Wei Liu and Qing Li and Xudong Mao},
      year={2026},
      eprint={2512.01755},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.01755},
}
```

## Acknowledgements

We sincerely thank the teams behind [FLUX.1-Kontext](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev), [Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit), [Diffusers](https://github.com/huggingface/diffusers), [Multi-turn Consistent Image Editing](https://github.com/ZhouZJ-DL/Multi-turn_Consistent_Image_Editing.git), [FlowEdit](https://github.com/fallenshock/FlowEdit.git), and [VINCIE](https://github.com/ByteDance-Seed/VINCIE.git) for their excellent open-source contributions. Their work has been instrumental in enabling and inspiring this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
