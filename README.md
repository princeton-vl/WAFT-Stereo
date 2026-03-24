# WAFT-Stereo

<!-- [[Paper](https://arxiv.org/abs/2506.21526v2)] -->

  We introduce WAFT-Stereo, a simple and effective warping-based algorithm for stereo matching. WAFT-Stereo demonstrates that cost volumes, a common design used in many leading methods, are not necessary for strong performance. WAFT-Stereo ranks first on ETH3D, KITTI and Middlebury public benchmarks, reducing the zero-shot error by 81\% on ETH3D benchmark, while being 1.8-6.7x faster than competitive methods.

![demo](assets/Concat.gif)

<!-- If you find WAFT-Stereo useful for your work, please consider citing our academic paper:

<h3 align="center">
    <a href="https://arxiv.org/abs/2506.21526v2">
        WAFT-Stereo: Warping-Alone Field Transforms for Stereo Matching
    </a>
</h3>
<p align="center">
    <a href="https://memoryslices.github.io/">Yihan Wang</a>,
    <a href="http://www.cs.princeton.edu/~jiadeng">Jia Deng</a><br>
</p>

```
@article{wang2025waft,
  title={WAFT: Warping-Alone Field Transforms for Optical Flow},
  author={Wang, Yihan and Deng, Jia},
  journal={arXiv preprint arXiv:2506.21526},
  year={2025}
}
``` -->

## Installation

```bash
conda create --name waft-stereo python=3.12
conda activate waft-stereo
pip install -r requirements.txt
```

Please also install [xformers](https://github.com/facebookresearch/xformers) following instructions.

## Configs

All experiments are driven by config files in `configs/`.

Current examples:

- Synthetic training:
  - `configs/SynLarge/DAv2S-4.yaml`
  - `configs/SynLarge/DAv2B-4.yaml`
  - `configs/SynLarge/DAv2L-5.yaml`
- Real-domain finetuning:
  - `configs/Real/kitti.yaml`
  - `configs/Real/middlebury.yaml`
- Evaluation:
  - `configs/eval/kitti2012.yaml`
  - `configs/eval/kitti2015.yaml`
  - `configs/eval/eth3d.yaml`
  - `configs/eval/middlebury-Q.yaml`

## Datasets

Please check [README for datasets preparation](bridgedepth/dataloader/README).

## Weights (on Huggingface)

Please check the [link](https://huggingface.co/MemorySlices/WAFT-Stereo/tree/main). We recommend using the checkpoints in the 'SynLarge' folder for downstream applications due to their strong sim-to-real generalization. Several of our main results and demos are based on these checkpoints. For more details, please refer to the paper.

## Training

```bash
python main.py --num-gpus 8 --config-file configs/SynLarge/DAv2L-5.yaml
```

## Evaluation

```bash
python main.py --num-gpus 1 --eval-only --config-file configs/eval/kitti2015.yaml --ckpt ckpts/SynLarge/DAv2L-5.pth
```

Validation metrics are computed through `bridgedepth.utils.eval_disp`.

## Submissions

```bash
python submission.py --config-file configs/eval/eth3d.yaml --ckpt ckpts/SynLarge/DAv2L-5.pth --dataset eth3d --output eth3d_submission/
```

## Profiling

Use `profiler.py` to inspect parameter count, forward MACs, and memory usage:

```bash
python profiler.py --config-file configs/SynLarge/DAv2L-5.yaml
```

## Visualization

Qualitative visualization:

```bash
python visualize.py --config-file configs/SynLarge/DAv2L-5.yaml
```

Dataset inspection:

```bash
python view_dataset.py --dataset fsd
```

## Acknowledgments

This project relies on code from existing repositories: [BridgeDepth](https://github.com/aeolusguan/BridgeDepth), [FoundationStereo](https://github.com/NVlabs/FoundationStereo), [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo), [MonSter++](https://github.com/Junda24/MonSter-plusplus), [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) and [DINOv3](https://github.com/facebookresearch/dinov3). We thank the original authors for their excellent work.