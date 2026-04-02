# Custom Demos

Demos for custom stereo pairs, including disparity, uncertainty, and point cloud visualization. For high-resolution inputs (>1080p), we recommend downsampling the images by at least 2× for optimal performance. This file is largely adapted from [FoundationStereo](https://github.com/NVlabs/FoundationStereo/blob/master/scripts/run_demo.py). Please comply with the NVIDIA license if you plan to use it.

```bash
python demo.py --config-file configs/SynLarge/DAv2L-5.yaml --ckpt ckpts/SynLarge/DAv2L-5.pth
```