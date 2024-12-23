# BJTU-UVA: The First Dataset for Automatic Spectral Calibration of Hyperspectral Images

We are proud to introduce **BJTU-UVA**, the **first dataset designed specifically for the task of automatic spectral calibration** of hyperspectral images (HSIs). This dataset addresses the critical challenge of minimizing illumination variability without relying on manual intervention or physical references.

## Key Highlights

- **Task Proposal**:  
  We propose the novel task of **automatic spectral calibration**, aiming to advance the robustness of hyperspectral imaging in diverse real-world scenarios.

- **Dataset Characteristics**:  
  - **Camera**: Specim IQ, featuring a spectral resolution of 3nm across the 400–1000nm range.  
  - **Recording Method**: Each scene is captured twice:  
    1. **Without reference board**: Captures raw scene data.  
    2. **With white reference board**: Records illumination conditions under the same settings.  
  This approach ensures asynchronous yet precise pairing of **uncalibrated** and **calibrated** HSIs, effectively minimizing illumination variability.  
  - **Dark Current Correction**: Dark current noise, intrinsic to the camera sensor, is carefully recorded and subtracted during post-processing, ensuring high data accuracy.

- **Scene Diversity**:  
  The dataset encompasses a wide range of **urban and natural scenes**, captured under various weather conditions, lighting scenarios, and times of day.

- **Benchmarking Standard**:  
  BJTU-UVA establishes a new standard for spectral calibration by combining real-world scene variability with rigorous illumination recording, offering a robust foundation for testing and advancing spectral calibration techniques.

## Dataset Splits

The file `real_light_split.json` provides predefined **'train'**, **'validation'**, and **'test'** splits of the dataset, enabling standardized evaluation and comparison across different methods.

## Synthetic Data Generation

The script `gen_sim_data.py` can be used to generate the **BJTU-UVA-E** dataset, a synthetic extension of BJTU-UVA, by simulating 10 different illumination conditions recorded in `ill_sim.pkl`. This allows researchers to explore and benchmark their methods under controlled, yet diverse, illumination variations.

## Download

You can access the dataset via the following link:  
**[BaiduNetDisk](https://pan.baidu.com/s/1NawnIBN3ixH_qec70zenLQ)**  
**Password**: `pgkg`

## Citation
@misc{du2024spectral,<br>
      title={Automatic Spectral Calibration of Hyperspectral Images: Method, Dataset and Benchmark}, <br>
      author={Zhuoran Du and Shaodi You and Cheng Cheng and Shikui Wei},<br>
      year={2024},<br>
      eprint={2412.14925},<br>
      archivePrefix={arXiv},<br>
      primaryClass={cs.CV},<br>
      url={ https://arxiv.org/abs/2412.14925 }, <br>
}

We hope BJTU-UVA inspires new research and development in the field of hyperspectral imaging. If you use this dataset, please consider citing our work.
