# FAST Data Processing Toolkit

Welcome to the **FAST Data Processing Toolkit**! This repository provides a comprehensive suite of Python scripts designed to streamline and enhance the data reduction workflow for the Five-hundred-meter Aperture Spherical radio Telescope (FAST).

Built to complement the `hifast` pipeline, these tools cover the entire lifecycle of radio astronomy data processing—from raw FITS data grouping and preprocessing, to telescope tracking verification, chunk merging, and final scientific visualization (Spectra and Moment 0 maps).

## Workflow & Tool Index

This toolkit is structured around a standard FAST data reduction workflow:

### 1. Data Preprocessing & Formatting
* **`transformation.py`**: Scans raw chunked FITS files, intelligently groups them by Beam ID, sorts them chronologically, and converts them into `hifast`-compatible HDF5 formats.

### 2. Quality Control & Observation Verification
* **`RA-DEC_total.py`**: Extracts telescope pointing coordinates, filters out unstable slewing periods, and visualizes the actual RA/DEC drift tracking paths to ensure observation accuracy.
* **`RMS_analysis.py`**: Iterates through FITS datacubes to calculate per-channel RMS noise, automatically flagging anomalous channels (RFI) using a robust statistical threshold.

### 3. Data Merging
* **`merge.py`**: Safely concatenates time-chunked HDF5 segments back into continuous, full-beam files. It features dynamic memory resizing and strict preservation of `hifast` soft-links (crucial for Carta compatibility).

### 4. Scientific Visualization & Extraction
* **`my_analysis.py`**: A smart batch-plotter that uses a 1D convolution algorithm to dynamically locate the "cleanest" continuous time slices, extracting and plotting high-quality mean spectra while bypassing RFI.
* **`moment0.py`**: Computes 2D Integrated Intensity (Moment 0) maps from 3D FITS datacubes. Fully optimized for headless servers, supporting flexible spectral slicing and advanced visual contrast stretches (Log, Sqrt, Square, Exp).

## Installation & Dependencies

Ensure you have the `hifast` pipeline installed in your environment.

# FAST 射电天文数据处理工具箱 (Project Introduction)

欢迎使用 FAST 射电天文数据处理工具箱！本仓库提供了一整套基于 Python 的自动化脚本，旨在简化并增强 FAST (500米口径球面射电望远镜) 观测数据的后处理工作流。

这套工具链契合 `hifast` 处理管线，覆盖了射电天文数据处理的全生命周期——从原始 FITS 碎片的智能分组与格式转换，到望远镜轨迹核查、HDF5 分块无损合并，再到最终的科学级数据可视化（高分辨频谱提取与积分强度图生成）。

---

## 工作流与核心工具索引

本工具箱按照标准的 FAST 数据处理逻辑划分为以下四个核心模块：

### 1. 数据预处理与格式转换
* **`transformation.py`**: 自动扫描原始 FITS 碎片文件，按波束 (Beam) 智能分组、按时间顺序排列，并转化为完全兼容 `hifast` 格式的 HDF5 文件。

### 2. 数据质量控制与观测核查
* **`RA-DEC_total.py`**: 提取望远镜指向数据，智能剔除转场或不稳定的扫描片段，将实际的 RA/DEC 漂移轨迹合并绘制在同一张高对比度图表中。
* **`RMS_analysis.py`**: 遍历 FITS 数据立方体计算逐通道的均方根 (RMS) 噪声，并利用统计阈值自动检测和标记受射频干扰 (RFI) 污染的异常通道。

### 3. 分块数据合并
* **`merge.py`**: 将分块处理后的 HDF5 切片安全、无损地拼接回完整的单波束文件。支持动态扩容写入，并完美保留软链接（这对 Carta 等高级可视化软件至关重要）。

### 4. 科学级可视化与特征提取
* **`my_analysis.py`**: 智能频谱提取与批量绘图工具。采用一维卷积算法动态寻找“最干净”的连续时间切片，完美避开 RFI 密集区，提取高质量平均频谱并应用高斯平滑。
* **`moment0.py`**: 从 3D FITS 数据立方体生成 2D 积分强度图 (Moment 0)。专为无头服务器环境优化，支持灵活的光谱/通道切片及多种非线性视觉拉伸算法（对数、开方、平方、指数）。

---

## 环境配置

> **⚠️ 注意：** > 请确保你的运行环境中已经成功安装了 `hifast` 管线以及其他诸如numpy scipy pandas h5py astropy matplotlib等库。
