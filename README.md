# FAST Data Processing Toolkit

Welcome to the **FAST Data Processing Toolkit**! This repository provides a comprehensive suite of Python scripts and Jupyter Notebooks designed to streamline and enhance the data reduction and analysis workflow for the Five-hundred-meter Aperture Spherical radio Telescope (FAST).

Built to complement the `hifast` pipeline and the `SoFiA` (Source Finding Application) software, these tools cover the entire lifecycle of radio astronomy data processing—from raw FITS data preprocessing to final automated source cross-matching with local galaxy surveys.

## Workflow & Tool Index

This toolkit is structured around a standard FAST data reduction and analysis workflow:

### 1. Data Preprocessing & Formatting
* **`transformation.py`**: Scans raw chunked FITS files, intelligently groups them by Beam ID, and converts them into `hifast`-compatible HDF5 formats.

### 2. Quality Control & Observation Verification
* **`RA-DEC_total.py`**: Visualizes the actual RA/DEC drift tracking paths to ensure observation accuracy.
* **`RMS_analysis.py`**: Calculates per-channel RMS noise and flags RFI anomalous channels.
* **`path.ipynb`**: A lightweight notebook for inspecting FITS WCS information and plotting RA/DEC drift paths directly from intermediate HDF5 files.
* **`RA-DEC-RMS.py`**: Calculates the **spatial** distribution of RMS noise across the FITS datacube. It generates 2D noise maps in RA/Dec coordinates, helping to identify direction-dependent interference or beam edge effects.
### 3. Data Merging
* **`merge.py`**: Safely concatenates time-chunked HDF5 segments into full-beam files while strictly preserving `hifast` soft-links (crucial for **Carta** compatibility).

### 4. Scientific Visualization & Extraction
* **`my_analysis.py`**: A smart batch-plotter that uses a 1D convolution algorithm to dynamically locate the "cleanest" time slices and extract high-quality spectra.
* **`moment0.py`**: Computes 2D Integrated Intensity (Moment 0) maps with support for advanced visual contrast stretches (Log, Sqrt, Square, Exp).

### 5. Source Finding Analysis & GAMA Cross-Matching
* **`source.py`**: Generates spatial distribution plots (RA vs. Dec) of detected HI sources, scaling points by SNR and coloring by frequency.
* **`check_source.ipynb`**: An interactive tool for quickly scanning 1D spectra (`_spec.txt`) of hundreds of SoFiA-detected sources via a slider interface.
* **`GAMA_search.ipynb` (Local GAMA Match)**: An automated high-speed cross-matching tool. It calculates HI redshift ($z_{HI}$) and matches candidates against the **local GAMA DR4 (Galaxy And Mass Assembly)** database using `search_around_sky` spatial indexing, enforcing a strict redshift tolerance ($\Delta z < 0.002$).

### 6.Baseline Refinement
* **`iterative_hifast_bld.py`**: Performs a non-destructive second-pass iterative baseline subtraction for HiFAST `sw_nobld.hdf5` products, dynamically protecting potential HI line channels during fitting and writing the cleaned results to a new output file.

## Installation & Dependencies

Ensure you have the `hifast` pipeline installed. Then, install the required libraries:

pip install numpy scipy pandas h5py astropy matplotlib ipywidgets notebook

# FAST 射电天文数据处理工具箱 (Project Introduction)

欢迎使用 FAST 射电天文数据处理工具箱！本仓库提供了一整套基于 Python 的自动化脚本和 Jupyter Notebook，旨在简化并增强 FAST (500米口径球面射电望远镜) 观测数据的后处理与分析工作流。

这套工具链完美契合 `hifast` 处理管线以及 `SoFiA` (自动寻源软件)，覆盖了射电天文数据处理的全生命周期——从原始 FITS 碎片的智能分组与格式转换，到望远镜轨迹核查、HDF5 分块无损合并，再到科学级数据可视化以及最终的寻源结果自动化交叉匹配。

## 工作流与核心工具索引

本工具箱按照标准的 FAST 数据处理与寻源分析逻辑划分为以下五个模块：

### 1. 数据预处理与格式转换
* **`transformation.py`**: 自动扫描原始 FITS 碎片文件，按波束 (Beam) 智能分组、按时间顺序排列，并转化为完全兼容 `hifast` 格式的 HDF5 文件。

### 2. 数据质量控制与观测核查
* **`RA-DEC_total.py`**: 提取望远镜指向数据，智能剔除转场或不稳定的扫描片段，将实际的 RA/DEC 漂移轨迹合并绘制在同一张高对比度图表中。
* **`RMS_analysis.py`**: 遍历 FITS 数据立方体计算逐通道的均方根 (RMS) 噪声，并利用统计阈值自动检测和标记受射频干扰 (RFI) 污染的异常通道。
* **`path.ipynb`**: 轻量级实用工具本。用于快速检查 FITS 头文件中的 WCS 像素尺度（度/角分），并支持直接从 HDF5 过程文件中提取和绘制 RA/DEC 实际指向轨迹。
* **`RA-DEC-RMS.py`**: 计算 FITS 数据立方体的空间 RMS 噪声分布。生成基于 RA/Dec 坐标的 2D 噪声图，用于识别与方向相关的干扰或波束边缘效应。

### 3. 分块数据合并
* **`merge.py`**: 将分块处理后的 HDF5 切片安全、无损地拼接回完整的单波束文件。支持动态扩容写入，并完美保留软链接（这对 Carta 等高级可视化软件至关重要）。

### 4. 科学级可视化与特征提取
* **`my_analysis.py`**: 智能频谱提取与批量绘图工具。采用一维卷积算法动态寻找“最干净”的连续时间切片，完美避开 RFI 密集区，提取高质量平均频谱并应用高斯平滑。
* **`moment0.py`**: 从 3D FITS 数据立方体生成 2D 积分强度图 (Moment 0)。专为无头服务器环境优化，支持灵活的光谱/通道切片及多种非线性视觉拉伸算法（对数、开方、平方、指数）。

### 5. 自动寻源分析与交叉匹配 (SoFiA 结果集成)
* **`source.py`**: 读取 SoFiA 寻源结果目录，绘制空间分布散点图 (RA vs. Dec)。自动根据信噪比 (SNR) 调整点的大小，以颜色映射观测频率，并自动标注出信噪比最高的前 10 个候选源。
* **`check_source.ipynb`**: 基于 `ipywidgets` 的交互式 Jupyter Notebook，用于快速浏览 SoFiA 生成的每一个源的 1D 频谱 (`_spec.txt`)。通过拖动滑动条即可无缝切换查看数百个源的积分通量，极大提升人工核查效率。
* **`GAMA_search.ipynb`**: 自动化数据库交叉匹配工具。根据观测频率计算中性氢红移 ($z_{HI}$)，自动批量连接本地GAMA DR4 数据库。在指定的搜索半径和红移误差范围内寻找对应天体，并输出直观的匹配结果汇总表。

### 6. 二次去基线
* **`iterative_hifast_bld.py`**: 针对 HiFAST 的 `sw_nobld.hdf5` 中间产物执行非破坏性的二次迭代去基线，在拟合过程中动态保护可能的 HI 谱线通道，并将处理结果写入新的输出文件。

## 环境配置

> **⚠️ 注意：** > 请确保你的运行环境中已经成功安装了 `hifast` 管线以及其他诸如numpy scipy pandas h5py astropy matplotlib等库。
> 在使用交叉匹配脚本 (GAMA_search.ipynb) 之前，你必须预先下载 GAMA DR4 光谱天体表。
下载地址： [GAMA Survey DR4 官方入口](http://www.gama-survey.org/dr4/)
配置说明： 请将脚本中 gama_local_path 变量的值修改为你本地 CSV 文件的存储路径。
