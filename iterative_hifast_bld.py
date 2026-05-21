#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FAST/HiFAST 二次去基线脚本（面向 sw_nobld.hdf5）。

核心思想：
1) 读取 HiFAST 中间文件中的频谱数据与 RFI/排除掩膜；
2) 对每条时序频谱做“迭代鲁棒多项式拟合”；
3) 每一轮根据残差的 MAD 估计动态扩展谱线掩膜，避免真实 HI 信号被当作基线；
4) 收敛后输出 baseline-subtracted 结果到新文件，绝不覆盖输入文件；
5) 支持 checkpoint 断点续跑、日志追踪、前后对比图输出。

示例：
    python iterative_hifast_bld.py \
      --input /path/to/*sw_nobld.hdf5 \
      --output /path/to/output/new_iterbld.hdf5 \
      --poly-order 3 --max-iter 10 --pos-sigma 3.0 --grow-chan 8
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import h5py
import matplotlib
import numpy as np
from astropy.wcs import WCS
from scipy import sparse
from scipy.ndimage import binary_dilation, gaussian_filter1d
from scipy.sparse.linalg import spsolve

# 采用无交互后端，保证在服务器/终端环境下也能绘图。
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REST_FREQ_HI_MHZ = 1420.40575177


@dataclass
class BaselineConfig:
    """二次去基线的核心参数集合。"""

    method: str
    poly_order: int
    max_iter: int
    pos_sigma: float
    neg_sigma: float
    grow_chan: int
    detect_absorption: bool
    min_valid_frac: float
    max_line_frac: float
    asls_lambda: float
    asls_p: float
    asls_masked_weight: float


def build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器，方便在 macOS/Linux 环境中批处理调用。"""
    parser = argparse.ArgumentParser(
        description="对 HiFAST sw_nobld 数据做可迭代二次去基线（非破坏性输出）"
    )
    parser.add_argument("--input", help="输入 HDF5 文件路径（只读）")
    parser.add_argument(
        "--output",
        help="输出 HDF5 文件路径（新文件，不覆盖输入文件）",
    )
    parser.add_argument(
        "--batch-dir",
        default=None,
        help="批处理目录：若设置，则批量处理目录内匹配文件（与 --input/--output 二选一）",
    )
    parser.add_argument(
        "--batch-pattern",
        default="Dec-0011_09_05_arcdrift-M*_W_*_specs_T-flux-bld_p-rfi-sw_nobld.hdf5",
        help="批处理输入匹配模式（glob）",
    )
    parser.add_argument(
        "--batch-output-dir",
        default=None,
        help="批处理输出目录，默认与 --batch-dir 相同",
    )
    parser.add_argument(
        "--batch-output-suffix",
        default="-iterbld-v4",
        help="批处理输出文件名后缀（加在 .hdf5 之前）",
    )
    parser.add_argument(
        "--batch-workers",
        type=int,
        default=1,
        help="批处理并行工作数（按文件级并行）；1=串行，0=自动按CPU核数分配",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=128,
        help="按时间谱线分块处理的块大小，降低内存占用",
    )
    parser.add_argument(
        "--start-spec",
        type=int,
        default=0,
        help="处理起始时间谱索引（含）",
    )
    parser.add_argument(
        "--stop-spec",
        type=int,
        default=-1,
        help="处理结束时间谱索引（不含），-1 表示到末尾",
    )
    parser.add_argument("--poly-order", type=int, default=3, help="多项式阶数")
    parser.add_argument(
        "--method",
        choices=["poly", "asls"],
        default="asls",
        help="基线拟合方法：poly（多项式）或 asls（非对称最小二乘）",
    )
    parser.add_argument(
        "--asls-lambda",
        type=float,
        default=1e6,
        help="AsLS 平滑参数 lambda，越大越平滑",
    )
    parser.add_argument(
        "--asls-p",
        type=float,
        default=0.01,
        help="AsLS 非对称权重参数 p（常用 0.001~0.05）",
    )
    parser.add_argument(
        "--asls-masked-weight",
        type=float,
        default=1e-6,
        help="AsLS 对掩膜通道赋予的极小权重，抑制其对拟合的影响",
    )
    parser.add_argument("--max-iter", type=int, default=10, help="每条谱线最大迭代次数")
    parser.add_argument(
        "--pos-sigma",
        type=float,
        default=3.0,
        help="正残差阈值（单位 sigma），用于保护发射线",
    )
    parser.add_argument(
        "--neg-sigma",
        type=float,
        default=4.0,
        help="负残差阈值（单位 sigma），用于可选吸收线保护",
    )
    parser.add_argument(
        "--grow-chan",
        type=int,
        default=6,
        help="对检测到的谱线掩膜做通道方向膨胀，保护线翼",
    )
    parser.add_argument(
        "--detect-absorption",
        action="store_true",
        help="启用后同时保护负残差吸收特征（默认仅保护正残差）",
    )
    parser.add_argument(
        "--min-valid-frac",
        type=float,
        default=0.10,
        help="单条谱线用于拟合的最小有效通道占比，过低时降级为常数基线",
    )
    parser.add_argument(
        "--max-line-frac",
        type=float,
        default=0.25,
        help="单轮新识别谱线掩膜占有效通道上限，防止掩膜迭代失控",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="若存在 checkpoint，则从断点继续处理",
    )
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="在输出文件中额外保存拟合基线数据集 S/baseline_iter2",
    )
    parser.add_argument(
        "--cleanup-sidecars",
        action="store_true",
        help="运行成功后删除同名 *_iterbld.checkpoint.json / *_iterbld.log / *_spectral_wcs.txt",
    )
    parser.add_argument(
        "--batch-keep-sidecars",
        action="store_true",
        help="批处理模式下保留 sidecar 文件；默认批处理会自动清理 sidecar",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="输出处理前后对比图（学术风格）",
    )
    parser.add_argument(
        "--plot-waterfall",
        action="store_true",
        help="输出处理结果瀑布图（After only, 全时序范围）",
    )
    parser.add_argument(
        "--plot-pol",
        type=int,
        default=0,
        help="绘图所用极化索引",
    )
    parser.add_argument(
        "--plot-start",
        type=int,
        default=-1,
        help="绘图时间起始索引，-1 表示自动选择中段",
    )
    parser.add_argument(
        "--plot-width",
        type=int,
        default=256,
        help="绘图时在时间方向做中位数叠加的宽度",
    )
    parser.add_argument(
        "--plot-window-mode",
        choices=["auto", "center"],
        default="auto",
        help=(
            "当 --plot-start < 0 时的自动选窗策略："
            "auto=优先选择 after 有效点最多的窗口；center=使用中段窗口。"
        ),
    )
    parser.add_argument(
        "--vel-def",
        choices=["optical", "radio", "relativistic"],
        default="optical",
        help="频率到速度换算定义，用于图像坐标轴标注",
    )
    parser.add_argument(
        "--disable-common-mode",
        action="store_true",
        help="关闭后处理共模基线扣除（默认开启）",
    )
    parser.add_argument(
        "--common-mode-sigma",
        type=float,
        default=150.0,
        help="共模基线在频率方向高斯平滑的 sigma（通道数）",
    )
    parser.add_argument(
        "--local-ripple-sigma",
        type=float,
        default=0.0,
        help="逐条谱线局部波纹扣除高斯 sigma（通道数），0 表示关闭",
    )
    parser.add_argument(
        "--masked-fill",
        choices=["nan", "zero", "interp", "keep"],
        default="nan",
        help=(
            "输出时对输入非有限值通道（NaN/Inf）的填充值策略："
            "nan=置NaN；zero=置0；interp=频率向插值；keep=保持原值。"
            "注：is_rfi 将在拟合后单独强制置为 NaN。"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="若输出文件已存在，允许继续写入（建议配合 --resume 使用）",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
    )
    return parser


def setup_logger(log_path: Path, level: str) -> logging.Logger:
    """初始化文件+终端双通道日志，保留处理链路可追溯信息。"""
    logger = logging.getLogger("iterative_hifast_bld")
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def ensure_non_destructive_io(input_path: Path, output_path: Path, force: bool) -> None:
    """执行 I/O 安全检查，确保不会覆盖原始输入数据。"""
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("输出路径与输入路径相同，这会破坏原始数据，已阻止。")
    if output_path.exists() and not force:
        raise FileExistsError(
            f"输出文件已存在: {output_path}。如需继续，请显式添加 --force。"
        )


def copy_attrs(src_obj: h5py.HLObject, dst_obj: h5py.HLObject) -> None:
    """复制 HDF5 属性，保留数据来源与管线元信息。"""
    for key, value in src_obj.attrs.items():
        dst_obj.attrs[key] = value


def create_empty_dataset_like(src: h5py.Dataset, dst_group: h5py.Group, name: str) -> h5py.Dataset:
    """按源数据集的结构创建空数据集，用于后续分块写入处理结果。"""
    kwargs: Dict[str, object] = {}
    if src.chunks is not None:
        kwargs["chunks"] = src.chunks
    if src.compression is not None:
        kwargs["compression"] = src.compression
    if src.compression_opts is not None:
        kwargs["compression_opts"] = src.compression_opts
    if src.shuffle:
        kwargs["shuffle"] = src.shuffle
    if src.fletcher32:
        kwargs["fletcher32"] = src.fletcher32

    dst = dst_group.create_dataset(name, shape=src.shape, dtype=src.dtype, **kwargs)
    copy_attrs(src, dst)
    return dst


def copy_hdf5_skeleton(
    fin: h5py.File,
    fout: h5py.File,
    skip_paths: Iterable[str],
) -> None:
    """
    复制输入文件结构到输出文件。

    对 skip_paths 中的数据集仅创建空壳，避免先复制大数组再覆盖，节省 I/O。
    """
    skip_set = set(skip_paths)
    copy_attrs(fin, fout)

    def _copy_group(src_group: h5py.Group, dst_group: h5py.Group) -> None:
        copy_attrs(src_group, dst_group)
        for key, obj in src_group.items():
            rel_path = obj.name.lstrip("/")
            if isinstance(obj, h5py.Group):
                new_group = dst_group.create_group(key)
                _copy_group(obj, new_group)
            else:
                if rel_path in skip_set:
                    create_empty_dataset_like(obj, dst_group, key)
                else:
                    src_group.file.copy(obj, dst_group, name=key)

    for top_key, top_obj in fin.items():
        if isinstance(top_obj, h5py.Group):
            top_dst = fout.create_group(top_key)
            _copy_group(top_obj, top_dst)
        else:
            rel_path = top_obj.name.lstrip("/")
            if rel_path in skip_set:
                create_empty_dataset_like(top_obj, fout, top_key)
            else:
                fin.copy(top_obj, fout, name=top_key)


def robust_sigma_mad(residual: np.ndarray, valid_mask: np.ndarray) -> float:
    """使用 MAD 估计噪声尺度，对离群值与残余 RFI 更稳健。"""
    vals = residual[valid_mask]
    if vals.size == 0:
        return float("nan")
    med = np.nanmedian(vals)
    mad = np.nanmedian(np.abs(vals - med))
    return 1.4826 * mad


def masked_gaussian_smooth1d(
    values: np.ndarray,
    bad_mask: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """
    对 1D 序列做“掩膜感知”的高斯平滑。

    物理动机：
    - RFI/源线通道不应参与局部波纹（standing-wave 残差）估计；
    - 否则会把强干扰通过卷积核扩散到邻近干净通道，造成“被 mask 的 RFI 反向污染结果”。

    实现方法：
    - 分子：仅对未屏蔽点做高斯卷积；
    - 分母：对有效权重做同样卷积；
    - 最终取分子/分母，得到忽略 bad_mask 的平滑估计。
    """
    y = np.asarray(values, dtype=np.float64)
    bad = np.asarray(bad_mask, dtype=bool) | (~np.isfinite(y))

    # 全部无效时返回零模板，表示不做局部波纹扣除。
    if np.all(bad):
        return np.zeros_like(y, dtype=np.float64)

    w = (~bad).astype(np.float64)
    y0 = np.where(bad, 0.0, y)

    num = gaussian_filter1d(y0, sigma=float(sigma), mode="nearest")
    den = gaussian_filter1d(w, sigma=float(sigma), mode="nearest")

    out = np.zeros_like(y, dtype=np.float64)
    good = den > 1e-12
    out[good] = num[good] / den[good]

    # 极端情况下（长段连续掩膜）做邻近插值，避免局部空洞。
    if np.any(~good):
        idx = np.arange(y.size)
        if np.any(good):
            out[~good] = np.interp(idx[~good], idx[good], out[good])
        else:
            out[~good] = 0.0
    return out


def fill_masked_channels_1d(
    values: np.ndarray,
    input_mask: np.ndarray,
    mode: str,
) -> np.ndarray:
    """
    按策略回填输入掩膜通道。

    注意：
    - 该函数只根据调用方提供的 input_mask 生效；
    - 当前主流程中它用于处理输入非有限值（NaN/Inf）通道，
      is_rfi 的置 NaN 在拟合后单独执行，以保持与目标流程一致。
    """
    y = np.asarray(values, dtype=np.float64).copy()
    m = np.asarray(input_mask, dtype=bool) | (~np.isfinite(y))
    if not np.any(m) or mode == "keep":
        return y

    if mode == "nan":
        y[m] = np.nan
        return y
    if mode == "zero":
        y[m] = 0.0
        return y
    if mode == "interp":
        good = ~m
        n_good = int(np.count_nonzero(good))
        if n_good >= 2:
            idx = np.arange(y.size)
            y[m] = np.interp(idx[m], idx[good], y[good])
        elif n_good == 1:
            y[m] = y[good][0]
        else:
            # 整条谱都被输入掩膜时无法插值，回退为 NaN，避免人造亮条。
            y[m] = np.nan
        return y

    raise ValueError(f"未知 masked fill mode: {mode}")


def fit_polynomial_baseline(
    spectrum: np.ndarray,
    fit_mask: np.ndarray,
    design_full: np.ndarray,
) -> np.ndarray:
    """
    在给定掩膜条件下做最小二乘多项式拟合。

    design_full 是全频率轴的 Vandermonde 设计矩阵，可复用以减少重复构造开销。
    """
    good = ~fit_mask
    if np.count_nonzero(good) < design_full.shape[1]:
        # 有效点不足时退化为常数基线，避免线性代数病态。
        c0 = np.nanmedian(spectrum[good]) if np.any(good) else 0.0
        return np.full_like(spectrum, c0, dtype=np.float64)

    a = design_full[good]
    b = spectrum[good]
    coef, *_ = np.linalg.lstsq(a, b, rcond=None)
    baseline = design_full @ coef
    return baseline


def fit_asls_baseline(
    spectrum: np.ndarray,
    fit_mask: np.ndarray,
    asls_penalty: sparse.csc_matrix,
    p: float,
    masked_weight: float,
) -> np.ndarray:
    """
    AsLS（Asymmetric Least Squares）基线拟合。

    该方法通过二阶差分惩罚抑制高频起伏，并用非对称权重避免发射峰把基线抬高，
    对 FAST 频谱中的宽带波纹通常比低阶多项式更稳健。
    """
    y = np.asarray(spectrum, dtype=np.float64).copy()
    n_chan = y.size
    mask = fit_mask | (~np.isfinite(y))

    if np.any(np.isfinite(y)):
        y[~np.isfinite(y)] = np.nanmedian(y[np.isfinite(y)])
    else:
        return np.zeros_like(y, dtype=np.float64)

    # 初始化权重：掩膜通道设极小权重，其余通道均匀权重。
    w = np.where(mask, masked_weight, 1.0)
    z = np.zeros_like(y, dtype=np.float64)

    # AsLS 内循环一般 5~12 次即可收敛。
    for _ in range(8):
        w = np.clip(w, masked_weight, 1.0)
        w_mat = sparse.spdiags(w, 0, n_chan, n_chan, format="csc")
        z = spsolve(w_mat + asls_penalty, w * y)
        residual = y - z
        w = np.where(mask, masked_weight, np.where(residual > 0, p, 1.0 - p))

    return z


def iterative_baseline_one_spectrum(
    spectrum: np.ndarray,
    base_mask: np.ndarray,
    design_full: np.ndarray,
    asls_penalty: Optional[sparse.csc_matrix],
    cfg: BaselineConfig,
) -> Tuple[np.ndarray, np.ndarray, int, float]:
    """
    对单条频谱执行迭代去基线。

    返回：
    - baseline: 拟合基线
    - final_mask: 最终拟合排除掩膜
    - used_iter: 实际迭代次数
    - final_sigma: 最后一次残差 MAD 噪声估计
    """
    local_mask = base_mask.copy()
    local_mask |= ~np.isfinite(spectrum)

    n_chan = spectrum.size
    min_valid = max(int(cfg.min_valid_frac * n_chan), cfg.poly_order + 2)

    used_iter = 0
    final_sigma = float("nan")

    for idx_iter in range(cfg.max_iter):
        used_iter = idx_iter + 1

        valid_now = ~local_mask
        if np.count_nonzero(valid_now) < min_valid:
            # 有效通道太少时直接使用常数模型，防止过拟合。
            c0 = np.nanmedian(spectrum[valid_now]) if np.any(valid_now) else 0.0
            baseline = np.full_like(spectrum, c0, dtype=np.float64)
            return baseline, local_mask, used_iter, final_sigma

        if cfg.method == "poly":
            baseline = fit_polynomial_baseline(spectrum, local_mask, design_full)
        elif cfg.method == "asls":
            if asls_penalty is None:
                raise ValueError("method=asls 但未提供 asls_penalty。")
            baseline = fit_asls_baseline(
                spectrum=spectrum,
                fit_mask=local_mask,
                asls_penalty=asls_penalty,
                p=cfg.asls_p,
                masked_weight=cfg.asls_masked_weight,
            )
        else:
            raise ValueError(f"未知 method: {cfg.method}")
        residual = spectrum - baseline
        sigma = robust_sigma_mad(residual, valid_now)
        final_sigma = sigma

        if not np.isfinite(sigma) or sigma <= 0:
            return baseline, local_mask, used_iter, final_sigma

        # 先做中位数居中，避免 AsLS 等方法导致残差整体偏置而触发掩膜失控。
        residual_centered = residual - np.nanmedian(residual[valid_now])

        # 对河外 HI 发射线优先做正残差保护，避免被当作基线扣除。
        line_mask = residual_centered > (cfg.pos_sigma * sigma)

        # 如目标存在吸收结构，可显式启用负残差保护。
        if cfg.detect_absorption:
            line_mask |= residual_centered < (-cfg.neg_sigma * sigma)

        line_mask &= valid_now

        # 防止单轮新增掩膜比例过大导致迭代崩溃（例如 AsLS 正偏残差场景）。
        max_allowed = int(cfg.max_line_frac * np.count_nonzero(valid_now))
        if max_allowed > 0 and np.count_nonzero(line_mask) > max_allowed:
            score = np.abs(residual_centered)
            valid_score = score[valid_now]
            cutoff = np.nanpercentile(valid_score, 100.0 * (1.0 - cfg.max_line_frac))
            line_mask = line_mask & (score >= cutoff)

        # 对谱线掩膜做膨胀，保护线翼和窄线邻域。
        if cfg.grow_chan > 0 and np.any(line_mask):
            structure = np.ones(2 * cfg.grow_chan + 1, dtype=bool)
            line_mask = binary_dilation(line_mask, structure=structure)
            if max_allowed > 0 and np.count_nonzero(line_mask) > max_allowed:
                score = np.abs(residual_centered)
                cutoff = np.nanpercentile(score[valid_now], 100.0 * (1.0 - cfg.max_line_frac))
                line_mask = line_mask & (score >= cutoff)

        new_mask = local_mask | line_mask

        # 掩膜不再变化时认为收敛。
        if np.array_equal(new_mask, local_mask):
            return baseline, local_mask, used_iter, final_sigma

        local_mask = new_mask

    # 到达最大迭代次数后，用最终掩膜再拟合一次。
    if cfg.method == "poly":
        baseline = fit_polynomial_baseline(spectrum, local_mask, design_full)
    elif cfg.method == "asls":
        if asls_penalty is None:
            raise ValueError("method=asls 但未提供 asls_penalty。")
        baseline = fit_asls_baseline(
            spectrum=spectrum,
            fit_mask=local_mask,
            asls_penalty=asls_penalty,
            p=cfg.asls_p,
            masked_weight=cfg.asls_masked_weight,
        )
    else:
        raise ValueError(f"未知 method: {cfg.method}")
    return baseline, local_mask, used_iter, final_sigma


def frequency_to_velocity_mps(
    freq_mhz: np.ndarray,
    rest_mhz: float,
    definition: str,
) -> np.ndarray:
    """按指定定义将频率轴转换为速度轴（m/s）。"""
    c = 299792458.0
    nu = np.asarray(freq_mhz, dtype=np.float64)
    nu0 = float(rest_mhz)

    if definition == "optical":
        # 光学定义：v = c * (lambda-lambda0)/lambda0 = c*(nu0/nu - 1)
        return c * (nu0 / nu - 1.0)
    if definition == "radio":
        # 射电定义：v = c * (nu0-nu)/nu0
        return c * (nu0 - nu) / nu0
    if definition == "relativistic":
        # 相对论定义：beta = (nu0^2-nu^2)/(nu0^2+nu^2), v=beta*c
        beta = (nu0**2 - nu**2) / (nu0**2 + nu**2)
        return beta * c

    raise ValueError(f"未知速度定义: {definition}")


def write_spectral_wcs(freq_mhz: np.ndarray, out_txt: Path) -> None:
    """生成频率轴 WCS 头信息文本，便于后续数据立方流程接入。"""
    w = WCS(naxis=1)
    freq_hz = np.asarray(freq_mhz, dtype=np.float64) * 1e6
    cdelt = (freq_hz[1] - freq_hz[0]) if freq_hz.size > 1 else 1.0

    w.wcs.ctype = ["FREQ"]
    w.wcs.cunit = ["Hz"]
    w.wcs.crpix = [1.0]
    w.wcs.crval = [freq_hz[0]]
    w.wcs.cdelt = [cdelt]

    header_text = w.to_header_string()
    out_txt.write_text(header_text, encoding="utf-8")


def save_checkpoint(ckpt_path: Path, payload: Dict[str, object]) -> None:
    """保存 checkpoint，支持崩溃后断点续跑。"""
    ckpt_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_checkpoint(ckpt_path: Path) -> Optional[Dict[str, object]]:
    """读取 checkpoint。"""
    if not ckpt_path.exists():
        return None
    return json.loads(ckpt_path.read_text(encoding="utf-8"))


def append_history(fout: h5py.File, args: argparse.Namespace) -> None:
    """在 Header 组记录本次处理参数，保持处理链路可追溯。"""
    if "Header" not in fout:
        fout.create_group("Header")

    tag = datetime.utcnow().strftime("HISTORY-%Y%m%d-%H:%M:%S")
    payload = {
        "version": "custom_iterative_bld_v1",
        "utc_time": datetime.utcnow().isoformat() + "Z",
        "argv": " ".join(sys.argv),
        "args": vars(args),
    }
    fout["Header"].attrs[tag] = json.dumps(payload, ensure_ascii=False, indent=2)


def resolve_required_paths(fin: h5py.File) -> Dict[str, str]:
    """解析 HiFAST 常见路径并做健壮性检查。"""
    candidates = {
        "flux": "S/flux",
        "freq": "S/freq",
        "is_rfi": "S/is_rfi",
        "is_excluded": "S/is_excluded",
        "waterfall": "Waterfall/DATA",
    }
    missing = [k for k, v in candidates.items() if v not in fin]
    if missing:
        raise KeyError(f"输入文件缺少必要路径: {missing}")
    return candidates


def scientific_plot_style() -> None:
    """设置论文风格绘图参数。"""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 130,
            "savefig.dpi": 180,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
        }
    )


def generate_compare_plot(
    fin: h5py.File,
    fout: h5py.File,
    freq_mhz: np.ndarray,
    out_png: Path,
    pol: int,
    start: int,
    width: int,
    vel_def: str,
) -> None:
    """输出处理前后频谱对比图，便于目视检查二次去基线效果。"""
    in_flux = fin["S/flux"]
    out_flux = fout["S/flux"]

    n_spec = in_flux.shape[1]
    start = max(0, min(start, n_spec - 1))
    end = min(n_spec, start + max(width, 1))

    before = np.nanmedian(in_flux[pol, start:end, :], axis=0)
    after = np.nanmedian(out_flux[pol, start:end, :], axis=0)

    vel_kms = frequency_to_velocity_mps(freq_mhz, REST_FREQ_HI_MHZ, vel_def) / 1e3

    scientific_plot_style()
    fig, ax1 = plt.subplots(figsize=(10, 5.8))
    ax1.plot(freq_mhz, before, lw=1.0, color="#D55E00", label="Before 2nd BLD")
    ax1.plot(freq_mhz, after, lw=1.0, color="#0072B2", label="After 2nd BLD")
    ax1.set_xlabel("Frequency [MHz]")
    ax1.set_ylabel("Flux Density [Jy]")
    ax1.set_title(f"Iterative Baseline Subtraction (POL={pol}, spec={start}:{end})")
    ax1.legend(loc="best")

    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    tick_freq = ax1.get_xticks()
    tick_vel = np.interp(tick_freq, freq_mhz, vel_kms)
    ax2.set_xticks(tick_freq)
    ax2.set_xticklabels([f"{v:.0f}" for v in tick_vel])
    ax2.set_xlabel(f"Velocity [{vel_def}, km/s]")

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def auto_select_plot_start(
    out_flux: h5py.Dataset,
    in_rfi: h5py.Dataset,
    pol: int,
    width: int,
    chunk_size: int = 128,
) -> Tuple[int, Dict[str, float]]:
    """
    自动选择用于频谱对比图的时间窗口起点。

    选择准则：
    1) 优先最大化 after 结果中的有限值占比，避免出现整段全 NaN 导致“后谱线消失”；
    2) 若所有窗口的有限值占比都为 0，则退化为最小化 RFI 占比。
    """
    nspec = int(out_flux.shape[1])
    width = max(1, int(width))
    if nspec <= width:
        return 0, {
            "mode": "short_series",
            "best_after_finite_mean": float("nan"),
            "best_rfi_mean": float("nan"),
        }

    finite_frac_t = np.zeros(nspec, dtype=np.float32)
    rfi_frac_t = np.zeros(nspec, dtype=np.float32)

    # 分块统计每条时间谱线的有效通道占比与 RFI 占比，降低一次性内存压力。
    for start in range(0, nspec, max(1, int(chunk_size))):
        end = min(nspec, start + max(1, int(chunk_size)))
        block_after = np.asarray(out_flux[pol, start:end, :], dtype=np.float32)
        block_rfi = np.asarray(in_rfi[start:end, :], dtype=bool)
        finite_frac_t[start:end] = np.mean(np.isfinite(block_after), axis=1)
        rfi_frac_t[start:end] = np.mean(block_rfi, axis=1)

    ker = np.ones(width, dtype=np.float64) / float(width)
    roll_after = np.convolve(finite_frac_t, ker, mode="valid")
    roll_rfi = np.convolve(rfi_frac_t, ker, mode="valid")

    idx_after = int(np.nanargmax(roll_after))
    if np.isfinite(roll_after[idx_after]) and roll_after[idx_after] > 0.0:
        return idx_after, {
            "mode": "auto_after_finite",
            "best_after_finite_mean": float(roll_after[idx_after]),
            "best_rfi_mean": float(roll_rfi[idx_after]),
        }

    idx_rfi = int(np.nanargmin(roll_rfi))
    return idx_rfi, {
        "mode": "fallback_min_rfi",
        "best_after_finite_mean": float(roll_after[idx_rfi]),
        "best_rfi_mean": float(roll_rfi[idx_rfi]),
    }


def generate_waterfall_plot(
    fout: h5py.File,
    out_png: Path,
    pol: int,
) -> None:
    """
    输出处理结果瀑布图（time-channel, after only）。

    物理目的：
    - 检查二次基线后是否仍有条纹状伪结构；
    - 与 CARTA 全图视图保持一致，默认绘制全时序范围。
    """
    out_flux = fout["S/flux"]

    n_spec = out_flux.shape[1]
    after = np.asarray(out_flux[pol, 0:n_spec, :], dtype=np.float32)

    vals = after[np.isfinite(after)]
    if vals.size > 0:
        vmin = float(np.nanpercentile(vals, 5))
        vmax = float(np.nanpercentile(vals, 95))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
            vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
    else:
        vmin, vmax = -1.0, 1.0

    scientific_plot_style()
    fig, ax = plt.subplots(1, 1, figsize=(12.5, 6.2), constrained_layout=True)
    cmap_main = "magma"

    im = ax.imshow(
        after,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap_main,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(f"Waterfall After 2nd BLD (POL={pol}, spec=0:{n_spec})")
    ax.set_xlabel("Channel (pixel)")
    ax.set_ylabel("Time Index (pixel)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="Flux [Jy]")
    fig.savefig(out_png)
    plt.close(fig)


def estimate_common_mode_spectrum(
    flux_ds: h5py.Dataset,
    excluded_ds: h5py.Dataset,
    pol: int,
    start_spec: int,
    stop_spec: int,
    chunk_size: int,
) -> np.ndarray:
    """
    估计给定极化在处理区间内的共模频谱。

    这里采用“掩膜后逐通道均值”估计，再做平滑，目的是去除仍然残留的
    大尺度公共基线起伏（例如 standing-wave 宽带结构）。
    """
    n_chan = flux_ds.shape[2]
    sum_spec = np.zeros(n_chan, dtype=np.float64)
    cnt_spec = np.zeros(n_chan, dtype=np.float64)

    for start in range(start_spec, stop_spec, chunk_size):
        end = min(stop_spec, start + chunk_size)
        block = np.asarray(flux_ds[pol, start:end, :], dtype=np.float64)
        excluded = np.asarray(excluded_ds[start:end, :], dtype=bool)
        good = np.isfinite(block) & (~excluded)
        sum_spec += np.nansum(np.where(good, block, 0.0), axis=0)
        cnt_spec += np.sum(good, axis=0)

    common = np.zeros_like(sum_spec)
    valid = cnt_spec > 0
    common[valid] = sum_spec[valid] / cnt_spec[valid]
    return common


def apply_common_mode_correction(
    flux_ds: h5py.Dataset,
    wf_ds: h5py.Dataset,
    baseline_ds: Optional[h5py.Dataset],
    common_mode: np.ndarray,
    pol: int,
    start_spec: int,
    stop_spec: int,
    chunk_size: int,
) -> None:
    """把共模频谱从指定区间逐块扣除，并同步更新 Waterfall 与可选 baseline。"""
    common = np.asarray(common_mode, dtype=np.float64)
    for start in range(start_spec, stop_spec, chunk_size):
        end = min(stop_spec, start + chunk_size)
        block = np.asarray(flux_ds[pol, start:end, :], dtype=np.float64)
        block -= common[None, :]
        flux_ds[pol, start:end, :] = block.astype(flux_ds.dtype, copy=False)
        wf_ds[pol, start:end, :] = block.astype(wf_ds.dtype, copy=False)

        if baseline_ds is not None:
            base_block = np.asarray(baseline_ds[pol, start:end, :], dtype=np.float64)
            base_block += common[None, :]
            baseline_ds[pol, start:end, :] = base_block.astype(baseline_ds.dtype, copy=False)


def close_logger_handlers(logger: logging.Logger) -> None:
    """关闭 logger handler，避免删除 sidecar 文件时占用句柄。"""
    handlers = list(logger.handlers)
    for handler in handlers:
        try:
            handler.flush()
            handler.close()
        finally:
            logger.removeHandler(handler)


def cleanup_sidecar_files(output_path: Path) -> Tuple[Path, Path, Path]:
    """删除与输出文件同 stem 的 sidecar 文件（checkpoint/log/wcs）。"""
    run_dir = output_path.parent
    stem = output_path.stem
    log_path = run_dir / f"{stem}_iterbld.log"
    ckpt_path = run_dir / f"{stem}_iterbld.checkpoint.json"
    wcs_path = run_dir / f"{stem}_spectral_wcs.txt"
    for sidecar in [log_path, ckpt_path, wcs_path]:
        if sidecar.exists():
            sidecar.unlink()
    return log_path, ckpt_path, wcs_path


def build_single_run_subprocess_command(
    args: argparse.Namespace,
    input_path: Path,
    output_path: Path,
    cleanup_sidecars: bool,
) -> list[str]:
    """在批处理模式下构建子进程命令（单文件执行路径）。"""
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--chunk-size",
        str(args.chunk_size),
        "--start-spec",
        str(args.start_spec),
        "--stop-spec",
        str(args.stop_spec),
        "--poly-order",
        str(args.poly_order),
        "--method",
        str(args.method),
        "--asls-lambda",
        str(args.asls_lambda),
        "--asls-p",
        str(args.asls_p),
        "--asls-masked-weight",
        str(args.asls_masked_weight),
        "--max-iter",
        str(args.max_iter),
        "--pos-sigma",
        str(args.pos_sigma),
        "--neg-sigma",
        str(args.neg_sigma),
        "--grow-chan",
        str(args.grow_chan),
        "--min-valid-frac",
        str(args.min_valid_frac),
        "--max-line-frac",
        str(args.max_line_frac),
        "--plot-pol",
        str(args.plot_pol),
        "--plot-start",
        str(args.plot_start),
        "--plot-width",
        str(args.plot_width),
        "--plot-window-mode",
        str(args.plot_window_mode),
        "--vel-def",
        str(args.vel_def),
        "--common-mode-sigma",
        str(args.common_mode_sigma),
        "--local-ripple-sigma",
        str(args.local_ripple_sigma),
        "--masked-fill",
        str(args.masked_fill),
        "--log-level",
        str(args.log_level),
    ]

    if args.detect_absorption:
        cmd.append("--detect-absorption")
    if args.resume:
        cmd.append("--resume")
    if args.save_baseline:
        cmd.append("--save-baseline")
    if args.plot:
        cmd.append("--plot")
    if args.plot_waterfall:
        cmd.append("--plot-waterfall")
    if args.disable_common_mode:
        cmd.append("--disable-common-mode")
    if args.force:
        cmd.append("--force")
    if cleanup_sidecars:
        cmd.append("--cleanup-sidecars")
    return cmd


def run_batch_mode(args: argparse.Namespace) -> int:
    """批处理入口：遍历目录内匹配文件并逐个调用单文件流程。"""
    batch_dir = Path(args.batch_dir).expanduser().resolve()
    if not batch_dir.exists():
        raise FileNotFoundError(f"批处理目录不存在: {batch_dir}")

    output_dir = (
        Path(args.batch_output_dir).expanduser().resolve()
        if args.batch_output_dir
        else batch_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(batch_dir.glob(args.batch_pattern))
    total = len(input_files)
    if total == 0:
        raise FileNotFoundError(
            f"未匹配到输入文件: dir={batch_dir}, pattern={args.batch_pattern}"
        )

    print(f"[BATCH] 匹配到 {total} 个输入文件。")
    print(f"[BATCH] 输出目录: {output_dir}")
    print(f"[BATCH] 输出后缀: {args.batch_output_suffix}")

    done = 0
    skip = 0
    fail = 0
    cleanup_sidecars = not args.batch_keep_sidecars

    # 先扫描任务列表：已完成文件可直接跳过，避免占用并行槽位。
    run_tasks = []
    for idx, input_path in enumerate(input_files, start=1):
        output_name = f"{input_path.stem}{args.batch_output_suffix}.hdf5"
        output_path = output_dir / output_name

        if output_path.exists() and not args.force:
            print(f"[{idx}/{total}] [SKIP] 已存在输出: {output_path.name}")
            skip += 1
            continue

        run_tasks.append((idx, input_path, output_path))

    # 根据用户参数解析并行度：0 表示自动估算，>=1 表示显式指定。
    if int(args.batch_workers) <= 0:
        cpu_total = os.cpu_count() or 1
        # 预留 1 个核心给系统/IO 调度，避免满载导致整体吞吐下降。
        workers = max(1, cpu_total - 1)
    else:
        workers = int(args.batch_workers)

    # 并行度不超过待执行任务数，避免创建空闲 worker。
    workers = min(workers, max(1, len(run_tasks)))
    print(f"[BATCH] 并行工作数: {workers}")

    if len(run_tasks) == 0:
        print(f"[BATCH-SUMMARY] DONE={done} SKIP={skip} FAIL={fail} TOTAL={total}")
        return 0

    def _run_one(task: Tuple[int, Path, Path]) -> Tuple[int, Path, Path, int]:
        """执行单个批处理任务，返回索引、输入、输出与退出码。"""
        idx, input_path, output_path = task
        cmd = build_single_run_subprocess_command(
            args=args,
            input_path=input_path,
            output_path=output_path,
            cleanup_sidecars=cleanup_sidecars,
        )
        print(f"[{idx}/{total}] [RUN ] {input_path.name}")
        ret = subprocess.run(cmd)
        return idx, input_path, output_path, int(ret.returncode)

    if workers == 1:
        # 保持与历史行为一致的串行执行路径。
        for task in run_tasks:
            idx, input_path, output_path, code = _run_one(task)
            if code == 0:
                print(f"[{idx}/{total}] [DONE] {output_path.name}")
                done += 1
            else:
                print(f"[{idx}/{total}] [FAIL] {input_path.name} (code={code})")
                fail += 1
    else:
        # 并行执行：每个 worker 处理一个文件，充分利用多核 CPU。
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {executor.submit(_run_one, task): task for task in run_tasks}
            for future in as_completed(future_map):
                idx, input_path, output_path = future_map[future]
                try:
                    _, _, _, code = future.result()
                except Exception as exc:  # pragma: no cover - 防御式错误处理
                    print(f"[{idx}/{total}] [FAIL] {input_path.name} (exception={exc})")
                    fail += 1
                    continue

                if code == 0:
                    print(f"[{idx}/{total}] [DONE] {output_path.name}")
                    done += 1
                else:
                    print(f"[{idx}/{total}] [FAIL] {input_path.name} (code={code})")
                    fail += 1

    print(f"[BATCH-SUMMARY] DONE={done} SKIP={skip} FAIL={fail} TOTAL={total}")
    return 0 if fail == 0 else 2


def main() -> int:
    """主流程：参数解析 -> 安全检查 -> 分块迭代去基线 -> 输出与诊断。"""
    parser = build_arg_parser()
    args = parser.parse_args()

    # 批处理模式：遍历目录内全部匹配文件，逐个调用单文件流程。
    if args.batch_dir is not None:
        if args.input or args.output:
            parser.error("使用 --batch-dir 时，请不要同时传 --input/--output。")
        return run_batch_mode(args)

    # 单文件模式必须同时提供输入与输出路径。
    if not args.input or not args.output:
        parser.error("单文件模式需要同时指定 --input 和 --output。")

    if args.batch_output_dir is not None:
        parser.error("单文件模式下不需要 --batch-output-dir。")

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    ensure_non_destructive_io(input_path, output_path, force=(args.force or args.resume))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_dir = output_path.parent
    stem = output_path.stem

    log_path = run_dir / f"{stem}_iterbld.log"
    ckpt_path = run_dir / f"{stem}_iterbld.checkpoint.json"
    wcs_path = run_dir / f"{stem}_spectral_wcs.txt"
    plot_path = run_dir / f"{stem}_before_vs_after_iterbld.png"
    waterfall_path = run_dir / f"{stem}_waterfall_after_iterbld.png"

    logger = setup_logger(log_path, args.log_level)

    cfg = BaselineConfig(
        method=args.method,
        poly_order=args.poly_order,
        max_iter=args.max_iter,
        pos_sigma=args.pos_sigma,
        neg_sigma=args.neg_sigma,
        grow_chan=args.grow_chan,
        detect_absorption=args.detect_absorption,
        min_valid_frac=args.min_valid_frac,
        max_line_frac=float(args.max_line_frac),
        asls_lambda=float(args.asls_lambda),
        asls_p=float(args.asls_p),
        asls_masked_weight=float(args.asls_masked_weight),
    )

    logger.info("输入文件: %s", input_path)
    logger.info("输出文件: %s", output_path)
    logger.info("参数: %s", asdict(cfg))

    # 处理断点恢复逻辑，仅在 --resume 且 checkpoint 可用时生效。
    start_spec = 0
    if args.resume and ckpt_path.exists() and output_path.exists():
        checkpoint = load_checkpoint(ckpt_path)
        if checkpoint is not None:
            if checkpoint.get("input_path") == str(input_path) and checkpoint.get("output_path") == str(output_path):
                start_spec = int(checkpoint.get("last_done_spec", 0))
                logger.info("检测到 checkpoint，从第 %d 条时间谱继续。", start_spec)
            else:
                logger.warning("checkpoint 与当前输入/输出不匹配，忽略断点并从头处理。")

    write_mode = "r+" if output_path.exists() else "w"

    with h5py.File(input_path, "r") as fin, h5py.File(output_path, write_mode) as fout:
        paths = resolve_required_paths(fin)

        # 首次创建输出文件时复制结构（跳过要重新写入的大数组）。
        if write_mode == "w":
            copy_hdf5_skeleton(fin, fout, skip_paths=[paths["flux"], paths["waterfall"]])
            append_history(fout, args)
            logger.info("已完成输出文件结构初始化。")

            if args.save_baseline and "S/baseline_iter2" not in fout:
                src_flux = fin[paths["flux"]]
                fout.create_dataset(
                    "S/baseline_iter2",
                    shape=src_flux.shape,
                    dtype=np.float32,
                    chunks=src_flux.chunks,
                    compression=src_flux.compression,
                    compression_opts=src_flux.compression_opts,
                )

            if "S/bld2_iter_used" not in fout:
                src_flux = fin[paths["flux"]]
                fout.create_dataset(
                    "S/bld2_iter_used",
                    shape=(src_flux.shape[0], src_flux.shape[1]),
                    dtype=np.uint8,
                )

            if "S/bld2_sigma_mad" not in fout:
                src_flux = fin[paths["flux"]]
                fout.create_dataset(
                    "S/bld2_sigma_mad",
                    shape=(src_flux.shape[0], src_flux.shape[1]),
                    dtype=np.float32,
                )

            if "S/is_excluded_iter2" not in fout:
                src_exc = fin[paths["is_excluded"]]
                fout.create_dataset(
                    "S/is_excluded_iter2",
                    shape=src_exc.shape,
                    dtype=np.bool_,
                    chunks=src_exc.chunks,
                    compression=src_exc.compression,
                    compression_opts=src_exc.compression_opts,
                )

        in_flux = fin[paths["flux"]]
        out_flux = fout[paths["flux"]]
        out_wf = fout[paths["waterfall"]]
        in_rfi = fin[paths["is_rfi"]]
        in_exc = fin[paths["is_excluded"]]
        freq_mhz = np.asarray(fin[paths["freq"]][:], dtype=np.float64)

        # 归一化频率轴减少高阶多项式数值病态。
        fmid = 0.5 * (np.nanmin(freq_mhz) + np.nanmax(freq_mhz))
        fhalf = max(0.5 * (np.nanmax(freq_mhz) - np.nanmin(freq_mhz)), 1e-9)
        x_norm = (freq_mhz - fmid) / fhalf
        design_full = np.vander(x_norm, cfg.poly_order + 1, increasing=True)

        # AsLS 惩罚矩阵只与通道数有关，预先构建可减少重复开销。
        asls_penalty = None
        if cfg.method == "asls":
            n_chan_tmp = freq_mhz.size
            d2 = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n_chan_tmp - 2, n_chan_tmp), format="csc")
            asls_penalty = float(cfg.asls_lambda) * (d2.T @ d2)

        npol, nspec, nchan = in_flux.shape

        user_start = max(0, int(args.start_spec))
        user_stop = nspec if int(args.stop_spec) < 0 else min(int(args.stop_spec), nspec)
        if user_stop <= user_start:
            raise ValueError(f"非法处理范围: start={user_start}, stop={user_stop}, nspec={nspec}")

        # 若只处理子区间，先把原始 flux/waterfall 拷贝到输出，确保未处理区间不变成零值。
        if write_mode == "w" and (user_start > 0 or user_stop < nspec):
            logger.info("检测到子区间处理，先复制原始 flux/waterfall 全量数据以保护未处理区间。")
            copy_chunk = max(1, int(args.chunk_size))
            for cstart in range(0, nspec, copy_chunk):
                cend = min(nspec, cstart + copy_chunk)
                out_flux[:, cstart:cend, :] = in_flux[:, cstart:cend, :]
                out_wf[:, cstart:cend, :] = fin[paths["waterfall"]][:, cstart:cend, :]
            fout.flush()

        # 当同时指定 --resume 与显式范围时，从两者较大值继续，避免重复计算。
        start_spec = max(start_spec, user_start)

        if start_spec >= user_stop:
            logger.info("目标范围 [%d, %d) 已处理完成，无需重复运行。", user_start, user_stop)
        else:
            logger.info(
                "数据维度: npol=%d, nspec=%d, nchan=%d | 本次处理范围: [%d, %d)",
                npol,
                nspec,
                nchan,
                start_spec,
                user_stop,
            )

            chunk_size = max(1, int(args.chunk_size))
            total_chunks = math.ceil((user_stop - start_spec) / chunk_size)

            for ichunk, start in enumerate(range(start_spec, user_stop, chunk_size), start=1):
                end = min(user_stop, start + chunk_size)

                # 分块读取并转为 float64，提升拟合稳定性。
                flux_chunk = np.asarray(in_flux[:, start:end, :], dtype=np.float64)
                rfi_chunk = np.asarray(in_rfi[start:end, :], dtype=bool)
                exc_chunk = np.asarray(in_exc[start:end, :], dtype=bool)

                out_chunk = np.array(flux_chunk, copy=True)
                iter_used_chunk = np.zeros((npol, end - start), dtype=np.uint8)
                sigma_chunk = np.full((npol, end - start), np.nan, dtype=np.float32)
                exc_iter2_chunk = np.array(exc_chunk, copy=True)

                baseline_chunk = None
                if args.save_baseline:
                    baseline_chunk = np.zeros_like(flux_chunk, dtype=np.float64)

                # 逐条谱线迭代处理；每条内层计算使用 NumPy 向量化完成通道运算。
                for ipol in range(npol):
                    for it in range(end - start):
                        spec = flux_chunk[ipol, it, :]
                        # 与目标流程一致：拟合阶段仅排除 is_excluded 与原始非有限值。
                        nonfinite_mask = ~np.isfinite(spec)
                        fit_mask = exc_chunk[it, :] | nonfinite_mask

                        baseline, final_mask, used_iter, final_sigma = iterative_baseline_one_spectrum(
                            spectrum=spec,
                            base_mask=fit_mask,
                            design_full=design_full,
                            asls_penalty=asls_penalty,
                            cfg=cfg,
                        )

                        spec_sub = spec - baseline
                        # 可选的逐谱线局部波纹扣除：对剩余宽带项做高斯平滑并减去。
                        if float(args.local_ripple_sigma) > 0:
                            # 关键：局部波纹估计时显式屏蔽 final_mask（含 is_excluded/非有限值/迭代识别线区），
                            # 避免被屏蔽通道通过卷积扩散影响未屏蔽区域。
                            local_ripple = masked_gaussian_smooth1d(
                                values=spec_sub,
                                bad_mask=final_mask,
                                sigma=float(args.local_ripple_sigma),
                            )
                            spec_sub -= local_ripple
                            if baseline_chunk is not None:
                                baseline_chunk[ipol, it, :] = baseline + local_ripple

                        # 仅按输入非有限值策略回填，避免 is_excluded 被错误写空。
                        spec_sub = fill_masked_channels_1d(
                            values=spec_sub,
                            input_mask=nonfinite_mask,
                            mode=str(args.masked_fill),
                        )
                        # 拟合完成后再将 is_rfi 置为 NaN，语义上与“RFI 标记”一致。
                        spec_sub[rfi_chunk[it, :]] = np.nan
                        out_chunk[ipol, it, :] = spec_sub

                        iter_used_chunk[ipol, it] = used_iter
                        sigma_chunk[ipol, it] = np.float32(final_sigma) if np.isfinite(final_sigma) else np.nan
                        exc_iter2_chunk[it, :] = exc_iter2_chunk[it, :] | final_mask

                        if baseline_chunk is not None and float(args.local_ripple_sigma) <= 0:
                            baseline_chunk[ipol, it, :] = baseline

                # 写回输出文件，保持非破坏性处理链路。
                out_flux[:, start:end, :] = out_chunk.astype(out_flux.dtype, copy=False)
                out_wf[:, start:end, :] = out_chunk.astype(out_wf.dtype, copy=False)
                fout["S/bld2_iter_used"][:, start:end] = iter_used_chunk
                fout["S/bld2_sigma_mad"][:, start:end] = sigma_chunk
                fout["S/is_excluded_iter2"][start:end, :] = exc_iter2_chunk
                if baseline_chunk is not None:
                    fout["S/baseline_iter2"][:, start:end, :] = baseline_chunk.astype(np.float32)

                # 每个分块落盘并刷新 checkpoint，防止长任务中断导致进度丢失。
                fout.flush()
                checkpoint_payload = {
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "last_done_spec": end,
                    "utc_time": datetime.utcnow().isoformat() + "Z",
                    "config": asdict(cfg),
                }
                save_checkpoint(ckpt_path, checkpoint_payload)

                logger.info(
                    "chunk %d/%d 完成: [%d, %d) | 已处理 %.2f%%",
                    ichunk,
                    total_chunks,
                    start,
                    end,
                    100.0 * (end - user_start) / (user_stop - user_start),
                )

        # 对处理区间执行共模后处理，去除残余的大尺度公共基线起伏。
        if not args.disable_common_mode and user_stop > user_start:
            if "S/common_mode_iter2" not in fout:
                fout.create_dataset("S/common_mode_iter2", shape=(npol, nchan), dtype=np.float32)

            baseline_ds = fout["S/baseline_iter2"] if ("S/baseline_iter2" in fout) else None
            cm_chunk = max(1, int(args.chunk_size))
            for ipol in range(npol):
                common = estimate_common_mode_spectrum(
                    flux_ds=out_flux,
                    excluded_ds=fout["S/is_excluded_iter2"],
                    pol=ipol,
                    start_spec=user_start,
                    stop_spec=user_stop,
                    chunk_size=cm_chunk,
                )
                common_smooth = gaussian_filter1d(common, sigma=float(args.common_mode_sigma), mode="nearest")
                apply_common_mode_correction(
                    flux_ds=out_flux,
                    wf_ds=out_wf,
                    baseline_ds=baseline_ds,
                    common_mode=common_smooth,
                    pol=ipol,
                    start_spec=user_start,
                    stop_spec=user_stop,
                    chunk_size=cm_chunk,
                )
                fout["S/common_mode_iter2"][ipol, :] = common_smooth.astype(np.float32)

            fout["S/common_mode_iter2"].attrs["sigma_chan"] = float(args.common_mode_sigma)
            fout["S/common_mode_iter2"].attrs["applied_range"] = f"[{user_start}, {user_stop})"
            fout.flush()
            logger.info(
                "已完成共模后处理: sigma=%.1f chan, range=[%d, %d)",
                float(args.common_mode_sigma),
                user_start,
                user_stop,
            )

        # 输出频率轴 WCS 供后续 cube/坐标映射流程使用。
        write_spectral_wcs(freq_mhz, wcs_path)
        logger.info("已输出频率轴 WCS: %s", wcs_path)

        # 可选输出前后频谱对比图。
        if args.plot or args.plot_waterfall:
            plot_start = args.plot_start
            if plot_start < 0:
                if args.plot_window_mode == "center":
                    plot_start = max(0, nspec // 2 - args.plot_width // 2)
                    logger.info("绘图窗口模式=center，使用中段窗口起点: %d", plot_start)
                else:
                    plot_start, metrics = auto_select_plot_start(
                        out_flux=out_flux,
                        in_rfi=in_rfi,
                        pol=max(0, min(args.plot_pol, npol - 1)),
                        width=int(args.plot_width),
                        chunk_size=max(32, int(args.chunk_size)),
                    )
                    logger.info(
                        "绘图窗口模式=auto，起点=%d，after有效占比均值=%.4f，RFI占比均值=%.4f（%s）",
                        plot_start,
                        metrics["best_after_finite_mean"],
                        metrics["best_rfi_mean"],
                        metrics["mode"],
                    )
            pol = max(0, min(args.plot_pol, npol - 1))
        if args.plot:
            generate_compare_plot(
                fin=fin,
                fout=fout,
                freq_mhz=freq_mhz,
                out_png=plot_path,
                pol=pol,
                start=plot_start,
                width=args.plot_width,
                vel_def=args.vel_def,
            )
            logger.info("已输出前后对比图: %s", plot_path)
        if args.plot_waterfall:
            generate_waterfall_plot(
                fout=fout,
                out_png=waterfall_path,
                pol=pol,
            )
            logger.info("已输出结果瀑布图: %s", waterfall_path)

    logger.info("二次去基线处理完成。")
    if args.cleanup_sidecars:
        # 先关闭日志句柄，再删除 sidecar 文件，避免句柄占用导致删除失败。
        close_logger_handlers(logger)
        removed = cleanup_sidecar_files(output_path)
        print(
            "已清理 sidecar 文件: "
            + ", ".join(str(p) for p in removed)
        )
    else:
        logger.info("日志文件: %s", log_path)
        logger.info("checkpoint: %s", ckpt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
