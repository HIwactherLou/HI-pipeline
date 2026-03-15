import os
import glob
import re
import argparse
import sys
import h5py
import numpy as np

def determine_time_axis(name, obj, n_time):
    """
    判断当前数据集是否随时间变化，以及对应的时间拼接轴是哪一条。
    name: 数据集的绝对路径 (例如 '/S/Ta')
    """
    base_name = name.split('/')[-1]
    
    # 1. 明确的非时间维度白名单（不需要拼接的数据，静态写入一次即可）
    NON_TIME_KEYS = ['freq', 'Tcal', 'inds_ton', 'is_aband_whole', 
                     'pcals_merged', 'pcals_merged_s']
    
    # 如果处于 Header 组下，或者是白名单数据，则返回 None (不沿着时间轴扩展)
    if '/Header/' in name or name.startswith('/Header') or base_name in NON_TIME_KEYS:
        return None
        
    # 2. 已知数据集的拼接轴字典 (axis)
    KNOWN_TIME_AXES = {
        'mjd': 0,
        'Ta': 1,
        'DATA': 1,
        'is_delay': 0,
        'is_on': 0,
        'next_to_cal': 0,
        'pcals_amp_diff_interp_values': 0
    }
    
    if base_name in KNOWN_TIME_AXES:
        return KNOWN_TIME_AXES[base_name]
        
    # 3. 启发式推断时间轴 (加一个阈值 n_time > 2 防止形状中的小数字巧合匹配)
    if n_time is not None and n_time in obj.shape and n_time > 2:
        return list(obj.shape).index(n_time)
        
    return None

def merge_hdf5_files(flist, out_path):
    first_file = flist[0]
    
    with h5py.File(first_file, 'r') as h5_in, h5py.File(out_path, 'w') as h5_out:
        
        # 提取时间维度大小，辅助后续判断
        n_time = None
        if 'S/mjd' in h5_in:
            n_time = h5_in['S/mjd'].shape[0]
        
        time_axes = {}
        
        # 【关键修正】自定义的递归遍历函数，支持捕获软链接 (SoftLink)
        def init_structure(group_in, group_out):
            for key in group_in.keys():
                # getlink=True 是核心，这样遇到软链接就不会直接顺延到真实数据了
                link = group_in.get(key, getlink=True)
                
                if isinstance(link, h5py.SoftLink):
                    # 如果是软链接，原样复制软链接（这正是 Carta 找 DATA 的关键）
                    group_out[key] = h5py.SoftLink(link.path)
                elif isinstance(link, h5py.ExternalLink):
                    # 如果有外部链接，原样复制
                    group_out[key] = h5py.ExternalLink(link.filename, link.path)
                else:
                    obj = group_in[key]
                    if isinstance(obj, h5py.Group):
                        new_grp = group_out.create_group(key)
                        # 复制 Group 的属性
                        for k, v in obj.attrs.items():
                            new_grp.attrs[k] = v
                        # 递归进入下一层
                        init_structure(obj, new_grp)
                    elif isinstance(obj, h5py.Dataset):
                        name = obj.name  # 获取绝对路径，如 '/S/Ta'
                        time_axis = determine_time_axis(name, obj, n_time)
                        
                        if time_axis is not None:
                            # 这是一个随时间变化的数组，设定 maxshape 开启扩容机制
                            time_axes[name] = time_axis
                            maxshape = list(obj.shape)
                            maxshape[time_axis] = None  
                            
                            out_ds = group_out.create_dataset(
                                key, data=obj[:], maxshape=tuple(maxshape), chunks=True
                            )
                        else:
                            # 静态数组，直接完整写一遍即可
                            out_ds = group_out.create_dataset(key, data=obj[:])
                            
                        # 复制 Dataset 的属性
                        for k, v in obj.attrs.items():
                            out_ds.attrs[k] = v

        # 1. 完整复制原始数据的骨架和第一个文件的内容
        init_structure(h5_in, h5_out)
        for k, v in h5_in.attrs.items():
            h5_out.attrs[k] = v
            
        # 2. 循环处理剩下的切片，动态 Resize 塞入数据
        for fpath in flist[1:]:
            with h5py.File(fpath, 'r') as h5_append:
                for name, time_axis in time_axes.items():
                    if name in h5_append:
                        new_data = h5_append[name][:]
                        out_ds = h5_out[name]  # 这里直接用绝对路径获取数据集
                        
                        old_shape = out_ds.shape
                        append_size = new_data.shape[time_axis]
                        
                        # 计算新尺寸
                        new_shape = list(old_shape)
                        new_shape[time_axis] += append_size
                        
                        # 沿着时间轴扩容
                        out_ds.resize(tuple(new_shape))
                        
                        # 制作切片指向新开辟的空间，填入数据
                        slices = [slice(None)] * len(old_shape)
                        slices[time_axis] = slice(old_shape[time_axis], new_shape[time_axis])
                        out_ds[tuple(slices)] = new_data

def main():
    parser = argparse.ArgumentParser(description='Auto Merge Chunked HDF5 files into full Beam files')
    parser.add_argument('input_dir', type=str, help='Input directory containing chunked HDF5 files')
    parser.add_argument('--outdir', type=str, required=True, help='Directory to store the merged full HDF5 files')
    # 使用 target 作为过滤参数
    parser.add_argument('--target', type=str, default='-fc-ds.hdf5', help='Target suffix to filter final files')
    
    args = parser.parse_args()
    
    # 智能处理传参：不管你传 "-fc-ds.hdf5" 还是 "fc-ds.hdf5" 都能正确识别
    actual_target = args.target if args.target.startswith('-') else f"-{args.target}"

    if not os.path.isdir(args.input_dir):
        print(f"Error: 输入路径不是文件夹 -> {args.input_dir}")
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    search_pattern = os.path.join(args.input_dir, "*.hdf5")
    all_files = glob.glob(search_pattern)
    
    if len(all_files) == 0:
        print(f"Error: 在目录 {args.input_dir} 中没有找到 .hdf5 文件")
        sys.exit(1)

    # 匹配 M01_W_0001-0007_xxx.hdf5 格式
    pattern = re.compile(r'^(.*?-M\d{2}_W_)(\d+)-(\d+)(_.*\.hdf5)$')

    groups = {}

    for fpath in all_files:
        filename = os.path.basename(fpath)
        
        # 只选取特定后缀的文件
        if not filename.endswith(actual_target):
            continue
            
        match = pattern.search(filename)
        if match:
            prefix = match.group(1)
            start_chunk = int(match.group(2))
            end_chunk = int(match.group(3))
            suffix = match.group(4)
            
            group_key = (prefix, suffix)
            if group_key not in groups:
                groups[group_key] = []
                
            groups[group_key].append((start_chunk, end_chunk, fpath))

    if not groups:
        print(f"Warning: 没有找到带有后缀 '{actual_target}' 且符合命名规则的切片文件。")
        sys.exit(0)

    print(f"成功分组，共找到 {len(groups)} 个合并任务。目标后缀: {actual_target}")
    print("-" * 60)

    for (prefix, suffix), files_info in groups.items():
        # 按切片起始序号排序，确保时间连续
        files_info.sort(key=lambda x: x[0])
        flist = [x[2] for x in files_info]
        
        out_filename = prefix + suffix[1:]
        out_path = os.path.join(args.outdir, out_filename)
        
        beam_match = re.search(r'-M(\d{2})_W_', prefix)
        beam_id = beam_match.group(1) if beam_match else "Unknown"

        print(f"正在合并 Beam M{beam_id} | 发现 {len(flist)} 个 Chunk 文件片段")
        
        try:
            merge_hdf5_files(flist, out_path)
            print(f"  -> 成功生成: {out_filename}")
        except Exception as e:
            print(f"  Error 合并 Beam M{beam_id} 失败: {e}")
            import traceback
            traceback.print_exc()

    print("-" * 60)
    print("所有 Beam 合并完成。")

if __name__ == "__main__":
    main()
