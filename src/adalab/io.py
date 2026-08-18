"""
实验产物的压缩存储与加载工具。

本模块提供：
- joblib 对象的 lzma 压缩存储
- 大对象的分块保存与合并加载
- 模型与监控数据的统一读写接口

该模块用于实验结果的长期保存与跨环境传输。
"""

import lzma
import os

import joblib


def dump_compressed_chunks(obj, filepath: str, chunk_size_mb=50):
    """压缩并分块保存 Python 对象。

    该函数先使用 lzma 对对象进行压缩（joblib 序列化），
    再将压缩文件切分为多个固定大小的分片文件，
    适用于大模型或大监控对象的存储、传输与提交场景。

    分片文件将以 ``.partXXX`` 的形式保存在同一目录下。

    Args:
        obj (Any): 需要保存的 Python 对象（需可被 joblib 序列化）。
        filepath (str): 原始文件路径（不包含 .xz 后缀）。
        chunk_size_mb (int, optional): 每个分片的最大大小（MB），默认 50。

    Returns:
        list[str]: 生成的分片文件路径列表，按顺序排列。
    """
    compressed_path = filepath + ".xz"

    # 压缩到 .xz
    with lzma.open(compressed_path, "wb") as f:
        joblib.dump(obj, f)
    print(f"[Workflow] compressing finished: {compressed_path}")

    # 切片
    chunk_size = chunk_size_mb * 1024 * 1024
    chunks = []

    with open(compressed_path, "rb") as f:
        idx = 0
        while True:
            data = f.read(chunk_size)
            if not data:
                break

            part_path = f"{compressed_path}.part{idx:03d}"
            with open(part_path, "wb") as pf:
                pf.write(data)

            chunks.append(part_path)
            idx += 1

    print(f"[Workflow] cut into {len(chunks)} chunks，chunk size ≤ {chunk_size_mb}MB：")
    for p in chunks:
        print("  ", p)

    return chunks


def load_compressed_chunks(basepath: str):
    """从分块压缩文件中加载 Python 对象。

    该函数会在指定路径所在目录中自动查找
    ``.part000``, ``.part001`` 等分片文件，
    按顺序合并后解压并反序列化得到原始对象。

    适用于由 ``dump_compressed_chunks`` 生成的分片文件。

    Args:
        basepath (str): 原始压缩文件路径（如 ``/path/to/model.joblib.xz``）。

    Returns:
        Any: 反序列化后的 Python 对象。

    Raises:
        FileNotFoundError: 当未找到任何分片文件时抛出。
    """
    directory = os.path.dirname(basepath)
    filename = os.path.basename(basepath)

    # 在同一目录查找 .part 文件
    parts = sorted(
        [
            os.path.join(directory, p)
            for p in os.listdir(directory)
            if p.startswith(filename + ".part")
        ]
    )

    if not parts:
        raise FileNotFoundError(f"未找到分片：{filename}.part*** 在目录 {directory}")

    merged_path = os.path.join(directory, filename + ".merged")

    # 合并
    with open(merged_path, "wb") as fout:
        for p in parts:
            with open(p, "rb") as fin:
                fout.write(fin.read())

    # 解压并加载对象
    with lzma.open(merged_path, "rb") as f:
        obj = joblib.load(f)

    return obj


def dump_compressed(obj, compressed_path: str):
    """使用 lzma 压缩并保存 Python 对象。

    该函数适用于中小规模对象的直接压缩存储，
    保存格式为 ``.xz`` 压缩的 joblib 文件。

    Args:
        obj (Any): 需要保存的 Python 对象（需可被 joblib 序列化）。
        compressed_path (str): 输出文件路径（通常以 ``.xz`` 结尾）。

    Returns:
        str: 实际保存的压缩文件路径。
    """

    with lzma.open(compressed_path, "wb") as f:
        joblib.dump(obj, f)

    print(f"[Workflow] compressed and saved to : {compressed_path}")
    return compressed_path


def load_compressed(filepath: str):
    """从 lzma 压缩的 joblib 文件中加载 Python 对象。

    该函数用于读取由 ``dump_compressed`` 生成的压缩文件，
    并返回反序列化后的对象。

    Args:
        filepath (str): 压缩的 joblib 文件路径（``.xz``）。

    Returns:
        Any: 反序列化后的 Python 对象。
    """
    with lzma.open(filepath, "rb") as f:
        obj = joblib.load(f)
    print(f"[Workflow] loaded compressed joblib from : {filepath}")
    return obj
