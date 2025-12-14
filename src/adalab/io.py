import os
import joblib
import lzma


def dump_compressed_chunks(obj, filepath: str, chunk_size_mb=50):
    """
    压缩并切片保存 Python 对象。
    chunk_size_mb: 每块大小（默认 50MB）
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
    """
    basepath: 完整路径，如 /path/to/model.joblib.xz
    自动在同目录查找 model.joblib.xz.part000, part001, ...
    忽略原始文件、merged 文件
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


def dump_compressed(obj, filepath: str):
    """
    使用 lzma 压缩并保存任意 Python 对象
    """
    compressed_path = filepath + ".xz"

    with lzma.open(compressed_path, "wb") as f:
        joblib.dump(obj, f)

    print(f"[Workflow] compressed and saved to : {compressed_path}")
    return compressed_path


def load_compressed(filepath: str):
    """
    自动读取 .xz 压缩的 joblib 文件
    """
    with lzma.open(filepath, "rb") as f:
        obj = joblib.load(f)
    print(f"[Workflow] loaded compressed joblib from : {filepath}")
    return obj
