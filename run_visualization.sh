#!/bin/bash
# 批量自动可视化 experiments 下所有实验
# 仅对含 monitor.joblib.xz 的实验执行可视化

set -e

VIS_DIR="./experiments/visualization"
mkdir -p "$VIS_DIR"

echo "================ 批量可视化开始 ================"
echo "输出目录：$VIS_DIR"
echo ""

# 遍历 experiments/*/
for exp_dir in experiments/*/; do
  exp_name=$(basename "$exp_dir")
  monitor_file="$exp_dir/results/monitor.joblib.xz"

  # 判断 monitor 是否存在
  if [ -f "$monitor_file" ]; then
    echo "→ 发现实验：$exp_name"
    echo "  monitor 文件：$monitor_file"

    out_path="$VIS_DIR/${exp_name}.png"
    echo "  正在生成: $out_path"

    # 调用你的可视化脚本
    python visualize_from_results.py --experiment "$exp_name" --save "$out_path" 2>/dev/null

    if [ $? -eq 0 ] && [ -f "$out_path" ]; then
      echo "  ✓ 完成 ($out_path)"
    else
      echo "  ⚠️ 可视化失败：$exp_name"
    fi

    echo ""
  else
    echo "跳过：$exp_name （没有 monitor.joblib.xz）"
  fi
done

echo "================ 批量可视化结束 ================"
echo "全部图片已保存到：$VIS_DIR/"
