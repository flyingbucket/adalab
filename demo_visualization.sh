#!/bin/bash
# 可视化功能演示脚本
# 展示三种可视化方式的使用

echo "██████████████████████████████████████████████████████████"
echo "           可视化功能演示 - Visualization Demo           "
echo "██████████████████████████████████████████████████████████"
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ========== 方式3：从结果加载（最快） ==========
echo -e "${BLUE}【方式3】从已保存结果加载 - 最快！⭐${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo -e "${YELLOW}步骤1：查看可用实验${NC}"
echo "$ ls experiments/"
ls experiments/
echo ""

echo -e "${YELLOW}步骤2：快速查看摘要（无需绘图）${NC}"
echo "$ python visualize_from_results.py -e train_val_500rounds --no-plot"
echo ""
echo "执行中..."
python visualize_from_results.py -e train_val_500rounds --no-plot 2>/dev/null || echo "实验数据不存在，跳过"
echo ""

read -p "按 Enter 继续..."
echo ""

# ========== 对比实验 ==========
echo -e "${BLUE}【演示】批量对比多个实验${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo -e "${YELLOW}对比 baseline vs noise5${NC}"
echo "$ python visualize_from_results.py -e baseline_est500_depth2 --no-plot"
python visualize_from_results.py -e baseline_est500_depth2 --no-plot 2>/dev/null | head -20
echo ""
echo "$ python visualize_from_results.py -e noise5_est500_depth2 --no-plot"
python visualize_from_results.py -e noise5_est500_depth2 --no-plot 2>/dev/null | head -20
echo ""

read -p "按 Enter 继续..."
echo ""

# ========== 生成图表 ==========
echo -e "${BLUE}【演示】生成可视化图表${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo -e "${YELLOW}生成图表并保存${NC}"
echo "$ python visualize_from_results.py -e train_val_500rounds -s demo_result.png"
echo ""

if python visualize_from_results.py -e train_val_500rounds -s demo_result.png 2>/dev/null; then
    echo -e "${GREEN}✓ 图表已生成：demo_result.png${NC}"
    
    # 检查文件是否存在
    if [ -f "demo_result.png" ]; then
        ls -lh demo_result.png
        echo ""
        echo -e "${GREEN}✓ 可以使用以下命令查看：${NC}"
        echo "  open demo_result.png      # macOS"
        echo "  eog demo_result.png       # Linux"
        echo "  xdg-open demo_result.png  # Linux"
    fi
else
    echo "⚠️  实验数据不存在，跳过"
fi

echo ""
read -p "按 Enter 继续..."
echo ""

# ========== 方式1：原版脚本 ==========
echo -e "${BLUE}【方式1】使用 visualize_overfitting.py${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo -e "${YELLOW}特点：${NC}"
echo "  • 默认：基础过拟合分析（2个子图）"
echo "  • 可选：详细训练监控（+6个子图）"
echo "  • 启用方式：修改第138行 enable_detailed_monitoring = True"
echo ""

echo -e "${YELLOW}运行命令：${NC}"
echo "$ python visualize_overfitting.py"
echo ""
echo "注意：此方式需要重新训练，耗时 5-10 分钟"
echo ""

read -p "按 Enter 继续..."
echo ""

# ========== 方式2：增强版 ==========
echo -e "${BLUE}【方式2】使用 visualize_overfitting_enhanced.py${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo -e "${YELLOW}特点：${NC}"
echo "  • 完整版本，默认启用所有监控"
echo "  • 两阶段可视化（过拟合分析 + 详细监控）"
echo "  • 生成 8 个子图"
echo ""

echo -e "${YELLOW}运行命令：${NC}"
echo "$ python visualize_overfitting_enhanced.py"
echo ""
echo "注意：此方式需要重新训练，耗时 6-12 分钟"
echo ""

read -p "按 Enter 继续..."
echo ""

# ========== 总结 ==========
echo ""
echo "██████████████████████████████████████████████████████████"
echo "                         总结                            "
echo "██████████████████████████████████████████████████████████"
echo ""

echo -e "${GREEN}✓ 三种可视化方式：${NC}"
echo ""

echo "┌─────────────────────────────────────────────────────────┐"
echo "│ 方式1: visualize_overfitting.py                        │"
echo "│   • 时间：5-10分钟                                      │"
echo "│   • 图表：2个（可选+6个）                               │"
echo "│   • 适合：实时分析、参数调优                            │"
echo "└─────────────────────────────────────────────────────────┘"
echo ""

echo "┌─────────────────────────────────────────────────────────┐"
echo "│ 方式2: visualize_overfitting_enhanced.py               │"
echo "│   • 时间：6-12分钟                                      │"
echo "│   • 图表：8个                                           │"
echo "│   • 适合：深度研究、论文撰写                            │"
echo "└─────────────────────────────────────────────────────────┘"
echo ""

echo "┌─────────────────────────────────────────────────────────┐"
echo "│ 方式3: visualize_from_results.py ⭐ 推荐               │"
echo "│   • 时间：< 5秒                                         │"
echo "│   • 图表：6个                                           │"
echo "│   • 适合：快速查看、批量对比                            │"
echo "│   • 特点：直接读取 CSV/joblib，无需重新训练            │"
echo "└─────────────────────────────────────────────────────────┘"
echo ""

echo -e "${YELLOW}快速命令参考：${NC}"
echo ""
echo "# 查看所有实验"
echo "ls experiments/"
echo ""
echo "# 秒级可视化（推荐）"
echo "python visualize_from_results.py -e train_val_500rounds"
echo ""
echo "# 保存图表"
echo "python visualize_from_results.py -e train_val_500rounds -s result.png"
echo ""
echo "# 只看摘要"
echo "python visualize_from_results.py -e train_val_500rounds --no-plot"
echo ""
echo "# 批量对比"
echo "for exp in baseline_est500_depth2 noise5_est500_depth2; do"
echo "    python visualize_from_results.py -e \$exp -s \${exp}.png"
echo "done"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}✓ 演示完成！${NC}"
echo ""
echo "📚 详细文档："
echo "  • VISUALIZATION_METHODS.md       - 方法对比"
echo "  • EXPERIMENTS_INDEX.md           - 实验索引"
echo "  • docs/visualize_from_results_guide.md - 详细指南"
echo "  • VISUALIZATION_SUMMARY.md       - 功能总结"
echo ""
echo "🚀 立即尝试："
echo "  python visualize_from_results.py -e train_val_500rounds"
echo "██████████████████████████████████████████████████████████"

