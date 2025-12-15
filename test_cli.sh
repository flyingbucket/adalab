#!/bin/bash
# CLI功能测试脚本

echo "========================================="
echo "AdaLab CLI 功能测试"
echo "========================================="

cd "$(dirname "$0")"

echo ""
echo "1. 测试主帮助信息"
echo "-----------------------------------------"
python main.py --help

echo ""
echo "2. 测试train子命令帮助"
echo "-----------------------------------------"
python main.py train --help

echo ""
echo "3. 测试evaluate子命令帮助"
echo "-----------------------------------------"
python main.py evaluate --help

echo ""
echo "4. 测试visualize子命令帮助"
echo "-----------------------------------------"
python main.py visualize --help

echo ""
echo "5. 测试版本信息"
echo "-----------------------------------------"
python main.py --version

echo ""
echo "========================================="
echo "✅ CLI接口测试完成！"
echo "========================================="

