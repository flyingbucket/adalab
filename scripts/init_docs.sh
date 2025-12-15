#!/usr/bin/env bash
set -e

DOCS_DIR="mk-docs"

echo "[init] Initializing MkDocs documentation structure in '${DOCS_DIR}/'"

# -------------------------
# Create directories
# -------------------------
mkdir -p "${DOCS_DIR}/backend"
mkdir -p "${DOCS_DIR}/visualization"

# -------------------------
# Helper: create file if not exists
# -------------------------
create_file() {
  local file="$1"
  local content="$2"

  if [ -f "$file" ]; then
    echo "[skip] $file already exists"
  else
    echo "[create] $file"
    printf "%s\n" "$content" >"$file"
  fi
}

# -------------------------
# index.md
# -------------------------
create_file "${DOCS_DIR}/index.md" \
  "# adalab Documentation

本网站为 **adalab 实验平台** 的官方文档。

内容包括：
- 后端（adalab）API 文档
- 实验工作流与监控数据说明
- 可视化模块（adalab_viz）的设计与接口说明

本文档由 MkDocs + mkdocstrings 自动生成，  
API 内容直接来源于代码 docstring。
"

# -------------------------
# Backend docs
# -------------------------
create_file "${DOCS_DIR}/backend/overview.md" \
  "# Backend Overview

adalab 是实验平台的后端模块，主要负责：

- 数据准备与噪声注入
- AdaBoost 训练流程与监控
- 训练后评估与结果导出

以下文档均由代码 docstring 自动生成。
"

create_file "${DOCS_DIR}/backend/workflow.md" \
  "# Workflow

::: adalab.workflow
"

create_file "${DOCS_DIR}/backend/monitor.md" \
  "# Monitor

::: adalab.monitor
"

create_file "${DOCS_DIR}/backend/patch.md" \
  "# Patch

::: adalab.patch
"

create_file "${DOCS_DIR}/backend/evaluation.md" \
  "# Evaluation

::: adalab.evaluation
"

create_file "${DOCS_DIR}/backend/data.md" \
  "# Data Preparation

::: adalab.data
"

create_file "${DOCS_DIR}/backend/io.md" \
  "# IO Utilities

::: adalab.io
"

# -------------------------
# Visualization docs (placeholders)
# -------------------------
create_file "${DOCS_DIR}/visualization/overview.md" \
  "# Visualization (adalab_viz)

adalab_viz 是 adalab 实验平台的可视化前端模块。

该模块主要负责：
- 读取后端生成的实验结果
- 对训练过程与评估曲线进行可视化展示
- 支持不同实验配置的对比分析

当前文档为结构占位，具体 API 将在后续补充。
"

create_file "${DOCS_DIR}/visualization/concepts.md" \
  "# Visualization Concepts

本节用于说明可视化模块的设计理念与核心概念，
例如：
- 实验结果组织方式
- 曲线与指标的语义
- 与后端数据结构的对应关系
"

echo "[done] MkDocs documentation structure initialized."
