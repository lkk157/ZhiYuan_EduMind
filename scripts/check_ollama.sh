#!/usr/bin/env bash
# ============================================================
# 知源 · 环境自检脚本：Ollama 连通性 / 三模型清单 / GPU 显存基线
#
# 为什么需要它（昨天排查 R2 网络封锁就是靠分层探测的思路）：
# 1. 答辩演示前跑一遍 = 快速确认环境就绪、顺便预热模型；
# 2. 出问题时先跑它，能立刻分清是「服务没起 / 模型没装 / 显存不够」哪一层。
#
# 用法（在项目根目录）：bash scripts/check_ollama.sh
# ============================================================
set -u

# 切到项目根目录（脚本在 scripts/ 下），保证能读到 .env
cd "$(dirname "$0")/.." || exit 1

echo "===== 1. Ollama 服务连通性 ====="
if curl -s --max-time 5 http://127.0.0.1:11434/api/tags -o /dev/null; then
  echo "[OK] Ollama API 可连接 (127.0.0.1:11434)"
else
  echo "[FAIL] 无法连接 Ollama——请先启动 Ollama 应用"
  exit 1
fi

echo ""
echo "===== 2. 已安装模型清单 ====="
ollama list 2>/dev/null

echo ""
echo "===== 3. .env 配置的三个模型是否在列 ====="
# 逐个核对 .env 里的模型名是否出现在 ollama list 中
# （模型名以 ollama list 的 NAME 为准——CLAUDE.md §5）
check_model() {
  local key="$1"
  local name
  name=$(grep -E "^${key}=" .env 2>/dev/null | head -1 | cut -d= -f2- | tr -d '\r' | xargs)
  if [ -z "$name" ]; then
    echo "[FAIL] ${key}: .env 中未配置"
    return
  fi
  # 匹配 name 或 name:tag（tag 可省略，如 glm-ocr 能配 glm-ocr:latest）
  if ollama list 2>/dev/null | awk '{print $1}' | grep -qE "^${name}(:|$)"; then
    echo "[OK] ${key}=${name}"
  else
    echo "[FAIL] ${key}=${name} 不在模型列表中"
  fi
}

check_model LLM_MODEL
check_model OCR_MODEL
check_model EMBED_MODEL

echo ""
echo "===== 4. OCR 模型不可用时的 fallback 提示 ====="
if ollama list 2>/dev/null | awk '{print $1}' | grep -qE "^$(grep -E '^OCR_MODEL=' .env | cut -d= -f2- | tr -d '\r' | xargs)(:|$)"; then
  echo "[OK] OCR 模型就绪（GLM-OCR：PPT 图表/公式识别用）"
else
  echo "[HINT] OCR 模型缺失：优先 ollama pull glm-ocr；网络受限时用 ModelScope 镜像或 GGUF+ollama create，"
  echo "       再不行 fallback qwen2.5-vl:3b（改 .env 的 OCR_MODEL 一行即可）"
fi

echo ""
echo "===== 5. GPU 显存基线 ====="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
  echo "(演示前建议记录此刻显存占用作为基线，入库/问答峰值不应打满 8G)"
else
  echo "[HINT] 未找到 nvidia-smi（无独显驱动或不在 PATH）"
fi

echo ""
echo "===== 自检完成 ====="
