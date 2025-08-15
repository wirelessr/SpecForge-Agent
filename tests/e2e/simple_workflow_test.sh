#!/bin/bash

# 簡化的端到端工作流程測試
# 使用 integration 配置測試完整的工作流程：Requirements → Design → Tasks → Implementation

set -e  # Exit on any error

# 設置測試環境變數
export INTEGRATION_TESTING=true
export TESTING=true
export LOG_LEVEL=ERROR  # 只顯示錯誤信息

echo "=== AutoGen Framework 端到端工作流程測試 ==="
echo "開始時間: $(date)"
echo ""

# 測試配置
TEST_TASK="Create a simple calculator function that can add, subtract, multiply, and divide two numbers"

# 清理函數
cleanup_test_files() {
    echo "正在清理測試檔案..."
    rm -rf integration_test_workspace
    rm -f *.txt
    echo "測試檔案清理完成"
}

# 設置 trap 來確保清理
trap cleanup_test_files EXIT

echo "=== 環境檢查 ==="

# 確保在正確的專案根目錄中執行
if [ ! -f "pyproject.toml" ] || [ ! -d "autogen_framework" ]; then
    echo "錯誤: 請在專案根目錄中執行此測試腳本"
    exit 1
fi

# 檢查 .env 文件是否存在
if [ ! -f ".env" ]; then
    echo "錯誤: .env 文件不存在，請創建該文件並配置真實的 LLM 設定"
    exit 1
fi

# 確保 autogen-framework 命令可用
if ! command -v autogen-framework &> /dev/null; then
    echo "錯誤: autogen-framework 命令不存在，請先安裝專案: pip install -e ."
    exit 1
fi

# 載入環境變數
echo "載入測試配置..."
set -a  # 自動導出變數
source .env
set +a

echo "✓ 環境檢查通過"
echo "✓ 配置已載入"
echo "  - LLM Model: $LLM_MODEL"
echo "  - LLM Base URL: $LLM_BASE_URL"
echo ""

# 清理並創建測試工作空間
echo "=== 準備測試環境 ==="
rm -rf integration_test_workspace
mkdir -p integration_test_workspace/memory/global
mkdir -p integration_test_workspace/memory/projects
mkdir -p integration_test_workspace/logs
echo "✓ 測試工作空間已創建"
echo ""

# 重置會話
echo "=== 重置會話 ==="
autogen-framework --workspace integration_test_workspace --reset-session > reset_output.txt 2>&1
if [ $? -eq 0 ]; then
    echo "✓ 會話重置成功"
else
    echo "⚠️ 會話重置失敗，但繼續測試"
    cat reset_output.txt
fi
echo ""

# 處理初始請求 (Requirements 階段)
echo "=== 階段 1: 處理初始請求 (生成 Requirements) ==="
echo "測試任務: $TEST_TASK"

autogen-framework --workspace integration_test_workspace --request "$TEST_TASK" > request_output.txt 2>&1
REQUEST_EXIT_CODE=$?

if [ $REQUEST_EXIT_CODE -ne 0 ]; then
    echo "❌ 初始請求執行失敗"
    echo "錯誤詳情:"
    cat request_output.txt
    exit 1
else
    echo "✓ Requirements 生成成功"
fi

# 查找工作目錄
echo "查找生成的工作目錄..."
WORK_DIR=""

# 從 framework 狀態獲取工作目錄
STATUS_OUTPUT=$(autogen-framework --workspace integration_test_workspace --status 2>/dev/null)
WORK_DIR_FULL=$(echo "$STATUS_OUTPUT" | grep -o '"work_directory": "[^"]*"' | sed 's/"work_directory": "//g' | sed 's/"//g')

if [ -n "$WORK_DIR_FULL" ] && [ -d "$WORK_DIR_FULL" ]; then
    WORK_DIR="$WORK_DIR_FULL"
    echo "✓ 找到工作目錄: $WORK_DIR"
else
    # 備用方法：查找最新創建的目錄
    WORK_DIR=$(find integration_test_workspace -maxdepth 1 -type d -not -name "memory" -not -name "logs" -not -name "integration_test_workspace" 2>/dev/null | head -1)
    if [ -n "$WORK_DIR" ] && [ -d "$WORK_DIR" ]; then
        echo "✓ 找到工作目錄: $WORK_DIR"
    else
        echo "❌ 未找到工作目錄"
        echo "Integration test workspace 內容:"
        ls -la integration_test_workspace/
        exit 1
    fi
fi

# 檢查 requirements.md 是否生成
if [ ! -f "$WORK_DIR/requirements.md" ]; then
    echo "❌ requirements.md 未生成"
    exit 1
fi

echo "✓ requirements.md 已生成"
echo "Requirements 內容預覽:"
head -10 "$WORK_DIR/requirements.md"
echo ""

# 批准 Requirements 階段
echo "=== 階段 2: 批准 Requirements (生成 Design) ==="
autogen-framework --workspace integration_test_workspace --approve requirements > approve_req_output.txt 2>&1
APPROVE_REQ_EXIT_CODE=$?

if [ $APPROVE_REQ_EXIT_CODE -ne 0 ]; then
    echo "❌ Requirements 批准失敗"
    echo "錯誤詳情:"
    cat approve_req_output.txt
    exit 1
else
    echo "✓ Design 生成成功"
fi

# 檢查 design.md 是否生成
if [ ! -f "$WORK_DIR/design.md" ]; then
    echo "❌ design.md 未生成"
    exit 1
fi

echo "✓ design.md 已生成"
echo "Design 內容預覽:"
head -10 "$WORK_DIR/design.md"
echo ""

# 批准 Design 階段
echo "=== 階段 3: 批准 Design (生成 Tasks) ==="
autogen-framework --workspace integration_test_workspace --approve design > approve_design_output.txt 2>&1
APPROVE_DESIGN_EXIT_CODE=$?

echo "Design 批准結果:"
cat approve_design_output.txt
echo ""

if [ $APPROVE_DESIGN_EXIT_CODE -ne 0 ]; then
    echo "❌ Design 批准失敗"
    exit 1
fi

# 檢查 tasks.md 是否生成
if [ ! -f "$WORK_DIR/tasks.md" ]; then
    echo "❌ tasks.md 未生成"
    exit 1
fi

echo "✓ tasks.md 已生成"
echo "Tasks 內容預覽:"
head -15 "$WORK_DIR/tasks.md"
echo ""

# 批准 Tasks 階段 (執行 Implementation)
echo "=== 階段 4: 批准 Tasks (執行 Implementation) ==="
autogen-framework --workspace integration_test_workspace --approve tasks > approve_tasks_output.txt 2>&1
APPROVE_TASKS_EXIT_CODE=$?

echo "Tasks 批准結果:"
cat approve_tasks_output.txt
echo ""

if [ $APPROVE_TASKS_EXIT_CODE -ne 0 ]; then
    echo "❌ Tasks 批准失敗"
    exit 1
fi

echo "✓ Tasks 批准成功，Implementation 階段已執行"
echo ""

# 檢查執行結果
echo "=== 階段 5: 檢查執行結果 ==="

# 檢查工作目錄內容
echo "工作目錄內容:"
ls -la "$WORK_DIR"
echo ""

# 檢查是否有生成的代碼文件
CODE_FILES=($(find "$WORK_DIR" -name "*.py" -type f 2>/dev/null))

if [ ${#CODE_FILES[@]} -gt 0 ]; then
    echo "✓ 找到生成的代碼文件:"
    for file in "${CODE_FILES[@]}"; do
        echo "  - $file"
    done
    
    # 顯示主要代碼文件內容
    MAIN_CODE_FILE="${CODE_FILES[0]}"
    echo ""
    echo "主要代碼文件內容 ($MAIN_CODE_FILE):"
    echo "===================="
    cat "$MAIN_CODE_FILE"
    echo "===================="
    
    # 語法檢查
    if python -m py_compile "$MAIN_CODE_FILE" 2>/dev/null; then
        echo "✓ 代碼語法正確"
    else
        echo "⚠️ 代碼可能有語法錯誤"
    fi
else
    echo "⚠️ 未找到生成的 Python 代碼文件"
fi

# 檢查 tasks.md 中的任務完成狀態
echo ""
echo "檢查任務完成狀態:"
if [ -f "$WORK_DIR/tasks.md" ]; then
    TOTAL_TASKS=$(grep -c "^- \[" "$WORK_DIR/tasks.md" 2>/dev/null || echo "0")
    COMPLETED_TASKS=$(grep -c "^- \[x\]" "$WORK_DIR/tasks.md" 2>/dev/null || echo "0")
    
    # 確保變量是純數字
    TOTAL_TASKS=$(echo "$TOTAL_TASKS" | tr -d '\n\r' | grep -o '[0-9]*' | head -1)
    COMPLETED_TASKS=$(echo "$COMPLETED_TASKS" | tr -d '\n\r' | grep -o '[0-9]*' | head -1)
    
    # 如果變量為空，設為0
    TOTAL_TASKS=${TOTAL_TASKS:-0}
    COMPLETED_TASKS=${COMPLETED_TASKS:-0}
    
    echo "  總任務數: $TOTAL_TASKS"
    echo "  已完成: $COMPLETED_TASKS"
    
    if [ "$TOTAL_TASKS" -gt 0 ]; then
        COMPLETION_RATE=$((COMPLETED_TASKS * 100 / TOTAL_TASKS))
        echo "  完成率: $COMPLETION_RATE%"
    fi
else
    echo "⚠️ tasks.md 文件不存在"
fi

echo ""

# 最終總結
echo "=== 測試總結 ==="
echo "測試完成時間: $(date)"
echo ""

# 檢查所有預期文檔是否存在
EXPECTED_DOCS=("requirements.md" "design.md" "tasks.md")
MISSING_DOCS=()

for doc in "${EXPECTED_DOCS[@]}"; do
    if [ ! -f "$WORK_DIR/$doc" ]; then
        MISSING_DOCS+=("$doc")
    fi
done

if [ ${#MISSING_DOCS[@]} -eq 0 ]; then
    echo "✅ 所有預期文檔都已生成: ${EXPECTED_DOCS[*]}"
else
    echo "❌ 缺少以下文檔: ${MISSING_DOCS[*]}"
fi

echo ""
echo "工作流程執行結果:"
echo "1. ✓ Requirements 生成 (PlanAgent)"
echo "2. ✓ Design 生成 (DesignAgent)" 
echo "3. ✓ Tasks 生成 (TasksAgent)"
echo "4. ✓ Implementation 執行 (ImplementAgent)"
echo ""

if [ ${#CODE_FILES[@]} -gt 0 ]; then
    echo "🎉 端到端測試成功完成！"
    echo "Framework 正確執行了完整的工作流程並生成了代碼文件"
else
    echo "⚠️ 端到端測試部分成功"
    echo "Framework 正確執行了文檔生成流程，但可能沒有生成代碼文件"
fi

echo ""
echo "🎉 測試完成！"

exit 0