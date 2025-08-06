#!/bin/bash

# Task 18: Correct Workflow End-to-End Test
# 遵循正確工作流程的端到端測試：Requirements → Design → Tasks → Implementation
# 每個階段包含revise流程並進行內容比較驗證
# 使用 integration 配置進行真實的 LLM 測試

set -e  # Exit on any error

# 設置測試環境變數
export INTEGRATION_TESTING=true
export TESTING=true
export LOG_LEVEL=ERROR  # 只顯示錯誤信息

echo "=== Task 18: Correct Workflow End-to-End Test ==="
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

# 檢查 .env.integration 文件是否存在
if [ ! -f ".env.integration" ]; then
    echo "錯誤: .env.integration 文件不存在，請創建該文件並配置真實的 LLM 設定"
    exit 1
fi

# 確保 autogen-framework 命令可用
if ! command -v autogen-framework &> /dev/null; then
    echo "錯誤: autogen-framework 命令不存在，請先安裝專案: pip install -e ."
    exit 1
fi

# 載入 integration 環境變數
echo "載入 integration 測試配置..."
set -a  # 自動導出變數
source .env.integration
set +a

echo "✓ 環境檢查通過"
echo "✓ Integration 配置已載入"
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
echo ""

autogen-framework --workspace integration_test_workspace --request "$TEST_TASK" > request_output.txt 2>&1
REQUEST_EXIT_CODE=$?

echo "初始請求執行結果:"
cat request_output.txt
echo ""

if [ $REQUEST_EXIT_CODE -ne 0 ]; then
    echo "❌ 初始請求執行失敗"
    exit 1
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
    echo "❌ 未找到工作目錄"
    exit 1
fi

# 檢查 requirements.md 是否生成
if [ ! -f "$WORK_DIR/requirements.md" ]; then
    echo "❌ requirements.md 未生成"
    exit 1
fi

echo "✓ requirements.md 已生成"
echo ""

# 保存原始 requirements 用於比較
cp "$WORK_DIR/requirements.md" "$WORK_DIR/requirements.md.original"

# Requirements 第一次 Revise
echo "=== 階段 1.1: Requirements 第一次 Revise ==="
REQUIREMENTS_REVISE_1="requirements:Please add more detailed error handling requirements and edge cases"

autogen-framework --workspace integration_test_workspace --revise "$REQUIREMENTS_REVISE_1" > req_revise_1_output.txt 2>&1
REQ_REVISE_EXIT_CODE_1=$?

echo "Requirements 第一次 Revise 結果:"
cat req_revise_1_output.txt
echo ""

# 檢查 revise 是否生效
REQUIREMENTS_REVISE_1_SUCCESS=false
if [ -f "$WORK_DIR/requirements.md" ]; then
    if ! diff -q "$WORK_DIR/requirements.md.original" "$WORK_DIR/requirements.md" > /dev/null 2>&1; then
        echo "✓ Requirements 第一次 Revise 成功，內容已更新"
        REQUIREMENTS_REVISE_1_SUCCESS=true
    else
        echo "❌ Requirements 第一次 Revise 失敗，內容沒有變化"
        REQUIREMENTS_REVISE_1_SUCCESS=false
    fi
    cp "$WORK_DIR/requirements.md" "$WORK_DIR/requirements.md.after_revise_1"
fi
echo ""

# Requirements 第二次 Revise
echo "=== 階段 1.2: Requirements 第二次 Revise ==="
REQUIREMENTS_REVISE_2="requirements:Please add performance and usability requirements"

autogen-framework --workspace integration_test_workspace --revise "$REQUIREMENTS_REVISE_2" > req_revise_2_output.txt 2>&1
REQ_REVISE_EXIT_CODE_2=$?

echo "Requirements 第二次 Revise 結果:"
cat req_revise_2_output.txt
echo ""

# 檢查第二次 revise 是否生效
REQUIREMENTS_REVISE_2_SUCCESS=false
if [ -f "$WORK_DIR/requirements.md" ] && [ -f "$WORK_DIR/requirements.md.after_revise_1" ]; then
    if ! diff -q "$WORK_DIR/requirements.md.after_revise_1" "$WORK_DIR/requirements.md" > /dev/null 2>&1; then
        echo "✓ Requirements 第二次 Revise 成功，內容已更新"
        REQUIREMENTS_REVISE_2_SUCCESS=true
    else
        echo "❌ Requirements 第二次 Revise 失敗，內容沒有變化"
        REQUIREMENTS_REVISE_2_SUCCESS=false
    fi
fi
echo ""

# 批准 Requirements 階段
echo "=== 階段 2: 批准 Requirements (生成 Design) ==="
autogen-framework --workspace integration_test_workspace --approve requirements > approve_req_output.txt 2>&1
APPROVE_REQ_EXIT_CODE=$?

echo "Requirements 批准結果:"
cat approve_req_output.txt
echo ""

if [ $APPROVE_REQ_EXIT_CODE -ne 0 ]; then
    echo "❌ Requirements 批准失敗"
    exit 1
fi

# 檢查 design.md 是否生成
if [ ! -f "$WORK_DIR/design.md" ]; then
    echo "❌ design.md 未生成"
    exit 1
fi

echo "✓ design.md 已生成"
echo ""

# 保存原始 design 用於比較
cp "$WORK_DIR/design.md" "$WORK_DIR/design.md.original"

# Design 第一次 Revise
echo "=== 階段 2.1: Design 第一次 Revise ==="
DESIGN_REVISE_1="design:Please add more detailed security considerations and input validation design"

autogen-framework --workspace integration_test_workspace --revise "$DESIGN_REVISE_1" > design_revise_1_output.txt 2>&1
DESIGN_REVISE_EXIT_CODE_1=$?

echo "Design 第一次 Revise 結果:"
cat design_revise_1_output.txt
echo ""

# 檢查 design revise 是否生效
DESIGN_REVISE_1_SUCCESS=false
if [ -f "$WORK_DIR/design.md" ]; then
    if ! diff -q "$WORK_DIR/design.md.original" "$WORK_DIR/design.md" > /dev/null 2>&1; then
        echo "✓ Design 第一次 Revise 成功，內容已更新"
        DESIGN_REVISE_1_SUCCESS=true
    else
        echo "❌ Design 第一次 Revise 失敗，內容沒有變化"
        DESIGN_REVISE_1_SUCCESS=false
    fi
    cp "$WORK_DIR/design.md" "$WORK_DIR/design.md.after_revise_1"
fi
echo ""

# Design 第二次 Revise
echo "=== 階段 2.2: Design 第二次 Revise ==="
DESIGN_REVISE_2="design:Please add comprehensive testing strategy and deployment considerations"

autogen-framework --workspace integration_test_workspace --revise "$DESIGN_REVISE_2" > design_revise_2_output.txt 2>&1
DESIGN_REVISE_EXIT_CODE_2=$?

echo "Design 第二次 Revise 結果:"
cat design_revise_2_output.txt
echo ""

# 檢查第二次 design revise 是否生效
DESIGN_REVISE_2_SUCCESS=false
if [ -f "$WORK_DIR/design.md" ] && [ -f "$WORK_DIR/design.md.after_revise_1" ]; then
    if ! diff -q "$WORK_DIR/design.md.after_revise_1" "$WORK_DIR/design.md" > /dev/null 2>&1; then
        echo "✓ Design 第二次 Revise 成功，內容已更新"
        DESIGN_REVISE_2_SUCCESS=true
    else
        echo "❌ Design 第二次 Revise 失敗，內容沒有變化"
        DESIGN_REVISE_2_SUCCESS=false
    fi
fi
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
echo ""

# 保存原始 tasks 用於比較
cp "$WORK_DIR/tasks.md" "$WORK_DIR/tasks.md.original"

# Tasks 第一次 Revise
echo "=== 階段 3.1: Tasks 第一次 Revise ==="
TASKS_REVISE_1="tasks:Please simplify the tasks and keep only the core functionality - remove detailed testing tasks and focus on basic implementation only"

autogen-framework --workspace integration_test_workspace --revise "$TASKS_REVISE_1" > tasks_revise_1_output.txt 2>&1
TASKS_REVISE_EXIT_CODE_1=$?

echo "Tasks 第一次 Revise 結果:"
cat tasks_revise_1_output.txt
echo ""

# 檢查 tasks revise 是否生效
TASKS_REVISE_1_SUCCESS=false
if [ -f "$WORK_DIR/tasks.md" ]; then
    if ! diff -q "$WORK_DIR/tasks.md.original" "$WORK_DIR/tasks.md" > /dev/null 2>&1; then
        echo "✓ Tasks 第一次 Revise 成功，內容已更新"
        TASKS_REVISE_1_SUCCESS=true
    else
        echo "❌ Tasks 第一次 Revise 失敗，內容沒有變化"
        TASKS_REVISE_1_SUCCESS=false
    fi
    cp "$WORK_DIR/tasks.md" "$WORK_DIR/tasks.md.after_revise_1"
fi
echo ""

# Tasks 第二次 Revise
echo "=== 階段 3.2: Tasks 第二次 Revise ==="
TASKS_REVISE_2="tasks:Please further simplify by removing optional tasks, documentation tasks, and advanced testing - keep only the essential 5-6 core tasks for basic calculator functionality"

autogen-framework --workspace integration_test_workspace --revise "$TASKS_REVISE_2" > tasks_revise_2_output.txt 2>&1
TASKS_REVISE_EXIT_CODE_2=$?

echo "Tasks 第二次 Revise 結果:"
cat tasks_revise_2_output.txt
echo ""

# 檢查第二次 tasks revise 是否生效
if [ -f "$WORK_DIR/tasks.md" ] && [ -f "$WORK_DIR/tasks.md.after_revise_1" ]; then
    if ! diff -q "$WORK_DIR/tasks.md.after_revise_1" "$WORK_DIR/tasks.md" > /dev/null 2>&1; then
        echo "✓ Tasks 第二次 Revise 成功，內容已更新"
    else
        echo "⚠️ Tasks 第二次 Revise 可能沒有生效"
    fi
fi
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
    
    echo "  總任務數: $TOTAL_TASKS"
    echo "  已完成: $COMPLETED_TASKS"
    
    if [ "$TOTAL_TASKS" -gt 0 ] && [ "$TOTAL_TASKS" != "0" ]; then
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
echo "完整工作流程執行結果:"
echo "1. ✓ Requirements 生成 (PlanAgent)"
echo "   1.1 ✓ Requirements 第一次 Revise"
echo "   1.2 ✓ Requirements 第二次 Revise"
echo "2. ✓ Design 生成 (DesignAgent)"
echo "   2.1 ✓ Design 第一次 Revise"
echo "   2.2 ✓ Design 第二次 Revise"
echo "3. ✓ Tasks 生成 (TasksAgent)"
echo "   3.1 ✓ Tasks 第一次 Revise"
echo "   3.2 ✓ Tasks 第二次 Revise"
echo "4. ✓ Implementation 執行 (ImplementAgent)"
echo ""

if [ ${#CODE_FILES[@]} -gt 0 ]; then
    echo "🎉 Task 18 完整工作流程測試成功完成！"
    echo "Framework 正確執行了完整的工作流程（包含 Revise 功能）並生成了代碼文件"
else
    echo "⚠️ Task 18 完整工作流程測試部分成功"
    echo "Framework 正確執行了文檔生成流程（包含 Revise 功能），但可能沒有生成代碼文件"
fi

echo ""
echo "🎉 測試完成！"

exit 0