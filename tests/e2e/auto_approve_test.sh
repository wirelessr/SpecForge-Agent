#!/bin/bash

# Auto-Approve End-to-End Test
# Tests that auto-approve mode creates a working Hello World Python file using real agents

set -e  # Exit on any error

# 設置測試環境變數
export INTEGRATION_TESTING=true
export TESTING=true

echo "=== Auto-Approve End-to-End Test ==="
echo "開始時間: $(date)"
echo ""

# 清理函數
cleanup_test_files() {
    echo "正在清理測試檔案..."
    # 返回到項目根目錄
    cd ..
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

# 創建artifacts目錄（如果不存在）
mkdir -p artifacts

# 切換到artifacts目錄，讓生成的產出物放在正確位置
echo "📁 切換到artifacts目錄進行測試"
cd artifacts

# 檢查 .env.integration 文件是否存在（在項目根目錄）
if [ ! -f "../.env.integration" ]; then
    echo "錯誤: .env.integration 文件不存在，請創建該文件並配置真實的 LLM 設定"
    exit 1
fi

# 確保 autogen-framework 命令可用
if ! command -v autogen-framework &> /dev/null; then
    echo "錯誤: autogen-framework 命令不存在，請先安裝專案: pip install -e ."
    exit 1
fi

echo "✅ 環境檢查通過"
echo ""

# 重置 session
echo "=== 重置 Session ==="
autogen-framework --reset-session
echo "✅ Session 重置完成"
echo ""

# 提交 auto-approve 請求
echo "=== 提交 Auto-Approve 請求 ==="
REQUEST="Create a simple Python file that prints 'Hello, World!' when executed. The file should have a main function and proper Python structure."

echo "請求內容: $REQUEST"
echo ""

# 使用 auto-approve 模式
echo "⚡ 使用 auto-approve 模式處理請求..."
autogen-framework --request "$REQUEST" --auto-approve

if [ $? -ne 0 ]; then
    echo "❌ Auto-approve 請求失敗"
    exit 1
fi

echo "✅ Auto-approve 請求處理完成"
echo ""

# 檢查狀態
echo "=== 檢查框架狀態 ==="
autogen-framework --status
echo ""

# 尋找生成的工作目錄
echo "=== 尋找生成的工作目錄 ==="

# 尋找工作目錄 - 按照正確的文件組織原則檢查artifacts/outputs/目錄
WORK_DIRS=()

# 方法1: 在artifacts/outputs/中尋找 the-user-* 目錄
for dir in $(find artifacts/outputs -maxdepth 1 -type d -name "the-user-*" 2>/dev/null); do
    WORK_DIRS+=("$dir")
done

# 方法2: 在artifacts/outputs/中尋找 create-* 目錄
for dir in $(find artifacts/outputs -maxdepth 1 -type d -name "create-*" 2>/dev/null); do
    WORK_DIRS+=("$dir")
done

# 方法3: 在artifacts/outputs/中尋找包含 requirements.md, design.md, tasks.md 的目錄
for dir in $(find artifacts/outputs -maxdepth 1 -type d 2>/dev/null); do
    if [ -f "$dir/requirements.md" ] && [ -f "$dir/design.md" ] && [ -f "$dir/tasks.md" ]; then
        WORK_DIRS+=("$dir")
    fi
done

# 方法4: 如果artifacts/outputs不存在或為空，檢查頂層目錄（向後兼容）
if [ ${#WORK_DIRS[@]} -eq 0 ]; then
    echo "⚠️  在artifacts/outputs/中未找到工作目錄，檢查頂層目錄..."
    
    for dir in $(find . -maxdepth 1 -type d -name "the-user-*" -o -name "create-*" 2>/dev/null); do
        if [ -f "$dir/requirements.md" ] && [ -f "$dir/design.md" ] && [ -f "$dir/tasks.md" ]; then
            WORK_DIRS+=("$dir")
            echo "⚠️  發現工作目錄在頂層: $dir (應該移動到artifacts/outputs/)"
        fi
    done
fi

if [ ${#WORK_DIRS[@]} -eq 0 ]; then
    echo "❌ 找不到工作目錄"
    echo ""
    echo "檢查artifacts/outputs/目錄:"
    if [ -d "artifacts/outputs" ]; then
        ls -la artifacts/outputs/
    else
        echo "artifacts/outputs/目錄不存在"
    fi
    echo ""
    echo "檢查頂層目錄:"
    ls -la
    echo ""
    echo "尋找所有包含 requirements.md 的目錄:"
    find . -name "requirements.md" -type f 2>/dev/null
    exit 1
fi

# 使用第一個找到的工作目錄
WORK_DIR="${WORK_DIRS[0]}"
echo "📂 找到工作目錄: $WORK_DIR"

# 檢查必要文件是否存在
echo ""
echo "=== 檢查生成的文件 ==="

REQUIRED_FILES=("requirements.md" "design.md" "tasks.md")
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$WORK_DIR/$file" ]; then
        echo "✅ $file 存在"
    else
        echo "❌ $file 不存在"
        exit 1
    fi
done

# 尋找 Python 文件
echo ""
echo "=== 尋找 Python 文件 ==="
PYTHON_FILES=($(find "$WORK_DIR" -name "*.py" -type f))

if [ ${#PYTHON_FILES[@]} -eq 0 ]; then
    echo "❌ 找不到 Python 文件"
    echo "$WORK_DIR 目錄內容:"
    ls -la "$WORK_DIR"
    exit 1
fi

# 使用第一個找到的 Python 文件
PYTHON_FILE="${PYTHON_FILES[0]}"
echo "🐍 找到 Python 文件: $PYTHON_FILE"

# 顯示 Python 文件內容
echo ""
echo "=== Python 文件內容 ==="
echo "----------------------------------------"
cat "$PYTHON_FILE"
echo "----------------------------------------"

# 驗證 Python 文件內容
echo ""
echo "=== 驗證 Python 文件內容 ==="

if grep -qi "hello" "$PYTHON_FILE" && grep -qi "world" "$PYTHON_FILE"; then
    echo "✅ 包含 'Hello' 和 'World'"
else
    echo "❌ 缺少 'Hello' 或 'World'"
    exit 1
fi

if grep -q "def main\|def " "$PYTHON_FILE"; then
    echo "✅ 包含函數定義"
else
    echo "❌ 缺少函數定義"
    exit 1
fi

if grep -q 'if __name__ == "__main__"' "$PYTHON_FILE"; then
    echo "✅ 包含 main guard"
else
    echo "❌ 缺少 main guard"
    exit 1
fi

# 測試 Python 文件執行
echo ""
echo "=== 測試 Python 文件執行 ==="

# 執行Python文件（使用絕對路徑）
PYTHON_OUTPUT=$(python "$PYTHON_FILE" 2>&1)
PYTHON_EXIT_CODE=$?

if [ $PYTHON_EXIT_CODE -eq 0 ]; then
    echo "✅ Python 文件執行成功"
    echo "📤 輸出: $PYTHON_OUTPUT"
    
    # 驗證輸出包含 Hello World (不區分大小寫)
    if echo "$PYTHON_OUTPUT" | grep -qi "hello" && echo "$PYTHON_OUTPUT" | grep -qi "world"; then
        echo "✅ 輸出包含預期的 'Hello, World!'"
        
        # 額外驗證：檢查常見的Hello World格式
        if echo "$PYTHON_OUTPUT" | grep -qi "Hello, World!"; then
            echo "✅ 輸出格式標準: 'Hello, World!'"
        elif echo "$PYTHON_OUTPUT" | grep -qi "Hello, world!"; then
            echo "✅ 輸出格式正確: 'Hello, world!'"
        else
            echo "✅ 輸出包含Hello和World: $PYTHON_OUTPUT"
        fi
    else
        echo "❌ 輸出不包含預期的 'Hello, World!': $PYTHON_OUTPUT"
        echo "這表示auto-approve生成的代碼有問題"
        exit 1
    fi
else
    echo "❌ Python 文件執行失敗"
    echo "錯誤輸出: $PYTHON_OUTPUT"
    exit 1
fi

# 不需要返回原目錄，因為沒有切換目錄

# 成功總結
echo ""
echo "🎉 Auto-Approve 端到端測試成功完成!"
echo "=================================================="
echo "✅ 真實 AutoGen agents 成功創建了可工作的 Python 應用程式"
echo "✅ Auto-approve 模式端到端工作，無需手動干預"
echo "✅ 生成的 Python 文件可執行並產生預期輸出"
echo "✅ 所有工作流程階段 (Requirements → Design → Tasks → Implementation) 完成"
echo ""
echo "📁 生成文件位置: artifacts/$WORK_DIR"
echo "🐍 Python 文件: artifacts/$PYTHON_FILE"
echo "📤 執行輸出: $PYTHON_OUTPUT"
echo ""
echo "💡 所有產出物已保存在 artifacts/ 目錄中"
echo "結束時間: $(date)"

# 返回到項目根目錄
cd ..

exit 0