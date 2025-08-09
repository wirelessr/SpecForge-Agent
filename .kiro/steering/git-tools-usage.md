---
inclusion: always
---

# Git 工具使用指導原則

## 核心原則
**永遠使用 MCP git 工具進行 git 操作，而不是 executeBash 命令**

## 正確的工具用法

### 1. 檢查狀態
```
✅ 正確: mcp_docker_mcp_toolkit_git_status
❌ 錯誤: executeBash("git status")
```

### 2. 查看提交歷史
```
✅ 正確: mcp_docker_mcp_toolkit_git_log
❌ 錯誤: executeBash("git log")
```

### 3. 查看提交詳情
```
✅ 正確: mcp_docker_mcp_toolkit_git_show
❌ 錯誤: executeBash("git show")
```

### 4. 添加文件到暫存區
```
✅ 正確: mcp_docker_mcp_toolkit_git_add
❌ 錯誤: executeBash("git add")
```

### 5. 提交變更
```
✅ 正確: mcp_docker_mcp_toolkit_git_commit
❌ 錯誤: executeBash("git commit")
```

### 6. 查看差異
```
✅ 正確: mcp_docker_mcp_toolkit_git_diff
❌ 錯誤: executeBash("git diff")
```

### 7. 創建分支
```
✅ 正確: mcp_docker_mcp_toolkit_git_create_branch
❌ 錯誤: executeBash("git branch")
```

### 8. 切換分支
```
✅ 正確: mcp_docker_mcp_toolkit_git_checkout
❌ 錯誤: executeBash("git checkout")
```

### 9. 重置變更
```
✅ 正確: mcp_docker_mcp_toolkit_git_reset
❌ 錯誤: executeBash("git reset")
```

## 重要參數說明

### repo_path 參數
- **必須使用絕對路徑**: `/Users/ctw/Trial/spec-agent`
- **不要使用相對路徑**: `.` 或 `./`
- **獲取當前路徑**: 先用 `executeBash("pwd")` 獲取，然後使用完整路徑

### 常見錯誤和解決方案

#### 錯誤 1: 路徑問題
```
❌ 錯誤: repo_path="."
✅ 正確: repo_path="/Users/ctw/Trial/spec-agent"
```

#### 錯誤 2: 使用 bash 命令
```
❌ 錯誤: executeBash("git status")
✅ 正確: mcp_docker_mcp_toolkit_git_status(repo_path="/full/path")
```

## 標準工作流程

### 1. 檢查當前狀態
```python
mcp_docker_mcp_toolkit_git_status(repo_path="/full/path")
```

### 2. 添加文件
```python
mcp_docker_mcp_toolkit_git_add(
    repo_path="/full/path",
    files=["file1.py", "file2.py"]
)
```

### 3. 提交變更
```python
mcp_docker_mcp_toolkit_git_commit(
    repo_path="/full/path",
    message="feat: descriptive commit message"
)
```

### 4. 查看結果
```python
mcp_docker_mcp_toolkit_git_log(
    repo_path="/full/path",
    max_count=1
)
```

## 提交消息格式

### 標準格式
```
type(scope): description

- Detailed explanation
- What was changed
- Why it was changed
- Requirements addressed

Addresses: requirement references
```

### 類型說明
- `feat`: 新功能
- `fix`: 錯誤修復
- `refactor`: 重構
- `test`: 測試相關
- `docs`: 文檔更新
- `chore`: 維護任務

## 實際使用範例

### 完整的 commit 流程
```python
# 1. 檢查狀態
status = mcp_docker_mcp_toolkit_git_status(
    repo_path="/Users/ctw/Trial/spec-agent"
)

# 2. 添加文件
mcp_docker_mcp_toolkit_git_add(
    repo_path="/Users/ctw/Trial/spec-agent",
    files=["autogen_framework/session_manager.py", "tests/unit/test_session_manager.py"]
)

# 3. 提交
mcp_docker_mcp_toolkit_git_commit(
    repo_path="/Users/ctw/Trial/spec-agent",
    message="feat: Extract session management into dedicated SessionManager component

- Create new SessionManager class in autogen_framework/session_manager.py
- Extract session management methods from MainController
- Integrate SessionManager into MainController with delegation pattern
- Add comprehensive unit tests for SessionManager (14 test cases)

Addresses requirements 1.1-1.4 from framework-architecture-refactoring spec."
)

# 4. 驗證結果
mcp_docker_mcp_toolkit_git_log(
    repo_path="/Users/ctw/Trial/spec-agent",
    max_count=1
)
```

## 故障排除

### 如果工具返回錯誤
1. 檢查 repo_path 是否為絕對路徑
2. 確認當前目錄是 git 倉庫
3. 檢查文件路徑是否正確
4. 確認有適當的 git 權限

### 獲取當前工作目錄
```python
# 先獲取當前路徑
result = executeBash("pwd")
current_path = result.strip()

# 然後使用完整路徑
mcp_docker_mcp_toolkit_git_status(repo_path=current_path)
```

## 🚨 重要：不要提交 artifacts

### 什麼是 artifacts？
- `artifacts/` 目錄下的所有文件
- 測試報告、質量分析、日誌文件
- 臨時生成的文件和輸出結果
- 開發過程中的中間產物

### 為什麼不能提交？
- **已在 .gitignore 中**: artifacts/ 目錄被明確排除
- **文件過大**: 報告和日誌會讓倉庫變得臃腫
- **頻繁變動**: 每次測試都會生成新的文件
- **本地特定**: 這些文件只對本地開發有意義

### 提交前檢查
```bash
# 檢查是否有 artifacts 被意外追蹤
git ls-files | grep artifacts

# 如果有，立即移除
git rm -r artifacts/

# 確認 .gitignore 包含 artifacts/
grep "artifacts/" .gitignore
```

### 正確的工作流程
```python
# 1. 檢查狀態（確保沒有 artifacts）
status = mcp_docker_mcp_toolkit_git_status(repo_path="/full/path")

# 2. 只添加源代碼和測試文件
mcp_docker_mcp_toolkit_git_add(
    repo_path="/full/path",
    files=["src/module.py", "tests/test_module.py"]  # 不包含 artifacts/
)

# 3. 提交時確認沒有 artifacts
mcp_docker_mcp_toolkit_git_commit(
    repo_path="/full/path",
    message="feat: implement new feature"
)
```

## 記住
- **永遠使用 MCP git 工具**
- **永遠使用絕對路徑**
- **詳細的提交消息**
- **先檢查狀態，再執行操作**
- **🚨 絕不提交 artifacts/ 目錄**