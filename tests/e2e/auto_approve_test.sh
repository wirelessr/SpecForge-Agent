#!/bin/bash

# Auto-Approve End-to-End Test
# Tests that auto-approve mode creates a working Hello World Python file using real agents

set -e  # Exit on any error

# Set test environment variables
export INTEGRATION_TESTING=true
export TESTING=true
export LOG_LEVEL=ERROR  # 只顯示錯誤信息

echo "=== Auto-Approve End-to-End Test ==="
echo "Start time: $(date)"
echo ""

# Test configuration
TEST_TASK="Create a simple Python file that prints 'Hello, World!' when executed. The file should have a main function and proper Python structure."
TEST_WORKSPACE="auto_approve_test_workspace"

# Cleanup function
cleanup_test_files() {
    echo "Cleaning up test files..."
    # Remove test workspace and temporary files
    rm -rf "$TEST_WORKSPACE"
    rm -f auto_approve_*.txt
    echo "Test file cleanup completed"
}

# Set trap to ensure cleanup runs on exit
trap cleanup_test_files EXIT

echo "=== Environment Check ==="

# Ensure running in correct project root directory
if [ ! -f "pyproject.toml" ] || [ ! -d "autogen_framework" ]; then
    echo "Error: Please run this test script from the project root directory"
    exit 1
fi

# Check if .env.integration file exists
if [ ! -f ".env.integration" ]; then
    echo "Error: .env.integration file does not exist. Please create it and configure real LLM settings"
    exit 1
fi

# Ensure autogen-framework command is available
if ! command -v autogen-framework &> /dev/null; then
    echo "Error: autogen-framework command not found. Please install the project: pip install -e ."
    exit 1
fi

echo "✅ Environment check passed"
echo ""

# Reset session
echo "=== Reset Session ==="
autogen-framework --reset-session
echo "✅ Session reset completed"
echo ""

# Submit auto-approve request
echo "=== Submit Auto-Approve Request ==="
echo "Request: $TEST_TASK"
echo ""

# Use auto-approve mode with workspace
echo "⚡ Processing request with auto-approve mode..."
autogen-framework --workspace "$TEST_WORKSPACE" --request "$TEST_TASK" --auto-approve > auto_approve_output.txt 2>&1

if [ $? -ne 0 ]; then
    echo "❌ Auto-approve request failed"
    echo "Error output:"
    cat auto_approve_output.txt
    exit 1
fi

echo "✅ Auto-approve request processing completed"
echo ""

# Check status
echo "=== Check Framework Status ==="
autogen-framework --workspace "$TEST_WORKSPACE" --status
echo ""

# Find generated workspace
echo "=== Find Generated Workspace ==="

# Check if workspace directory exists
if [ ! -d "$TEST_WORKSPACE" ]; then
    echo "❌ Workspace directory not found: $TEST_WORKSPACE"
    echo ""
    echo "Current directory contents:"
    ls -la
    echo ""
    echo "Auto-approve output:"
    if [ -f "auto_approve_output.txt" ]; then
        cat auto_approve_output.txt
    fi
    exit 1
fi

echo "📂 Found workspace directory: $TEST_WORKSPACE"

# Find the project directory within workspace
echo ""
echo "=== Find Project Directory ==="

# Look for project directories (exclude system directories)
PROJECT_DIRS=()
for dir in "$TEST_WORKSPACE"/*; do
    if [ -d "$dir" ]; then
        dirname=$(basename "$dir")
        # Skip system directories
        if [[ "$dirname" != "logs" && "$dirname" != "memory" && "$dirname" != "." && "$dirname" != ".." ]]; then
            PROJECT_DIRS+=("$dir")
        fi
    fi
done

if [ ${#PROJECT_DIRS[@]} -eq 0 ]; then
    echo "❌ No project directory found"
    echo "Workspace contents:"
    ls -la "$TEST_WORKSPACE"
    exit 1
fi

# Use the first project directory found
PROJECT_DIR="${PROJECT_DIRS[0]}"
echo "📂 Found project directory: $PROJECT_DIR"

# Check required files exist
echo ""
echo "=== Check Generated Files ==="

REQUIRED_FILES=("requirements.md" "design.md" "tasks.md")
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$PROJECT_DIR/$file" ]; then
        echo "✅ $file exists"
    else
        echo "❌ $file does not exist"
        echo "Project directory contents:"
        ls -la "$PROJECT_DIR"
        exit 1
    fi
done

# Find Python files
echo ""
echo "=== Find Python Files ==="
PYTHON_FILES=($(find "$PROJECT_DIR" -name "*.py" -type f))

if [ ${#PYTHON_FILES[@]} -eq 0 ]; then
    echo "❌ No Python files found"
    echo "$TEST_WORKSPACE directory contents:"
    ls -la "$TEST_WORKSPACE"
    exit 1
fi

# Use the first Python file found
PYTHON_FILE="${PYTHON_FILES[0]}"
echo "🐍 Found Python file: $PYTHON_FILE"

# Display Python file content
echo ""
echo "=== Python File Content ==="
echo "----------------------------------------"
cat "$PYTHON_FILE"
echo "----------------------------------------"

# Check if file is empty
if [ ! -s "$PYTHON_FILE" ]; then
    echo "❌ Python file is empty!"
    echo "This indicates the auto-approve process failed to generate code."
    echo ""
    echo "Checking auto-approve output for errors:"
    if [ -f "auto_approve_output.txt" ]; then
        echo "--- Auto-approve output ---"
        cat auto_approve_output.txt
        echo "--- End auto-approve output ---"
    fi
    echo ""
    echo "Checking project directory for other files:"
    ls -la "$PROJECT_DIR"
    exit 1
fi

# Validate Python file content
echo ""
echo "=== Validate Python File Content ==="

if grep -qi "hello" "$PYTHON_FILE" && grep -qi "world" "$PYTHON_FILE"; then
    echo "✅ Contains 'Hello' and 'World'"
else
    echo "❌ Missing 'Hello' or 'World'"
    echo "File content:"
    cat "$PYTHON_FILE"
    echo ""
    echo "This indicates the auto-approve process generated incomplete code."
    exit 1
fi

if grep -q "def main\|def " "$PYTHON_FILE"; then
    echo "✅ Contains function definition"
else
    echo "❌ Missing function definition"
    exit 1
fi

if grep -q 'if __name__ == "__main__"' "$PYTHON_FILE"; then
    echo "✅ Contains main guard"
else
    echo "❌ Missing main guard"
    exit 1
fi

# Test Python file execution
echo ""
echo "=== Test Python File Execution ==="

# Execute Python file
PYTHON_OUTPUT=$(python "$PYTHON_FILE" 2>&1)
PYTHON_EXIT_CODE=$?

if [ $PYTHON_EXIT_CODE -eq 0 ]; then
    echo "✅ Python file executed successfully"
    echo "📤 Output: $PYTHON_OUTPUT"
    
    # Validate output contains Hello World (case insensitive)
    if echo "$PYTHON_OUTPUT" | grep -qi "hello" && echo "$PYTHON_OUTPUT" | grep -qi "world"; then
        echo "✅ Output contains expected 'Hello, World!'"
        
        # Additional validation: check common Hello World formats
        if echo "$PYTHON_OUTPUT" | grep -qi "Hello, World!"; then
            echo "✅ Standard output format: 'Hello, World!'"
        elif echo "$PYTHON_OUTPUT" | grep -qi "Hello, world!"; then
            echo "✅ Correct output format: 'Hello, world!'"
        else
            echo "✅ Output contains Hello and World: $PYTHON_OUTPUT"
        fi
    else
        echo "❌ Output does not contain expected 'Hello, World!': $PYTHON_OUTPUT"
        echo "This indicates the auto-approve generated code has issues"
        exit 1
    fi
else
    echo "❌ Python file execution failed"
    echo "Error output: $PYTHON_OUTPUT"
    exit 1
fi

# Success summary
echo ""
echo "🎉 Auto-Approve End-to-End Test Successfully Completed!"
echo "=================================================="
echo "✅ Real AutoGen agents successfully created a working Python application"
echo "✅ Auto-approve mode works end-to-end without manual intervention"
echo "✅ Generated Python file is executable and produces expected output"
echo "✅ All workflow stages (Requirements → Design → Tasks → Implementation) completed"
echo ""
echo "📁 Generated files location: $TEST_WORKSPACE"
echo "🐍 Python file: $PYTHON_FILE"
echo "📤 Execution output: $PYTHON_OUTPUT"
echo ""
echo "End time: $(date)"

exit 0