# Example: Creating a Simple REST API

This example demonstrates how to use the AutoGen Framework to create a simple FastAPI REST API.

## 📋 Project Overview

We will create a task management API with the following features:
- CRUD operations for tasks
- In-memory storage
- API documentation
- Basic validation

## 🚀 Execution Steps

### 1. Create a Project Directory

```bash
mkdir simple-task-api
cd simple-task-api
```

### 2. Submit a Development Request

```bash
autogen-framework --workspace . --request "
Create a FastAPI REST API for a simple task management system with the following requirements:

Functional Requirements:
1. Task Model:
   - id: unique integer identifier
   - title: string, required, max 100 characters
   - description: optional string, max 500 characters
   - completed: boolean, default False
   - created_at: timestamp
   - updated_at: timestamp

2. API Endpoints:
   - GET /tasks - List all tasks with optional filtering by completed status
   - GET /tasks/{id} - Get a specific task by ID
   - POST /tasks - Create a new task
   - PUT /tasks/{id} - Update an existing task
   - DELETE /tasks/{id} - Delete a task

3. Features:
   - In-memory storage using a Python list/dict
   - Input validation with Pydantic models
   - Proper HTTP status codes
   - Error handling with meaningful messages
   - API documentation with Swagger UI
   - CORS support for frontend integration

Technical Requirements:
- Use the FastAPI framework
- Python 3.11+ compatibility
- Type hints throughout the code
- Proper project structure
- A requirements.txt file
- Basic logging configuration
- A health check endpoint at /health

Quality Requirements:
- Clean, readable code with comments
- Proper error handling
- Input validation
- RESTful API design principles
- OpenAPI documentation
"
```

### 3. Review the Generated Requirements

```bash
# Check the current status
autogen-framework --workspace . --status

# View the generated requirements document
cat requirements.md
```

**Expected Requirements Document Content**:
- Detailed user stories
- Acceptance criteria in EARS format
- Technical requirement specifications
- Performance and quality requirements

### 4. Approve (or Revise) the Requirements

```bash
# If the requirements are satisfactory, approve them directly
autogen-framework --workspace . --approve requirements

# If revisions are needed, provide feedback
autogen-framework --workspace . --revise "requirements:Add rate limiting for API endpoints"
```

### 5. Review the Technical Design

```bash
# View the generated design document
cat design.md
```

**Expected Design Document Content**:
- System architecture diagram
- API endpoint design
- Data model definitions
- Error handling strategy
- Testing strategy

### 6. Approve the Design

```bash
autogen-framework --workspace . --approve design
```

### 7. Review the Task List

```bash
# View the generated task list
cat tasks.md
```

**Expected Task List**:
- Project structure setup
- Dependency installation
- Data model implementation
- API endpoint implementation
- Error handling
- Document generation
- Test writing

### 8. Approve and Execute Tasks

```bash
# Approve the task list
autogen-framework --workspace . --approve tasks

# Execute all tasks
autogen-framework --workspace . --execute-tasks
```

## 📁 Expected Project Structure

After execution is complete, your project directory should contain:

```
simple-task-api/
├── requirements.md          # Requirements document
├── design.md               # Design document
├── tasks.md                # Task list
├── main.py                 # FastAPI application main file
├── models.py               # Pydantic data models
├── database.py             # In-memory database implementation
├── requirements.txt        # Python dependencies
├── README.md              # Project description
└── tests/                 # Test files
    └── test_api.py
```

## 🔍 Generated Code Example

### main.py
```python
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import logging
from datetime import datetime

from models import Task, TaskCreate, TaskUpdate
from database import TaskDatabase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Simple Task Management API",
    description="A simple REST API for managing tasks",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the database
db = TaskDatabase()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now()}

@app.get("/tasks", response_model=List[Task])
async def get_tasks(completed: Optional[bool] = Query(None)):
    """Get all tasks, optionally filtering by completion status."""
    return db.get_tasks(completed=completed)

@app.get("/tasks/{task_id}", response_model=Task)
async def get_task(task_id: int):
    """Get a specific task by ID."""
    task = db.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@app.post("/tasks", response_model=Task, status_code=201)
async def create_task(task: TaskCreate):
    """Create a new task."""
    return db.create_task(task)

@app.put("/tasks/{task_id}", response_model=Task)
async def update_task(task_id: int, task_update: TaskUpdate):
    """Update an existing task."""
    task = db.update_task(task_id, task_update)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@app.delete("/tasks/{task_id}")
async def delete_task(task_id: int):
    """Delete a task."""
    success = db.delete_task(task_id)
    if not success:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"message": "Task deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### models.py
```python
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class TaskBase(BaseModel):
    title: str = Field(..., max_length=100, description="Task title")
    description: Optional[str] = Field(None, max_length=500, description="Task description")
    completed: bool = Field(False, description="Task completion status")

class TaskCreate(TaskBase):
    pass

class TaskUpdate(BaseModel):
    title: Optional[str] = Field(None, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    completed: Optional[bool] = None

class Task(TaskBase):
    id: int = Field(..., description="Unique task identifier")
    created_at: datetime = Field(..., description="Task creation timestamp")
    updated_at: datetime = Field(..., description="Task last update timestamp")
    
    class Config:
        from_attributes = True
```

## 🧪 Test the API

### 1. Start the Server

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
python main.py
```

### 2. Test the Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Create a task
curl -X POST http://localhost:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{"title": "Learn FastAPI", "description": "Study FastAPI documentation"}'

# Get all tasks
curl http://localhost:8000/tasks

# Get a specific task
curl http://localhost:8000/tasks/1

# Update a task
curl -X PUT http://localhost:8000/tasks/1 \
  -H "Content-Type: application/json" \
  -d '{"completed": true}'

# Delete a task
curl -X DELETE http://localhost:8000/tasks/1
```

### 3. View the API Documentation

Open a browser and visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🎯 Learning Points

### 1. Advantages of the AutoGen Framework

- **Complete Workflow**: End-to-end automation from requirements to implementation.
- **High-Quality Code**: Generated code follows best practices.
- **Detailed Documentation**: Automatically generates requirements, design, and task documents.
- **Controllability**: Each phase can be reviewed and revised.

### 2. Characteristics of the Generated Code

- **Type Safe**: Uses Python type hints.
- **Input Validation**: Pydantic models ensure data integrity.
- **Error Handling**: Appropriate HTTP status codes and error messages.
- **Documented**: OpenAPI specification is automatically generated.

### 3. Scalability

The generated code provides a good foundation for further development:
- Easy to add new endpoints.
- Can be easily integrated with a database.
- Supports the addition of authentication and authorization.
- Convenient for adding more business logic.

## 🔧 Customization and Extension

### 1. Add Database Support

```bash
# Revise the task to add database support
autogen-framework --workspace . --revise "tasks:Replace in-memory storage with SQLite database using SQLAlchemy"
```

### 2. Add Authentication

```bash
# Add JWT authentication
autogen-framework --workspace . --revise "design:Add JWT token-based authentication for all endpoints"
```

### 3. Add More Features

```bash
# Extend functionality
autogen-framework --workspace . --request "
Extend the existing task API with:
- Task categories and tags
- Due dates and reminders
- Task priority levels
- Search and filtering capabilities
- Bulk operations
- Export functionality
"
```

## 🐛 Common Issues

### Issue 1: Dependency Installation Failed

```bash
# Solution: Use a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# Or venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Issue 2: Port in Use

```bash
# Check port usage
lsof -i :8000

# Use a different port
python main.py --port 8001
```

### Issue 3: CORS Error

If the frontend cannot access the API, check the CORS configuration:
```python
# Adjust CORS settings in main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Specify the frontend domain
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)
```

## 📈 Performance Optimization Suggestions

1.  **Add Caching**: Use Redis to cache frequently queried data.
2.  **Database Optimization**: Add indexes and query optimization.
3.  **Asynchronous Processing**: Use background tasks for time-consuming operations.
4.  **Rate Limiting**: Add API rate limiting for protection.
5.  **Monitoring**: Integrate APM tools to monitor performance.

## 🎉 Summary

This example demonstrates the powerful features of the AutoGen Framework:

- ✅ **Rapid Development**: From requirements to a runnable API in just a few minutes.
- ✅ **High-Quality Code**: Follows best practices and coding standards.
- ✅ **Complete Documentation**: Automatically generated documents help with understanding and maintenance.
- ✅ **Scalability**: Provides a good foundation for further development.
- ✅ **Controllability**: Each phase can be reviewed and adjusted.

Through this simple example, you can see how the AutoGen Framework can significantly accelerate the software development process while maintaining code quality and maintainability.

---

**Example Difficulty**: Beginner  
**Estimated Time**: 15-30 minutes  
**Technology Stack**: FastAPI, Pydantic, Python 3.11+  
**Last Updated**: 2025-08-18
