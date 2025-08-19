# AutoGen Framework Usage Examples

This directory contains various usage examples and case studies for the AutoGen Multi-Agent Framework.

## 📁 Example Directory Structure

```
examples/
├── README.md                    # This file
├── basic-usage/                 # Basic usage examples
│   ├── simple-api.md           # Simple API creation
│   ├── data-analysis.md        # Data analysis tool
│   └── cli-tool.md             # Command-line tool
├── advanced-projects/           # Advanced project examples
│   ├── microservices.md        # Microservices architecture
│   ├── web-application.md      # Complete web application
│   └── ml-pipeline.md          # Machine learning pipeline
├── integration-examples/        # Integration examples
│   ├── ci-cd-integration.md    # CI/CD integration
│   ├── docker-deployment.md    # Docker deployment
│   └── cloud-deployment.md     # Cloud deployment
└── troubleshooting/            # Troubleshooting cases
    ├── common-issues.md        # Common issues
    ├── debugging-guide.md      # Debugging guide
    └── performance-tips.md     # Performance optimization
```

## 🚀 Quick Start Examples

### 1. Create a Simple REST API

This is the most basic usage example, showing how to create a simple REST API.

```bash
# Create a project directory
mkdir simple-api-demo
cd simple-api-demo

# Submit a request
autogen-framework --workspace . --request "
Create a FastAPI REST API for a simple task management system with:
- Task CRUD operations (Create, Read, Update, Delete)
- Task model with id, title, description, completed status
- In-memory storage (list/dict)
- Basic error handling
- API documentation with Swagger UI
- Simple validation for required fields
"

# Approve each phase
autogen-framework --workspace . --approve requirements
autogen-framework --workspace . --approve design
autogen-framework --workspace . --approve tasks

# Execute the implementation
autogen-framework --workspace . --execute-tasks
```

**Expected Result**:
- `main.py`: FastAPI application main file
- `models.py`: Data model definitions
- `requirements.txt`: Dependency list
- Complete CRUD API endpoints

### 2. Data Analysis Tool

Shows how to create a data processing and analysis tool.

```bash
mkdir data-analyzer
cd data-analyzer

autogen-framework --workspace . --request "
Build a Python data analysis tool that:
- Reads CSV files from a command line argument
- Performs basic statistical analysis (mean, median, std)
- Generates histograms and scatter plots
- Exports a summary report to a text file
- Handles missing data gracefully
- Includes command-line help and usage examples
"
```

### 3. Command-Line Tool

Create a practical command-line tool.

```bash
mkdir file-organizer
cd file-organizer

autogen-framework --workspace . --request "
Create a command-line file organizer tool that:
- Scans a directory for files
- Organizes files by extension into subdirectories
- Supports a dry-run mode to preview changes
- Provides progress feedback
- Handles duplicate files safely
- Includes undo functionality
- Uses Click for the command-line interface
"
```

## 🏗️ Advanced Project Examples

### Microservices Architecture

```bash
mkdir microservices-demo
cd microservices-demo

autogen-framework --workspace . --request "
Design and implement a microservices architecture for an e-commerce platform:

Services:
- User Service: Authentication, user profiles, JWT tokens
- Product Service: Product catalog, inventory management
- Order Service: Order processing, order history
- Payment Service: Payment processing simulation
- API Gateway: Request routing, rate limiting

Technical Requirements:
- FastAPI for all services
- PostgreSQL databases for each service
- Redis for caching and session storage
- Docker containers for each service
- Docker Compose for local development
- Service-to-service communication via HTTP
- Centralized logging configuration
- Health check endpoints for all services
- API documentation for each service
"
```

### Complete Web Application

```bash
mkdir blog-platform
cd blog-platform

autogen-framework --workspace . --request "
Create a complete blog platform with:

Backend (FastAPI):
- User authentication and authorization
- Blog post CRUD operations
- Comment system
- Tag and category management
- File upload for images
- RESTful API design
- SQLAlchemy ORM with PostgreSQL
- Alembic for database migrations

Frontend (React):
- User registration and login
- Blog post creation and editing (rich text editor)
- Comment functionality
- Responsive design
- Search and filtering
- Pagination
- Image upload interface

Additional Features:
- JWT token authentication
- Input validation and sanitization
- Error handling and user feedback
- Unit tests for backend
- Component tests for frontend
- Docker deployment configuration
"
```

## 🔧 Integration Examples

### CI/CD Integration

Shows how to integrate the framework into a CI/CD pipeline.

```yaml
# .github/workflows/autogen-development.yml
name: AutoGen Development Workflow

on:
  workflow_dispatch:
    inputs:
      project_name:
        description: 'Project name'
        required: true
      requirements:
        description: 'Project requirements'
        required: true

jobs:
  generate-code:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install AutoGen Framework
        run: |
          pip install uv
          uv sync
      
      - name: Generate Project
        run: |
          mkdir ${{ github.event.inputs.project_name }}
          cd ${{ github.event.inputs.project_name }}
          
          autogen-framework \
            --workspace . \
            --llm-base-url ${{ secrets.LLM_BASE_URL }} \
            --llm-model ${{ secrets.LLM_MODEL }} \
            --llm-api-key ${{ secrets.LLM_API_KEY }} \
            --request "${{ github.event.inputs.requirements }}"
          
          # Auto-approve for CI (be careful in production)
          autogen-framework --workspace . --approve requirements
          autogen-framework --workspace . --approve design
          autogen-framework --workspace . --approve tasks
          autogen-framework --workspace . --execute-tasks
      
      - name: Create Pull Request
        uses: peter-evans/create-pull-request@v5
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
          commit-message: "feat: generate ${{ github.event.inputs.project_name }}"
          title: "AutoGen: ${{ github.event.inputs.project_name }}"
          body: |
            Auto-generated project using AutoGen Framework
            
            Requirements: ${{ github.event.inputs.requirements }}
          branch: autogen/${{ github.event.inputs.project_name }}
```

### Docker Deployment

```dockerfile
# Dockerfile for AutoGen Framework
FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy framework
COPY . .
RUN uv sync

# Create entrypoint script
RUN echo '#!/bin/bash\n\
cd /workspace\n\
autogen-framework --workspace . "$@"' > /usr/local/bin/autogen-entrypoint.sh
RUN chmod +x /usr/local/bin/autogen-entrypoint.sh

VOLUME ["/workspace"]
ENTRYPOINT ["/usr/local/bin/autogen-entrypoint.sh"]
```

Usage:
```bash
# Build the image
docker build -t autogen-framework .

# Run the container
docker run -v $(pwd):/workspace \
  -e LLM_BASE_URL=http://your-llm-endpoint/v1 \
  -e LLM_MODEL=your-model \
  -e LLM_API_KEY=your-key \
  autogen-framework \
  --request "Create a simple web API"
```

## 📊 Performance Benchmarks

### Testing performance on different project scales

```bash
# Small project (< 5 files)
time autogen-framework --workspace ./small-project --request "Create a simple calculator CLI"

# Medium project (5-15 files)
time autogen-framework --workspace ./medium-project --request "Create a REST API with a database"

# Large project (15+ files)
time autogen-framework --workspace ./large-project --request "Create a microservices architecture"
```

### Performance Optimization Suggestions

1.  **Concurrent Processing**: For large projects, consider executing independent tasks in parallel.
2.  **Caching Strategy**: Utilize the memory system to avoid regenerating similar content.
3.  **Incremental Updates**: Use the patch strategy to minimize file modifications.

## 🎯 Best Practice Cases

### 1. Requirements Engineering Best Practices

```bash
# Good requirement description example
autogen-framework --workspace ./best-practice-example --request "
Project: E-commerce Product Recommendation System

Business Context:
- Online retail platform with 10k+ products
- Need to improve user engagement and sales conversion
- Target: 20% increase in click-through rate

Functional Requirements:
1. User Behavior Tracking
   - Track user browsing history
   - Record purchase patterns
   - Monitor search queries

2. Recommendation Engine
   - Collaborative filtering algorithm
   - Content-based filtering
   - Hybrid approach combining both methods
   - Real-time recommendation updates

3. API Interface
   - RESTful API for recommendation requests
   - Response time < 100ms for 95% of requests
   - Support for batch recommendations
   - A/B testing capability

Technical Requirements:
- Python with FastAPI framework
- PostgreSQL for user data storage
- Redis for caching recommendations
- Scikit-learn for ML algorithms
- Docker containerization
- Comprehensive unit tests (>90% coverage)
- API documentation with OpenAPI/Swagger
- Monitoring and logging integration

Performance Requirements:
- Handle 1000 concurrent users
- Process 10k recommendations per minute
- 99.9% uptime availability

Security Requirements:
- User data privacy compliance
- API rate limiting
- Input validation and sanitization
- Secure authentication for admin endpoints
"
```

### 2. Iterative Development Process

```bash
# First iteration: Basic functionality
autogen-framework --workspace ./iterative-project --request "
Phase 1: Create a basic user authentication system with:
- User registration and login
- JWT token management
- Password hashing
- Basic user profile management
"

# Review and execute the first phase
autogen-framework --workspace ./iterative-project --approve requirements
autogen-framework --workspace ./iterative-project --approve design
autogen-framework --workspace ./iterative-project --approve tasks
autogen-framework --workspace ./iterative-project --execute-tasks

# Second iteration: Add features
autogen-framework --workspace ./iterative-project --request "
Phase 2: Extend the existing authentication system with:
- Email verification
- Password reset functionality
- User role management (admin, user, moderator)
- OAuth integration (Google, GitHub)
- Session management
- Audit logging
"
```

### 3. Team Collaboration Workflow

```bash
# Team lead creates the project structure
autogen-framework --workspace ./team-project --request "
Create a project structure and architecture for a team collaboration tool:
- Define clear module boundaries
- Set up the development environment
- Create coding standards and guidelines
- Implement a basic project skeleton
- Set up the testing framework
- Configure the CI/CD pipeline structure
"

# Different team members are responsible for different modules
# Frontend developer
autogen-framework --workspace ./team-project --request "
Implement frontend components for the collaboration tool:
- User interface components
- State management
- API integration
- Responsive design
- Component testing
"

# Backend developer
autogen-framework --workspace ./team-project --request "
Implement backend services for the collaboration tool:
- API endpoints
- Database models
- Business logic
- Authentication middleware
- Unit tests
"
```

## 🔍 Debugging and Troubleshooting Examples

### Common Issue Resolution

```bash
# Issue 1: LLM response timeout
autogen-framework --workspace . --verbose --log-file debug.log --request "Your request"
# Check the logs for timeout errors, adjust network configuration or LLM endpoint

# Issue 2: Generated code has syntax errors
autogen-framework --workspace . --execute-task "Fix syntax errors in generated code"
# Or continue after manual fixing

# Issue 3: Task execution failed
autogen-framework --workspace . --status
# Check the failed task, re-execute the specific task using --execute-task

# Issue 4: Inconsistent workflow state
autogen-framework --workspace . --reset-session
# Reset the session state and restart the workflow
```

### Performance Tuning Example

```bash
# Enable detailed monitoring
autogen-framework \
  --workspace ./performance-test \
  --verbose \
  --log-file performance.log \
  --request "Create a high-performance web service"

# Analyze performance logs
grep -E "(duration|time|performance)" performance.log

# Optimization suggestions
# 1. Use more specific requirement descriptions to reduce LLM calls
# 2. Utilize the memory system to avoid repetitive work
# 3. Execute large projects in phases
```

## 📚 Learning Path

### Beginner Path

1.  **Basic Concepts** (1-2 hours)
    - Read README.md
    - Understand the multi-agent architecture
    - Learn basic commands

2.  **Simple Practice** (2-3 hours)
    - Create a simple API
    - Experience the complete workflow
    - Learn revision and feedback

3.  **Advanced Features** (3-4 hours)
    - Use the patch strategy
    - Understand the memory system
    - Master troubleshooting

### Advanced User Path

1.  **Architecture Deep Dive** (2-3 hours)
    - Study the agent coordination mechanism
    - Understand workflow state management
    - Learn extension development

2.  **Complex Projects** (4-6 hours)
    - Microservices architecture practice
    - Large web application development
    - CI/CD integration

3.  **Custom Development** (6+ hours)
    - Develop custom agents
    - Integrate external tools
    - Performance optimization and monitoring

## 🤝 Community Contributions

### Submitting New Examples

If you have a good use case, feel free to submit it to this example collection:

1.  Create a new example file.
2.  Include complete usage instructions.
3.  Provide expected results and screenshots.
4.  Add troubleshooting tips.
5.  Submit a Pull Request.

### Example Template

```markdown
# Example Title

## Overview
Briefly describe the purpose and applicable scenarios of this example.

## Prerequisites
- List the required environment and dependencies.
- Explain the technical background requirements.

## Step-by-Step Instructions
1.  Detailed execution steps.
2.  Include complete commands.
3.  Explain the expected result of each step.

## Expected Result
- Describe the final generated files.
- Provide key code snippets.
- Include running screenshots.

## Troubleshooting
- Common issues and solutions.
- Debugging tips.
- Performance optimization suggestions.

## Extension Suggestions
- How to further develop based on this example.
- Related advanced features.
- Suggestions for integrating other tools.
```

---

**Maintainer**: AutoGen Framework Team  
**Last Updated**: 2025-08-18
**Version**: 1.0.0
