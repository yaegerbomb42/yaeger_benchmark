from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import subprocess
import os
import json
import tempfile
import shutil
from typing import Dict, Any, Optional
import asyncio
import psutil
import time
from pathlib import Path

app = FastAPI(
    title="Yaeger Benchmark API",
    description="API for submitting and evaluating code solutions",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SubmissionRequest(BaseModel):
    code: str
    language: str = "python"
    timeout: int = 300  # 5 minutes default timeout

class SubmissionResponse(BaseModel):
    task_id: str
    score: float
    output: str
    details: Dict[str, Any]
    submission_id: str

VALID_TASKS = [f"task{i}" for i in range(1, 11)]
BASE_DIR = Path(__file__).parent.parent
TASKS_DIR = BASE_DIR / "tasks"
RESULTS_DIR = BASE_DIR / "results"

# Ensure results directory exists
RESULTS_DIR.mkdir(exist_ok=True)

def get_leaderboard_path():
    return RESULTS_DIR / "leaderboard.json"

def load_leaderboard():
    leaderboard_path = get_leaderboard_path()
    if leaderboard_path.exists():
        with open(leaderboard_path) as f:
            return json.load(f)
    return []

def save_leaderboard(leaderboard):
    leaderboard_path = get_leaderboard_path()
    with open(leaderboard_path, 'w') as f:
        json.dump(leaderboard, f, indent=2)

def update_leaderboard(model_name: str, task_id: str, score: float):
    leaderboard = load_leaderboard()
    
    # Find existing entry or create new one
    entry = None
    for item in leaderboard:
        if item["model"] == model_name:
            entry = item
            break
    
    if entry is None:
        entry = {"model": model_name}
        leaderboard.append(entry)
    
    entry[task_id] = score
    
    # Calculate total score
    total = sum(entry.get(f"task{i}", 0) for i in range(1, 11))
    entry["total"] = total
    
    # Sort by total score
    leaderboard.sort(key=lambda x: x.get("total", 0), reverse=True)
    
    save_leaderboard(leaderboard)

async def run_verification(task_id: str, submission_dir: Path) -> Dict[str, Any]:
    """Run the verification script for a task and return results."""
    verify_script = TASKS_DIR / task_id / "verify.sh"
    
    if not verify_script.exists():
        raise HTTPException(status_code=404, detail=f"Verification script not found for {task_id}")
    
    # Make script executable
    os.chmod(verify_script, 0o755)
    
    # Run verification script
    start_time = time.time()
    process = await asyncio.create_subprocess_exec(
        str(verify_script),
        cwd=submission_dir,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)
        execution_time = time.time() - start_time
        
        output = stdout.decode() + stderr.decode()
        
        # Parse output for metrics
        details = {
            "execution_time": execution_time,
            "tests_passed": 0,
            "runtime": 0.0,
            "vulnerabilities": 0,
            "memory_usage": 0.0
        }
        
        # Extract metrics from output
        lines = output.split('\n')
        for line in lines:
            if line.startswith("Tests:"):
                try:
                    details["tests_passed"] = int(line.split()[1].split('/')[0])
                except:
                    pass
            elif line.startswith("Runtime:"):
                try:
                    details["runtime"] = float(line.split()[1])
                except:
                    pass
            elif line.startswith("Vulns:"):
                try:
                    details["vulnerabilities"] = int(line.split()[1])
                except:
                    pass
            elif line.startswith("Memory:"):
                try:
                    details["memory_usage"] = float(line.split()[1])
                except:
                    pass
        
        # Calculate score
        score = 0.0
        if "Score:" in output:
            try:
                score = float(output.split("Score: ")[1].split()[0])
            except:
                pass
        
        return {
            "score": score,
            "output": output,
            "details": details
        }
        
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        raise HTTPException(status_code=408, detail="Verification timed out")

@app.post("/submit/{task_id}", response_model=SubmissionResponse)
async def submit_solution(task_id: str, request: SubmissionRequest):
    """Submit a solution for evaluation."""
    if task_id not in VALID_TASKS:
        raise HTTPException(status_code=400, detail="Invalid task ID")
    
    task_dir = TASKS_DIR / task_id
    if not task_dir.exists():
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    # Create temporary directory for submission
    with tempfile.TemporaryDirectory() as temp_dir:
        submission_dir = Path(temp_dir)
        
        # Copy task files to temp directory
        shutil.copytree(task_dir, submission_dir / task_id)
        
        # Write submission code
        if request.language == "python":
            submission_file = submission_dir / task_id / "submission.py"
        elif request.language == "javascript":
            submission_file = submission_dir / task_id / "submission.js"
        elif request.language == "go":
            submission_file = submission_dir / task_id / "submission.go"
        else:
            raise HTTPException(status_code=400, detail="Unsupported language")
        
        with open(submission_file, 'w') as f:
            f.write(request.code)
        
        # Run verification
        result = await run_verification(task_id, submission_dir / task_id)
        
        # Generate submission ID
        submission_id = f"{task_id}_{int(time.time())}"
        
        return SubmissionResponse(
            task_id=task_id,
            score=result["score"],
            output=result["output"],
            details=result["details"],
            submission_id=submission_id
        )

@app.get("/leaderboard")
async def get_leaderboard():
    """Get the current leaderboard."""
    return load_leaderboard()

@app.get("/task/{task_id}")
async def get_task_details(task_id: str):
    """Get details about a specific task."""
    if task_id not in VALID_TASKS:
        raise HTTPException(status_code=400, detail="Invalid task ID")
    
    task_dir = TASKS_DIR / task_id
    if not task_dir.exists():
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    issue_file = task_dir / "issue.md"
    if not issue_file.exists():
        raise HTTPException(status_code=404, detail=f"Issue description not found for {task_id}")
    
    with open(issue_file) as f:
        description = f.read()
    
    return {
        "task_id": task_id,
        "description": description,
        "files": [f.name for f in task_dir.iterdir() if f.is_file()]
    }

@app.get("/tasks")
async def list_tasks():
    """List all available tasks."""
    tasks = []
    for task_id in VALID_TASKS:
        task_dir = TASKS_DIR / task_id
        if task_dir.exists():
            issue_file = task_dir / "issue.md"
            title = task_id
            if issue_file.exists():
                with open(issue_file) as f:
                    content = f.read()
                    # Extract title from first line
                    first_line = content.split('\n')[0]
                    if first_line.startswith('#'):
                        title = first_line.strip('# ')
            
            tasks.append({
                "task_id": task_id,
                "title": title,
                "available": True
            })
        else:
            tasks.append({
                "task_id": task_id,
                "title": f"Task {task_id[4:]}",
                "available": False
            })
    
    return tasks

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
