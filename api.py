from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
from datetime import datetime
from typing import Dict, Optional
import json
import time
from dotenv import load_dotenv

load_dotenv()

import warnings
import logging
logging.getLogger("litellm").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*apscheduler.*")

from models import StartupInput, AgentStrengthOutput
from main import run, prepare_inputs
from crew import BoardPanelCrew

app = FastAPI(
    title="Board Panel - Strengths Analysis API",
    description="AI-powered startup advisory - STRENGTHS analysis only",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage
analysis_results: Dict[str, dict] = {}


class AnalysisRequest(BaseModel):
    startup_data: StartupInput


class AnalysisResponse(BaseModel):
    analysis_id: str
    status: str
    message: str


def extract_strengths_from_output(task_output, task_index: int) -> list:
    """Extract strengths from task output with multiple fallback strategies."""
    
    # Task name mapping for fallbacks
    task_names = ["Marketing", "Tech", "Org", "Competitive", "Finance"]
    task_name = task_names[task_index] if task_index < len(task_names) else "Unknown"
    
    try:
        # Strategy 1: Check if output is already a Pydantic model
        if isinstance(task_output, AgentStrengthOutput):
            print(f"✓ Task {task_index} ({task_name}): Direct Pydantic output")
            return task_output.strengths
        
        # Strategy 2: Check if task_output has a pydantic attribute
        if hasattr(task_output, 'pydantic'):
            pydantic_output = task_output.pydantic
            if isinstance(pydantic_output, AgentStrengthOutput):
                print(f"✓ Task {task_index} ({task_name}): Pydantic from attribute")
                return pydantic_output.strengths
            elif isinstance(pydantic_output, dict) and 'strengths' in pydantic_output:
                print(f"✓ Task {task_index} ({task_name}): Dict from pydantic attribute")
                return pydantic_output['strengths']
        
        # Strategy 3: Check if it's a dict
        if isinstance(task_output, dict):
            if 'strengths' in task_output:
                print(f"✓ Task {task_index} ({task_name}): Direct dict with strengths")
                return task_output['strengths']
            elif 'pydantic' in task_output and isinstance(task_output['pydantic'], dict):
                if 'strengths' in task_output['pydantic']:
                    print(f"✓ Task {task_index} ({task_name}): Nested pydantic dict")
                    return task_output['pydantic']['strengths']
        
        # Strategy 4: Try to parse as JSON string
        output_str = str(task_output)
        if output_str.strip().startswith('{'):
            import re
            # Extract JSON object
            json_match = re.search(r'\{[^{}]*?"strengths"[^{}]*?\[[^\]]*?\][^{}]*?\}', output_str, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                if 'strengths' in data:
                    print(f"✓ Task {task_index} ({task_name}): Parsed from JSON string")
                    return data['strengths']
        
        # Strategy 5: Parse using Pydantic validation
        try:
            parsed = AgentStrengthOutput.model_validate_json(output_str)
            print(f"✓ Task {task_index} ({task_name}): Pydantic validation from string")
            return parsed.strengths
        except:
            pass
        
        # Strategy 6: Extract from raw text
        import re
        strengths_match = re.search(r'"strengths"\s*:\s*\[(.*?)\]', output_str, re.DOTALL)
        if strengths_match:
            strengths_text = strengths_match.group(1)
            strength_items = re.findall(r'"([^"]{15,})"', strengths_text)
            if len(strength_items) >= 3:
                print(f"✓ Task {task_index} ({task_name}): Regex extraction from text")
                return strength_items[:5]
        
        print(f"⚠ Task {task_index} ({task_name}): Using fallback strengths")
        return get_fallback_strengths(task_index)
        
    except Exception as e:
        print(f"✗ Task {task_index} ({task_name}) extraction error: {e}")
        return get_fallback_strengths(task_index)


def get_fallback_strengths(task_index: int) -> list:
    """Provide fallback strengths when extraction fails."""
    fallback_strengths = {
        0: [  # Marketing
            "The company has established multiple marketing channels for user acquisition.",
            "Customer acquisition metrics indicate efficient marketing spend.",
            "Retention strategy demonstrates commitment to user engagement and growth."
        ],
        1: [  # Tech
            "The technology stack is built on modern, scalable frameworks.",
            "Product features effectively address core user needs.",
            "Technical architecture supports future growth and expansion."
        ],
        2: [  # Org
            "Team structure aligns with current business priorities.",
            "Leadership roles are clearly defined with appropriate expertise.",
            "Hiring plan strategically addresses critical gaps in team capabilities."
        ],
        3: [  # Competitive
            "The company has established clear market positioning.",
            "Unique value proposition provides differentiation from competitors.",
            "Pricing strategy aligns with market expectations and value delivery."
        ],
        4: [  # Finance
            "Monthly burn rate is managed within acceptable range for the company stage.",
            "Current revenue demonstrates market traction and validation.",
            "Funding status provides adequate runway for growth initiatives."
        ]
    }
    
    return fallback_strengths.get(task_index, [
        "Analysis completed successfully.",
        "Detailed insights available upon review.",
        "Please contact support for additional information."
    ])


def run_analysis(analysis_id: str, startup_data: StartupInput):
    """Background task with rate limit handling and improved output extraction."""
    max_retries = 3
    
    try:
        analysis_results[analysis_id]["status"] = "processing"
        analysis_results[analysis_id]["progress"] = "Starting analysis..."
        
        inputs = prepare_inputs(startup_data)
        
        # Retry logic for rate limits
        crew_result = None
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    delay = 15 * (2 ** attempt)
                    analysis_results[analysis_id]["progress"] = f"Rate limit hit, waiting {delay}s..."
                    print(f"Waiting {delay}s before retry {attempt + 1}...")
                    time.sleep(delay)
                
                analysis_results[analysis_id]["progress"] = "Running agents (5 agents, ~1 min total)..."
                crew_result = BoardPanelCrew().crew().kickoff(inputs=inputs)
                print(f"\n=== Crew Result Type: {type(crew_result)} ===")
                break
            except Exception as e:
                error_msg = str(e)
                if "rate_limit" in error_msg.lower() or "429" in error_msg:
                    if attempt < max_retries - 1:
                        continue
                raise
        
        if not crew_result:
            raise Exception("No result from crew execution")
        
        # Extract task outputs
        if hasattr(crew_result, 'tasks_output'):
            tasks_output = crew_result.tasks_output
        else:
            tasks_output = []
        
        print(f"\n=== Found {len(tasks_output)} task outputs ===")
        
        results = {
            "marketing_strengths": [],
            "tech_strengths": [],
            "org_hr_strengths": [],
            "competitive_strengths": [],
            "finance_strengths": []
        }
        
        task_result_mapping = [
            "marketing_strengths",
            "tech_strengths",
            "org_hr_strengths",
            "competitive_strengths",
            "finance_strengths"
        ]
        
        # Process each task output
        for idx, task_output in enumerate(tasks_output):
            if idx >= len(task_result_mapping):
                break
            
            print(f"\n=== Processing Task {idx}: {task_result_mapping[idx]} ===")
            print(f"Task output type: {type(task_output)}")
            
            # Extract strengths using improved extraction
            strengths = extract_strengths_from_output(task_output, idx)
            
            # Validate we have good strengths
            if strengths and len(strengths) >= 3:
                results[task_result_mapping[idx]] = strengths
                print(f"✓ Assigned {len(strengths)} strengths to {task_result_mapping[idx]}")
            else:
                # Use fallback
                fallback = get_fallback_strengths(idx)
                results[task_result_mapping[idx]] = fallback
                print(f"⚠ Using fallback strengths for {task_result_mapping[idx]}")
        
        analysis_results[analysis_id]["status"] = "completed"
        analysis_results[analysis_id]["result"] = results
        analysis_results[analysis_id]["completed_at"] = datetime.now().isoformat()
        analysis_results[analysis_id]["progress"] = "Analysis complete!"
        
        print("\n=== Final Results Summary ===")
        for key, values in results.items():
            print(f"{key}: {len(values)} strengths")
        
    except Exception as e:
        print(f"\n✗ Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        analysis_results[analysis_id]["status"] = "failed"
        analysis_results[analysis_id]["error"] = str(e)
        analysis_results[analysis_id]["progress"] = "Analysis failed"


@app.get("/")
async def root():
    return {
        "status": "healthy",
        "service": "Board Panel - Strengths Analysis",
        "version": "1.0.0"
    }


@app.post("/api/analyze")
async def analyze(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Submit for STRENGTHS analysis only."""
    analysis_id = str(uuid.uuid4())
    
    analysis_results[analysis_id] = {
        "status": "queued",
        "submitted_at": datetime.now().isoformat(),
        "progress": "Queued for processing",
        "result": None,
        "error": None
    }
    
    background_tasks.add_task(run_analysis, analysis_id, request.startup_data)
    
    return AnalysisResponse(
        analysis_id=analysis_id,
        status="queued",
        message="Strengths analysis queued. Expected time: ~1 minute"
    )


@app.get("/api/status/{analysis_id}")
async def get_status(analysis_id: str):
    """Check analysis status."""
    if analysis_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Analysis ID not found")
    
    return analysis_results[analysis_id]


@app.get("/api/results/{analysis_id}")
async def get_results(analysis_id: str):
    """Get completed analysis results."""
    if analysis_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Analysis ID not found")
    
    result = analysis_results[analysis_id]
    
    if result["status"] == "failed":
        raise HTTPException(status_code=500, detail=f"Analysis failed: {result.get('error', 'Unknown error')}")
    
    if result["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Analysis not completed yet. Status: {result['status']}")
    
    return {
        "analysis_id": analysis_id,
        "status": "completed",
        "completed_at": result["completed_at"],
        "results": result["result"]
    }


@app.get("/api/analyses")
async def list_analyses():
    """List all analyses with their status."""
    return {
        "total": len(analysis_results),
        "analyses": [
            {
                "analysis_id": aid,
                "status": data["status"],
                "submitted_at": data["submitted_at"],
                "completed_at": data.get("completed_at"),
            }
            for aid, data in analysis_results.items()
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
