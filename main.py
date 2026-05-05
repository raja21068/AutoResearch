"""
main.py — NexusAI entry point. App + all routes in one file.

Run: python main.py → http://localhost:8000
"""

import json, logging, os, sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# ── Windows asyncio fix ───────────────────────────────────
# Prevents 'RuntimeError: Event loop is closed' and SSL transport
# errors on Windows Python 3.10+ with the Proactor event loop.
if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from orchestrator import Orchestrator
from tools.file_reader import read_file, read_pdf_bytes, read_docx_bytes, read_dataset_bytes, summarize_files

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── App ──────────────────────────────────────────────────
app = FastAPI(title="NexusAI", version="1.0.0",
    description="Unified multi-agent research & coding framework.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"])

_orch = None
def get_orch():
    global _orch
    if _orch is None: _orch = Orchestrator()
    return _orch

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")

# ── Models ───────────────────────────────────────────────
class TaskRequest(BaseModel):
    task: str; context_files: list[str] = []; mode: str = "auto"
    repo_url: str = ""; repo_changes: str = ""
    papers_context: str = ""; dataset_context: str = ""; file_context: str = ""

# ── Health ───────────────────────────────────────────────
@app.get("/health")
async def health(): return {"status": "ok", "service": "NexusAI"}

# ── Upload ───────────────────────────────────────────────
@app.post("/api/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    results = []
    for f in files:
        content = await f.read()
        ext = Path(f.filename or "x").suffix.lower()
        if ext == ".pdf": r = read_pdf_bytes(content, f.filename)
        elif ext in (".docx",".doc"): r = read_docx_bytes(content, f.filename)
        elif ext in (".csv",".tsv",".json",".jsonl",".yaml",".yml"): r = read_dataset_bytes(content, f.filename, ext)
        else:
            tmp = os.path.join(UPLOAD_DIR, f.filename); open(tmp,"wb").write(content); r = read_file(tmp)
        results.append(r)
    papers = [r for r in results if r["type"] in ("pdf","docx")]
    datasets = [r for r in results if r["type"] == "dataset"]
    return {"files": results, "papers_context": summarize_files(papers),
            "dataset_context": summarize_files(datasets), "combined_context": summarize_files(results),
            "count": len(results), "categories": {"papers": len(papers), "datasets": len(datasets)}}

# ── Run ──────────────────────────────────────────────────
@app.post("/api/agent/run")
async def run_agent(req: TaskRequest):
    return await get_orch().run(req.task, req.context_files,
        papers_context=req.papers_context or req.file_context,
        repo_url=req.repo_url, repo_changes=req.repo_changes, dataset_context=req.dataset_context)

@app.post("/api/agent/stream")
async def run_stream(req: TaskRequest):
    async def gen():
        async for item in get_orch().run_streaming(req.task, req.context_files,
            papers_context=req.papers_context or req.file_context,
            repo_url=req.repo_url, repo_changes=req.repo_changes, dataset_context=req.dataset_context):
            evt, data = item.get("event","status"), item.get("data","")
            if evt == "token": data = data.replace("\n","↵")
            yield {"event": evt, "data": data}
    return EventSourceResponse(gen())

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_json()
            task = data.get("task","")
            if not task: await ws.send_json({"type":"error","data":"No task"}); continue
            async for item in get_orch().run_streaming(task, data.get("context_files",[]),
                papers_context=data.get("papers_context",""), repo_url=data.get("repo_url",""),
                repo_changes=data.get("repo_changes",""), dataset_context=data.get("dataset_context","")):
                await ws.send_json({"type":item["event"],"data":item["data"]})
            await ws.send_json({"type":"done","data":""})
    except WebSocketDisconnect: pass

# ── Outputs ──────────────────────────────────────────────
@app.get("/api/outputs")
async def list_outputs():
    out_dir = os.getenv("OUTPUT_DIR","./output")
    if not os.path.exists(out_dir): return {"runs":[]}
    runs = []
    for d in sorted(Path(out_dir).iterdir(), reverse=True):
        s = d / "summary.json"
        runs.append(json.loads(s.read_text()) if s.exists() else {"run_id":d.name})
    return {"runs":runs[:50]}

# ── Skills ───────────────────────────────────────────────
@app.get("/api/skills/agents")
async def list_skills():
    from skills.loader import get_agent_registry
    reg = get_agent_registry()
    return {"count": len(reg), "agents": [a.to_dict() for a in reg.agents.values()]}

@app.get("/api/skills/rules")
async def list_rules():
    from skills.engine import get_rule_engine
    return {"languages": get_rule_engine().available_languages()}

@app.get("/api/skills/rules/{lang}")
async def get_rules(lang: str):
    from skills.engine import get_rule_engine
    rules = get_rule_engine().get_rules(lang)
    return {"language": lang, "content": rules[:5000], "length": len(rules)}

# ── Domain APIs ──────────────────────────────────────────
class TopicReq(BaseModel):
    topic: str; papers: str = ""; results: str = ""

@app.post("/api/conception/ideate")
async def ideate(req: TopicReq):
    from agents.conception import ConceptionService
    return await ConceptionService().run_full(req.topic, req.papers)

@app.post("/api/experiment/run")
async def run_exp(req: TopicReq):
    from agents.experiment import ExperimentOrchestrator
    return await ExperimentOrchestrator().run_full(req.topic, req.papers)

@app.post("/api/paper/write")
async def write_paper(req: TopicReq):
    from agents.paper import PaperOrchestrator
    return await PaperOrchestrator().run_full(req.topic, req.papers, req.results)

@app.post("/api/gan/run")
async def run_gan(req: TopicReq):
    from agents.gan import GANHarness
    result = await GANHarness().run(req.topic, req.papers)
    return {"passed": result.passed, "score": result.final_score,
            "iterations": result.total_iterations, "elapsed": result.elapsed_sec}



@app.post("/api/paper/orchestra")
async def run_orchestra(req: TopicReq):
    """Run the full PaperOrchestra 5-step pipeline."""
    from agents.paper import PaperOrchestrator
    po = PaperOrchestrator()
    result = await po.run_full(req.topic, idea=req.papers, experiments=req.results)
    return {"topic": result["topic"], "outline": result["outline"][:1000],
            "refinement": result["refinement"], "paper_length": len(result.get("paper_tex",""))}

@app.get("/api/paper/skills")
async def list_paper_skills():
    """List available PaperOrchestra pipeline skills."""
    from skills.paper_pipeline_loader import get_paper_pipeline
    pp = get_paper_pipeline()
    return {"count": len(pp), "skills": [
        {"name": s.name, "step": s.step, "scripts": list(s.scripts.keys()),
         "references": list(s.references.keys())}
        for s in pp.skills.values()]}

# ── GUI ──────────────────────────────────────────────────
if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    idx = Path("static/index.html")
    return FileResponse(str(idx)) if idx.exists() else {"service":"NexusAI","docs":"/docs"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT","8000")), reload=False)
