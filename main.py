import asyncio
import os
import time
import json
import hashlib
import shutil
from datetime import datetime
from collections import defaultdict
from functools import wraps
from typing import Dict, List, Any, Callable, Optional

# --- Third-Party Imports ---
import aiohttp
import uvicorn
import yaml
import structlog
from fastapi import (
    FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, Float
from sqlalchemy.orm import declarative_base, sessionmaker

# --- Setup logging, DB Models, Request/Response Models ---
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True
)
logger = structlog.get_logger()
Base = declarative_base()

class ChatHistory(Base):
    __tablename__ = 'chat_history'
    id = Column(Integer, primary_key=True)
    user_id = Column(String(100), nullable=False, index=True)
    message = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    model_used = Column(String(100), nullable=False)
    cost = Column(Float, default=0.0)
    response_time = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)

class ChatRequest(BaseModel):
    message: str
    user_id: str = "anonymous"
    task_type: str = "auto"
    user_tier: str = "free"

class ChatResponse(BaseModel):
    success: bool
    response: str
    model: str
    provider: str
    cost: float
    response_time: float
    cached: bool = False
    reasoning: str = ""

class FunctionCallRequest(BaseModel):
    function_name: str
    arguments: Dict[str, Any]
    user_id: str = "anonymous"

class FunctionCallResponse(BaseModel):
    success: bool
    result: Any
    function_name: str
    execution_time: float

class DevelopmentTask(BaseModel):
    task_description: str

# --- Core Service Classes ---
class ConfigManager:
    def __init__(self, config_file: str = "config.yaml"): 
        self.config_file = config_file
        self.config = {}
        self.load_config()
    
    def load_config(self):
        try:
            with open(self.config_file, 'r') as f: 
                self.config = yaml.safe_load(f)
        except Exception as e: 
            logger.warning("Config file not found, using defaults", error=str(e))
            # Provide default config if file doesn't exist
            self.config = self._get_default_config()
        
        # Override with environment variables if they exist
        if os.environ.get("OPENROUTER_API_KEY"):
            self.config["openrouter_api_key"] = os.environ.get("OPENROUTER_API_KEY")
        
        # Log configuration status (without sensitive data)
        logger.info("Configuration loaded", 
                   has_openrouter_key=bool(self.config.get("openrouter_api_key")),
                   database_url=self.config.get("database_url", "not_set"))
    
    def _get_default_config(self):
        """Provide default configuration if config.yaml is missing"""
        return {
            "openrouter_api_key": os.environ.get("OPENROUTER_API_KEY", ""),
            "classifier_model": "openai/gpt-4o-mini",
            "database_url": "sqlite:///ai_portal.db",
            "valid_task_types": ["simple_qa", "code_generation", "creative_writing"],
            "model_tiers": {
                "economy": ["openai/gpt-4o-mini"],
                "standard": ["openai/gpt-4o"],
                "premium": ["anthropic/claude-3.5-sonnet"]
            },
            "task_tier_map": {
                "simple_qa": "economy",
                "code_generation": "standard",
                "creative_writing": "premium"
            },
            "model_providers": {
                "openai/gpt-4o-mini": "OpenAI",
                "openai/gpt-4o": "OpenAI",
                "anthropic/claude-3.5-sonnet": "Anthropic"
            },
            "copyshark": {
                "base_url": "http://localhost:3000",
                "api_token": "",
                "endpoints": {
                    "generate_copy": "/api/generate-copy",
                    "get_frameworks": "/api/frameworks",
                    "get_niches": "/api/niches",
                    "get_user": "/api/user/me"
                }
            },
            "copyshark_functions": [
                {
                    "name": "generateAdCopy",
                    "description": "Generate advertising copy for a product",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "productName": {"type": "string"},
                            "audience": {"type": "string"},
                            "niche": {"type": "string"},
                            "framework": {"type": "string"},
                            "tone": {"type": "string"}
                        },
                        "required": ["productName", "audience"]
                    }
                }
            ],
            "agent_urls": {},
            "api": {
                "host": "0.0.0.0",
                "port": 8000
            }
        }
    
    def get(self, key: str, default=None): 
        return self.config.get(key, default)

class CopySharkService:
    def __init__(self, config: Dict):
        self.config = config.get('copyshark', {})
        self.base_url = self.config.get('base_url', 'http://localhost:3000')
        self.api_token = self.config.get('api_token')
        self.session = None
        
    async def initialize(self):
        self.session = aiohttp.ClientSession()
        
    async def cleanup(self):
        if self.session:
            await self.session.close()
    
    def _get_headers(self):
        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        return headers
    
    async def _api_call(self, endpoint: str, data: Dict = None, method: str = "POST"):
        url = f"{self.base_url}{endpoint}"
        headers = self._get_headers()
        
        try:
            if method == "GET":
                async with self.session.get(url, headers=headers) as response:
                    response.raise_for_status()
                    return await response.json()
            else:
                async with self.session.post(url, headers=headers, json=data) as response:
                    response.raise_for_status()
                    return await response.json()
        except Exception as e:
            logger.error(f"CopyShark API call failed: {e}")
            raise HTTPException(status_code=500, detail=f"CopyShark service error: {str(e)}")
    
    async def generate_ad_copy(self, product_name: str, audience: str, niche: str = None, framework: str = None, tone: str = "professional"):
        endpoint = self.config.get('endpoints', {}).get('generate_copy', '/api/generate-copy')
        data = {
            "productName": product_name,
            "audience": audience,
            "niche": niche or "general",
            "framework": framework or "AIDA", 
            "tone": tone
        }
        return await self._api_call(endpoint, data)
    
    async def get_frameworks(self):
        endpoint = self.config.get('endpoints', {}).get('get_frameworks', '/api/frameworks')
        return await self._api_call(endpoint, method="GET")
    
    async def get_niches(self):
        endpoint = self.config.get('endpoints', {}).get('get_niches', '/api/niches')
        return await self._api_call(endpoint, method="GET")
    
    async def get_user_usage(self):
        endpoint = self.config.get('endpoints', {}).get('get_user', '/api/user/me')
        return await self._api_call(endpoint, method="GET")

class OpenSourceAIService:
    def __init__(self, config: Dict):
        self.openrouter_key = config.get('openrouter_api_key')
        self.session = None
        
    async def initialize(self): 
        self.session = aiohttp.ClientSession()
        
    async def cleanup(self):
        if self.session: 
            await self.session.close()
            
    async def _api_call(self, messages: List[Dict], model: str):
        headers = {
            "Authorization": f"Bearer {self.openrouter_key}", 
            "Content-Type": "application/json"
        }
        payload = {
            "model": model, 
            "messages": messages, 
            "temperature": 0.7, 
            "max_tokens": 2048
        }
        async with self.session.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload) as response:
            response.raise_for_status()
            return await response.json()
            
    async def chat_completion(self, messages: List[Dict], model: str):
        data = await self._api_call(messages, model)
        return {
            'response': data['choices'][0]['message']['content'], 
            'cost': data.get('usage', {}).get('total_tokens', 0) * 0.000001
        }
        
    async def detect_task_type(self, user_prompt: str, classifier_model: str, valid_types: List[str]):
        categories = ", ".join(f'"{t}"' for t in valid_types)
        system_prompt = f"Classify the user's message into one of these categories: {categories}. Respond with ONLY the category name in quotes."
        data = await self._api_call([
            {"role": "system", "content": system_prompt}, 
            {"role": "user", "content": user_prompt}
        ], classifier_model)
        detected_type = data['choices'][0]['message']['content'].strip().replace('"', '')
        return detected_type if detected_type in valid_types else "simple_qa"
    
    async def determine_function_calls(self, user_prompt: str, available_functions: List[Dict]):
        """Determine if the user's request requires function calls"""
        function_descriptions = []
        for func in available_functions:
            function_descriptions.append(f"- {func['name']}: {func['description']}")
        
        system_prompt = f"""You are a function call router. Analyze the user's request and determine if it requires calling any of these functions:

{chr(10).join(function_descriptions)}

If the request requires function calls, respond with a JSON array of function calls in this format:
[{{"function_name": "functionName", "arguments": {{"param1": "value1"}}}}]

If no function calls are needed, respond with: []

Only respond with the JSON array, nothing else."""

        data = await self._api_call([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ], "openai/gpt-4o-mini")
        
        try:
            function_calls = json.loads(data['choices'][0]['message']['content'].strip())
            return function_calls if isinstance(function_calls, list) else []
        except:
            return []

class SimpleIntelligentRouter:
    def __init__(self, config: Dict):
        self.model_tiers = config.get('model_tiers', {})
        self.task_tier_map = config.get('task_tier_map', {})
        self.model_providers = config.get('model_providers', {})
        self.round_robin_counter = defaultdict(int)
        
    def route_simple(self, task_type: str, user_tier: str):
        tier_name = self.task_tier_map.get(task_type, 'economy')
        if user_tier == 'pro' and tier_name == 'economy': 
            tier_name = 'standard'
        models = self.model_tiers.get(tier_name, self.model_tiers['economy'])
        model = models[self.round_robin_counter[tier_name] % len(models)]
        self.round_robin_counter[tier_name] += 1
        return {
            'model': model, 
            'provider': self.model_providers.get(model, 'unknown'), 
            'reasoning': f"Detected '{task_type}', routed to {tier_name.upper()}"
        }

def handle_errors(func: Callable) -> Callable:
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try: 
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}", exc_info=True)
            return JSONResponse(
                status_code=500, 
                content={
                    "success": False, 
                    "response": f"An internal error occurred: {type(e).__name__}", 
                    "detail": str(e)
                }
            )
    return wrapper

# --- UNIFIED AI PORTAL APPLICATION ---
class UnifiedAIPortal:
    def __init__(self, config_file: str = "config.yaml"):
        logger.info("Initializing Unified AI Portal...")
        self.config_manager = ConfigManager(config_file)
        self.config = self.config_manager.config
        self.app = FastAPI(title="Command Center V3 - Unified", version="3.2.0")
        self.ai_service = OpenSourceAIService(self.config)
        self.copyshark_service = CopySharkService(self.config)
        self.router = SimpleIntelligentRouter(self.config)
        
        db_url = self.config.get('database_url', 'sqlite:///ai_portal.db')
        self.db_engine = create_engine(db_url)
        Base.metadata.create_all(self.db_engine)
        self.DbSession = sessionmaker(bind=self.db_engine)
        
        self.setup_app()
        logger.info("Initialization complete.")

    def setup_app(self):
        if not os.path.exists("uploads"): 
            os.makedirs("uploads")
        
        # Check if frontend directory exists
        if os.path.exists("frontend"):
            self.app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")
        
        self.app.add_middleware(
            CORSMiddleware, 
            allow_origins=["*"], 
            allow_credentials=True, 
            allow_methods=["*"], 
            allow_headers=["*"]
        )
        
        @self.app.on_event("startup")
        async def startup(): 
            await self.ai_service.initialize()
            await self.copyshark_service.initialize()
            
        @self.app.on_event("shutdown")
        async def shutdown(): 
            await self.ai_service.cleanup()
            await self.copyshark_service.cleanup()
            
        self.setup_routes()

    def setup_routes(self):
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "message": "AI Portal is running"}

        @self.app.get("/", response_class=HTMLResponse, include_in_schema=False)
        async def root():
            try:
                with open("frontend/index.html", "r", encoding="utf-8") as f: 
                    return HTMLResponse(content=f.read())
            except FileNotFoundError:
                return HTMLResponse(content="<h1>AI Portal Backend Running</h1><p>Frontend files not found. <a href='/health'>Check Health</a></p>")

        @self.app.post("/chat", response_model=ChatResponse)
        @handle_errors
        async def chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
            start_time = time.time()
            
            # First check if this requires function calls
            available_functions = self.config.get('copyshark_functions', [])
            function_calls = await self.ai_service.determine_function_calls(request.message, available_functions)
            
            if function_calls:
                # Execute function calls and integrate results
                function_results = []
                for func_call in function_calls:
                    try:
                        result = await self.execute_function_call(func_call['function_name'], func_call.get('arguments', {}))
                        function_results.append(f"Function {func_call['function_name']} returned: {json.dumps(result)}")
                    except Exception as e:
                        function_results.append(f"Function {func_call['function_name']} failed: {str(e)}")
                
                # Create enhanced prompt with function results
                enhanced_message = f"{request.message}\n\nFunction Results:\n" + "\n".join(function_results)
                
                # Determine task type for the enhanced message
                final_task_type = request.task_type
                if final_task_type == "auto":
                    final_task_type = await self.ai_service.detect_task_type(
                        enhanced_message, 
                        self.config.get('classifier_model'), 
                        self.config.get('valid_task_types', [])
                    )
                
                routing_decision = self.router.route_simple(final_task_type, request.user_tier)
                result = await self.ai_service.chat_completion([
                    {"role": "user", "content": enhanced_message}
                ], routing_decision['model'])
                
                response_time = time.time() - start_time
                chat_response = ChatResponse(
                    success=True, 
                    response=result['response'], 
                    model=routing_decision['model'], 
                    provider=routing_decision['provider'], 
                    cost=result['cost'], 
                    response_time=response_time, 
                    reasoning=f"{routing_decision['reasoning']} + Function Calls"
                )
            else:
                # Standard chat without function calls
                final_task_type = request.task_type
                if final_task_type == "auto":
                    final_task_type = await self.ai_service.detect_task_type(
                        request.message, 
                        self.config.get('classifier_model'), 
                        self.config.get('valid_task_types', [])
                    )
                
                routing_decision = self.router.route_simple(final_task_type, request.user_tier)
                result = await self.ai_service.chat_completion([
                    {"role": "user", "content": request.message}
                ], routing_decision['model'])
                
                response_time = time.time() - start_time
                chat_response = ChatResponse(
                    success=True, 
                    response=result['response'], 
                    model=routing_decision['model'], 
                    provider=routing_decision['provider'], 
                    cost=result['cost'], 
                    response_time=response_time, 
                    reasoning=routing_decision['reasoning']
                )
            
            background_tasks.add_task(self._save_chat_history, request, chat_response)
            return chat_response

        # CopyShark Function Call Endpoints
        @self.app.post("/functions/call", response_model=FunctionCallResponse)
        @handle_errors
        async def call_function(request: FunctionCallRequest):
            start_time = time.time()
            result = await self.execute_function_call(request.function_name, request.arguments)
            execution_time = time.time() - start_time
            
            return FunctionCallResponse(
                success=True,
                result=result,
                function_name=request.function_name,
                execution_time=execution_time
            )

        @self.app.get("/functions/available")
        async def get_available_functions():
            return {"functions": self.config.get('copyshark_functions', [])}

        # CopyShark specific endpoints
        @self.app.post("/copyshark/generate", tags=["CopyShark"])
        @handle_errors
        async def generate_ad_copy(
            product_name: str = Form(...),
            audience: str = Form(...),
            niche: str = Form(None),
            framework: str = Form(None),
            tone: str = Form("professional")
        ):
            result = await self.copyshark_service.generate_ad_copy(product_name, audience, niche, framework, tone)
            return JSONResponse(content=result)

        @self.app.get("/copyshark/frameworks", tags=["CopyShark"])
        @handle_errors
        async def get_frameworks():
            result = await self.copyshark_service.get_frameworks()
            return JSONResponse(content=result)

        @self.app.get("/copyshark/niches", tags=["CopyShark"])
        @handle_errors
        async def get_niches():
            result = await self.copyshark_service.get_niches()
            return JSONResponse(content=result)

        # Legacy endpoints
        agent_urls = self.config.get('agent_urls', {})
        
        @self.app.post("/delegate/marketing-copy", tags=["Agent Orchestration"])
        @handle_errors
        async def delegate_marketing_copy(topic: str = Form(...)):
            marketing_agent_url = agent_urls.get("marketing")
            if not marketing_agent_url: 
                raise HTTPException(status_code=500, detail="Marketing agent URL not configured.")
            async with aiohttp.ClientSession() as session:
                async with session.post(marketing_agent_url, json={"topic": topic}, timeout=20.0) as response:
                    response.raise_for_status()
                    return JSONResponse(content=await response.json())

        @self.app.post("/develop-feature", tags=["Agent Orchestration"])
        @handle_errors
        async def develop_feature(task: DevelopmentTask):
            return JSONResponse(content={
                "status": "Development Cycle Simulated", 
                "task": task.task_description, 
                "outcome": "SUCCESS"
            })

        @self.app.post("/upload-file", tags=["File Handling"])
        @handle_errors
        async def upload_file(file: UploadFile = File(...)):
            upload_dir = "uploads"
            file_path = os.path.join(upload_dir, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            return JSONResponse(
                status_code=200, 
                content={
                    "success": True, 
                    "filename": file.filename, 
                    "detail": "File uploaded successfully."
                }
            )

    async def execute_function_call(self, function_name: str, arguments: Dict[str, Any]):
        """Execute a function call and return the result"""
        try:
            if function_name == "generateAdCopy":
                return await self.copyshark_service.generate_ad_copy(
                    product_name=arguments.get("productName", ""),
                    audience=arguments.get("audience", ""),
                    niche=arguments.get("niche"),
                    framework=arguments.get("framework"),
                    tone=arguments.get("tone", "professional")
                )
            elif function_name == "getFrameworks":
                return await self.copyshark_service.get_frameworks()
            elif function_name == "getNiches":
                return await self.copyshark_service.get_niches()
            elif function_name == "getUserUsage":
                return await self.copyshark_service.get_user_usage()
            else:
                raise HTTPException(status_code=404, detail=f"Function {function_name} not found")
        except Exception as e:
            logger.error(f"Function execution failed: {e}")
            raise HTTPException(status_code=500, detail=f"Function execution failed: {str(e)}")

    def _save_chat_history(self, request: ChatRequest, response: ChatResponse):
        with self.DbSession() as session:
            try:
                history = ChatHistory(
                    user_id=request.user_id, 
                    message=request.message, 
                    response=response.response, 
                    model_used=response.model, 
                    cost=response.cost, 
                    response_time=response.response_time
                )
                session.add(history)
                session.commit()
            except Exception as e:
                logger.error("Failed to save chat history", error=str(e))
                session.rollback()

    def run(self):
        """Run the unified AI portal server"""
        try:
            # Render uses PORT environment variable
            port = int(os.environ.get('PORT', 8000))
            host = "0.0.0.0"  # Accept external connections
        
            logger.info(f"🚀 Starting Uvicorn server on {host}:{port}")
        
            # Start the server with additional config
            uvicorn.run(
                self.app, 
                host=host, 
                port=port,
                log_level="info",
                access_log=True
            )
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            raise

# Create the portal instance and expose the app for Gunicorn
portal = UnifiedAIPortal()
app = portal.app  # Expose for Gunicorn deployment

# Main execution block
if __name__ == "__main__":
    try:
        # Run the portal directly (for local development)
        portal.run()
    except Exception as e:
        logger.error(f"Failed to start portal: {e}")
        raise