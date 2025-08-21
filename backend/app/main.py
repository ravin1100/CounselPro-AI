from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.params import Depends
from app.db.database import create_tables, get_sync_db
from app.routes.counselor_route import router as counselor_router
from app.routes.session_route import router as session_router
from app.routes.catalog_route import router as catalog_router
from app.routes.session_analysis_route import router as session_analysis_router
from app.routes.raw_transcript_route import router as raw_transcript_router
from app.routes.cloudinary_route import router as cloudinary_test_router
from contextlib import asynccontextmanager
import uvicorn
from sqlalchemy.orm import Session

from app.exceptions.global_exception_handler import register_exception_handlers
from fastapi.middleware.cors import CORSMiddleware
from app.config.log_config import get_logger
from app.service.email_service import (
    send_simple_email_template,
)
from dotenv import load_dotenv

load_dotenv()

# Initialize logger
logger = get_logger("CounselPro")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_tables()
    for route in app.routes:
        print("🚀 Loaded route:", route.path, route.methods)
    yield


app = FastAPI(
    title="CounselPro AI - API - Version 1",
    description="AI-Powered Counselor Excellence System",
    version="1.0.0",
    lifespan=lifespan,
    debug=True,
)

# Register exception handlers FIRST
register_exception_handlers(app)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React development server
        "https://counselpro-ai.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    import time, uuid

    request_id = str(uuid.uuid4())
    start_time = time.time()
    logger.info(
        f"[{request_id}] 📥 Incoming request: {request.method} {request.url.path}"
    )

    try:
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000
        logger.info(
            f"[{request_id}] 📤 Response {response.status_code} completed_in={process_time:.2f}ms"
        )
        return response
    except Exception as e:
        logger.exception(f"[{request_id}] ❌ Error handling request: {str(e)}")
        raise


app.include_router(counselor_router)
app.include_router(session_router)
app.include_router(catalog_router)
app.include_router(session_analysis_router)
app.include_router(raw_transcript_router)
app.include_router(cloudinary_test_router)


@app.get("/", tags=["Health Check"])
async def root():
    return {"message": "AI-Powered Counselor Excellence System"}


# @app.post("/email/test", tags=["Test Email"])
# async def test_email(email: str):
#     """
#     Test endpoint to verify email functionality
#     """
#     # success = await test_email_sending(email)
#     success = await test_modern_email_template(email)
#     if success:
#         return {"message": f"Test email sent successfully to {email}"}
#     else:
#         raise HTTPException(status_code=500, detail="Failed to send test email")


@app.get("/send-email/{session_uid}")
def send_email_route(session_uid: str, db: Session = Depends(get_sync_db)):
    try:
        # Run sync email sender in a thread to avoid blocking event loop
        import asyncio

        success = send_simple_email_template(db, session_uid)

        if success:
            return {"success": True, "message": f"Email sent for session {session_uid}"}
        else:
            return {"success": False, "message": "Email sending failed"}
    except Exception as e:
        return {"success": False, "message": str(e)}


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app", host="127.0.0.1", port=8000, reload=True, log_level="debug"
    )
