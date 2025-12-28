from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import os
import tempfile
from typing import Optional

from backend.transcription import transcribe_audio_pipeline, transcribe_audio

app = FastAPI()

# Helper parsers
def parse_bool_env(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


# Allow CORS for frontend
allowed_origins_env = os.getenv("MNC_ALLOWED_ORIGINS")
allowed_origins = (
    [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]
    if allowed_origins_env
    else ["http://localhost:5173"]
)
allow_credentials = parse_bool_env(os.getenv("MNC_ALLOW_CREDENTIALS"), False)

# Browsers block wildcard origins when allow_credentials=True, so disable credentials in that case
if "*" in allowed_origins and allow_credentials:
    allow_credentials = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional: mock mode via env variable
USE_MOCK = parse_bool_env(os.getenv("MNC_USE_MOCK"), False)


@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    stereo_mode: bool = Form(False),
    start_offset: float = Form(0.0),
    max_duration: Optional[float] = Form(None),
):
    """
    Endpoint to handle audio file upload and return MusicXML.

    - stereo_mode: whether to keep stereo processing (for now, mid-channel).
    - start_offset: segment start (seconds) – supports your 10s-segment idea.
    - max_duration: maximum duration (seconds) to process from the offset.
    """
    try:
        # Save uploaded file temporarily
        suffix = os.path.splitext(file.filename or "upload")[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            content = await file.read()
            tmp.write(content)

        try:
            # Run full pipeline
            result = transcribe_audio_pipeline(
                tmp_path,
                stereo_mode=stereo_mode,
                use_mock=USE_MOCK,
                start_offset=start_offset,
                max_duration=max_duration,
            )

            # For now, keep API body as MusicXML for compatibility
            xml_bytes = result.musicxml.encode("utf-8")
            return Response(content=xml_bytes, media_type="application/xml")

            # If later you want timeline + notes, you can:
            # return {
            #     "musicxml": result.musicxml,
            #     "analysis": result.analysis_data.to_dict(),
            # }

        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    except HTTPException:
        raise
    except Exception as e:
        print(f"API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {"status": "ok", "mock_mode": USE_MOCK}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
