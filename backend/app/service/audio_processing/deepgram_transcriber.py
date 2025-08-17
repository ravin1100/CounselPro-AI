#!/usr/bin/env python3
import os
import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from uuid import UUID
from dotenv import load_dotenv
from deepgram import DeepgramClient, PrerecordedOptions, FileSource
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.models.raw_transcript import RawTranscript
from app.models.session import CounselingSession
from app.schemas.raw_transcript_schema import RawTranscriptCreate
from app.service.raw_transcript_service import create_raw_transcript
from app.exceptions.custom_exception import BadRequestException, NotFoundException

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DeepgramTranscriber:
    def __init__(
        self, api_key: Optional[str] = None, db_session: Optional[AsyncSession] = None
    ):
        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPGRAM_API_KEY not found")

        self.client = DeepgramClient(self.api_key)
        self.db_session = db_session
        logger.info("DeepgramTranscriber initialized for database storage")

    def _format_time(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"

    def _extract_utterances(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        utterances = []
        dg_utterances = response.get("results", {}).get("utterances", [])

        for utterance in dg_utterances:
            utterances.append(
                {
                    "speaker": utterance.get("speaker", 0),
                    "text": utterance.get("transcript", ""),
                    "start_time": self._format_time(utterance.get("start", 0)),
                    "end_time": self._format_time(utterance.get("end", 0)),
                    "confidence": round(utterance.get("confidence", 0), 2),
                }
            )

        return utterances

    async def transcribe_chunk(
        self, chunk_path: str, chunk_index: int
    ) -> Dict[str, Any]:
        chunk_file = Path(chunk_path)

        logger.info(f"Transcribing chunk {chunk_index:03d}: {chunk_file.name}")

        options = PrerecordedOptions(
            model="nova-2",
            language="en-US",
            punctuate=True,
            diarize=True,
            smart_format=True,
            utterances=True,
            utt_split=0.8,
        )

        with open(chunk_path, "rb") as audio_file:
            payload: FileSource = {"buffer": audio_file.read()}

        start_time = time.time()
        response = self.client.listen.rest.v("1").transcribe_file(payload, options)  # type: ignore
        processing_time = time.time() - start_time

        utterances = self._extract_utterances(response.to_dict())  # type: ignore

        formatted_output = {
            "metadata": {
                "chunk_index": chunk_index,
                "chunk_file": f"{chunk_file.name}",
                "processing_time_seconds": round(processing_time, 2),
                "deepgram_model": "nova-2",
                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "api_version": "v1",
            },
            "utterances": utterances,
        }

        logger.info(f"Completed chunk {chunk_index:03d} - {len(utterances)} utterances")
        return formatted_output

    async def transcribe_chunks(self, chunk_paths: List[str]) -> Dict[str, Any]:
        logger.info(f"Starting transcription of {len(chunk_paths)} chunks")

        all_transcripts = []
        for i, chunk_path in enumerate(chunk_paths, 1):
            transcript_data = await self.transcribe_chunk(chunk_path, i)
            all_transcripts.append(transcript_data)

        combined_transcript = {
            "metadata": {
                "total_chunks": len(chunk_paths),
                "total_utterances": sum(len(t["utterances"]) for t in all_transcripts),
                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
            "chunks": all_transcripts,
        }

        return combined_transcript

    async def transcribe_and_store(
        self, chunk_paths: List[str], session_uid: UUID
    ) -> Dict[str, Any]:
        """Transcribe audio chunks and store the results directly in the database"""
        if not self.db_session:
            raise ValueError("Database session is required for storing transcripts")

        # Check if session exists
        stmt = select(CounselingSession).where(CounselingSession.uid == session_uid)
        result = await self.db_session.execute(stmt)
        session = result.scalar_one_or_none()

        if not session:
            raise NotFoundException(
                details=f"Counseling session {session_uid} not found"
            )

        # Check if transcript already exists for this session
        stmt = select(RawTranscript).where(RawTranscript.session_id == session.id)
        result = await self.db_session.execute(stmt)
        existing_transcript = result.scalar_one_or_none()

        if existing_transcript:
            raise BadRequestException(
                details=f"Transcript already exists for session {session_uid}"
            )

        # Transcribe all chunks
        transcript_data = await self.transcribe_chunks(chunk_paths)

        # Create transcript data for database
        transcript_create = RawTranscriptCreate(
            session_uid=session_uid,
            total_segments=transcript_data["metadata"]["total_utterances"],
            raw_transcript=transcript_data,
        )

        # Store in database
        transcript = await create_raw_transcript(self.db_session, transcript_create)

        return {
            "uid": transcript.uid,
            "session_uid": session_uid,
            "total_segments": transcript.total_segments,
            "message": "Transcript successfully created and stored in database",
        }


async def transcribe_session_audio(
    db: AsyncSession, session_uid: UUID, audio_file_paths: List[str]
) -> Dict[str, Any]:
    """
    Main function to transcribe audio files for a specific session and store in database

    Args:
        db: Database session
        session_uid: UUID of the counseling session
        audio_file_paths: List of paths to audio files to transcribe

    Returns:
        Dictionary with transcript details and status message
    """
    try:
        if not audio_file_paths:
            raise ValueError("No audio files provided")

        transcriber = DeepgramTranscriber(db_session=db)
        result = await transcriber.transcribe_and_store(audio_file_paths, session_uid)

        logger.info(f"Completed transcription for session {session_uid}")
        return result

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise e


# This is kept for backwards compatibility but now requires async execution
async def main():
    try:
        if len(sys.argv) < 3:
            logger.error(
                "Usage: python deepgram_transcriber.py <session_uid> <audio_file1> [audio_file2 ...]"
            )
            sys.exit(1)

        session_uid_str = sys.argv[1]
        chunk_paths = sys.argv[2:]

        # This would normally be done by FastAPI's dependency injection
        # Just a placeholder to show how it would work in command-line context
        logger.error(
            "This script can no longer be run directly - use the API endpoints instead"
        )
        sys.exit(1)

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
