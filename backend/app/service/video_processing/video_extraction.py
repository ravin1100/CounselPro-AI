import os
import io
import ffmpeg
import tempfile
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# Configure logging
logger = logging.getLogger(__name__)

# Define the scopes for Google Drive API access.
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

class VideoExtractor:
    """
    Highly optimized video extractor with single-pass frame extraction.
    """
    
    def __init__(self, max_workers=4):
        """Initialize the video extractor with configurable parallelism"""
        logger.info("Initializing OptimizedVideoExtractor")
        self.service = None
        self.temp_dir = tempfile.mkdtemp()
        self.max_workers = max_workers
        logger.debug(f"Created temporary directory: {self.temp_dir}")
        
    def get_drive_service(self):
        """Handles the Google Drive API authentication using service account."""
        try:
            if os.path.exists('credentials.json'):
                creds = service_account.Credentials.from_service_account_file(
                    'credentials.json',
                    scopes=SCOPES
                )
                logger.info("Successfully loaded service account credentials")
            else:
                error_msg = "Service account key file 'credentials.json' not found"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)

            self.service = build('drive', 'v3', credentials=creds)
            logger.info("Successfully built Google Drive service")
            return self.service

        except Exception as e:
            logger.error(f"Error setting up Google Drive service: {e}")
            raise Exception(f"Failed to authenticate with Google Drive: {str(e)}")

    def _extract_all_frames_single_pass(self, temp_video_path, timestamps, width, height):
        """
        Extract all frames in a single ffmpeg pass using select filter with memory optimization.
        Processes frames in batches to avoid memory issues with large videos.
        """
        try:
            if not timestamps:
                return {}
            
            # Sort timestamps for better performance
            sorted_timestamps = sorted(timestamps)
            
            # Memory optimization: Resize frames for analysis if they're too large
            # Face detection doesn't need full 4K resolution
            target_width = min(640, width)  # Limit to 640px width max
            target_height = int(height * (target_width / width))
            
            # Calculate memory usage and determine batch size
            frame_size_mb = (target_width * target_height * 3) / (1024 * 1024)  # MB per frame
            max_memory_mb = 500  # Limit to 500MB for frame processing
            batch_size = max(10, int(max_memory_mb / frame_size_mb))  # At least 10 frames per batch
            
            logger.info(f"Extracting {len(timestamps)} frames in batches of {batch_size} (resized to {target_width}x{target_height})")
            
            frames = {}
            
            # Process timestamps in batches to manage memory
            for batch_start in range(0, len(sorted_timestamps), batch_size):
                batch_end = min(batch_start + batch_size, len(sorted_timestamps))
                batch_timestamps = sorted_timestamps[batch_start:batch_end]
                
                logger.info(f"Processing batch {batch_start//batch_size + 1}/{(len(sorted_timestamps) + batch_size - 1)//batch_size}")
                
                # Create select filter expression for this batch
                select_expr = '+'.join([f'eq(t,{ts})' for ts in batch_timestamps])
                
                # Extract frames for this batch with resizing
                process = (
                    ffmpeg
                    .input(temp_video_path)
                    .filter('select', select_expr)
                    .filter('scale', target_width, target_height)  # Resize frames
                    .output('pipe:', format='rawvideo', pix_fmt='bgr24',
                           **{'preset': 'ultrafast', 'threads': '0'})  # Performance optimizations
                    .run_async(pipe_stdout=True, pipe_stderr=True, quiet=True)
                )
                
                batch_frame_size = target_height * target_width * 3  # 3 bytes per pixel (BGR)
                
                # Read frames for this batch
                for i, timestamp in enumerate(batch_timestamps):
                    try:
                        # Read frame data
                        frame_data = process.stdout.read(batch_frame_size)
                        
                        if len(frame_data) == batch_frame_size:
                            # Convert to numpy array
                            frame_array = np.frombuffer(frame_data, np.uint8)
                            frame = frame_array.reshape((target_height, target_width, 3))
                            frames[timestamp] = frame
                        else:
                            logger.warning(f"Incomplete frame data at timestamp {timestamp}")
                            frames[timestamp] = None
                            
                    except Exception as e:
                        logger.warning(f"Error reading frame at {timestamp}s: {e}")
                        frames[timestamp] = None
                
                # Wait for batch process to complete
                try:
                    process.wait(timeout=30)  # 30 second timeout per batch
                except:
                    logger.warning("Process timeout, terminating batch")
                    process.terminate()
            
            successful_frames = len([f for f in frames.values() if f is not None])
            logger.info(f"Single-pass extraction completed: {successful_frames}/{len(timestamps)} successful")
            
            return frames
            
        except Exception as e:
            logger.error(f"Single-pass extraction failed: {e}")
            # Fallback to interval-based extraction
            return self._extract_frames_interval_method(temp_video_path, timestamps, width, height)

    def _extract_frames_interval_method(self, temp_video_path, timestamps, width, height):
        """
        Alternative method: extract frames at regular intervals and pick closest matches.
        Memory optimized with frame resizing.
        """
        try:
            if not timestamps:
                return {}
                
            # Find min and max timestamps
            min_ts, max_ts = min(timestamps), max(timestamps)
            
            # Calculate optimal sampling rate
            total_duration = max_ts - min_ts
            avg_interval = total_duration / len(timestamps) if len(timestamps) > 1 else 1
            
            # Use a sampling rate that captures frames near our target timestamps
            sample_rate = max(0.5, min(avg_interval / 2, 2.0))  # Sample every 0.5-2 seconds
            
            # Memory optimization: Resize frames
            target_width = min(640, width)
            target_height = int(height * (target_width / width))
            
            logger.info(f"Using interval method with {sample_rate}s intervals, resizing to {target_width}x{target_height}")
            
            # Extract frames at regular intervals with resizing
            process = (
                ffmpeg
                .input(temp_video_path, ss=min_ts)
                .filter('fps', fps=f'1/{sample_rate}')
                .filter('scale', target_width, target_height)
                .output('pipe:', format='rawvideo', pix_fmt='bgr24', 
                       t=total_duration, **{'preset': 'ultrafast', 'threads': '0'})
                .run_async(pipe_stdout=True, pipe_stderr=True, quiet=True)
            )
            
            frames = {}
            frame_size = target_height * target_width * 3
            current_time = min_ts
            
            # Read frames and match to closest timestamps
            while True:
                try:
                    frame_data = process.stdout.read(frame_size)
                    if len(frame_data) < frame_size:
                        break
                        
                    # Find closest target timestamp
                    closest_ts = min(timestamps, key=lambda x: abs(x - current_time))
                    if abs(closest_ts - current_time) <= sample_rate and closest_ts not in frames:
                        frame_array = np.frombuffer(frame_data, np.uint8)
                        frame = frame_array.reshape((target_height, target_width, 3))
                        frames[closest_ts] = frame
                        
                    current_time += sample_rate
                    
                except Exception as e:
                    logger.warning(f"Error in interval extraction: {e}")
                    break
            
            process.terminate()
            return frames
            
        except Exception as e:
            logger.error(f"Interval extraction failed: {e}")
            return {}

    def _extract_frames_optimized_seeking(self, temp_video_path, timestamps, width, height):
        """
        Optimized seeking method - extract frames in timestamp order to minimize seeking.
        """
        try:
            frames = {}
            sorted_timestamps = sorted(timestamps)
            
            # Group nearby timestamps to minimize seeking
            timestamp_groups = []
            current_group = [sorted_timestamps[0]]
            
            for ts in sorted_timestamps[1:]:
                if ts - current_group[-1] <= 5:  # Group timestamps within 5 seconds
                    current_group.append(ts)
                else:
                    timestamp_groups.append(current_group)
                    current_group = [ts]
            timestamp_groups.append(current_group)
            
            logger.info(f"Processing {len(timestamp_groups)} timestamp groups")
            
            for group in timestamp_groups:
                try:
                    start_ts = group[0]
                    duration = group[-1] - group[0] + 1  # Add buffer
                    
                    # Extract segment
                    process = (
                        ffmpeg
                        .input(temp_video_path, ss=start_ts, t=duration)
                        .output('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{width}x{height}')
                        .run_async(pipe_stdout=True, pipe_stderr=True, quiet=True)
                    )
                    
                    # Read frames and match to timestamps
                    frame_size = height * width * 3
                    frame_count = 0
                    
                    while frame_count < len(group):
                        frame_data = process.stdout.read(frame_size)
                        if len(frame_data) < frame_size:
                            break
                            
                        current_ts = start_ts + (frame_count * duration / len(group))
                        closest_target = min(group, key=lambda x: abs(x - current_ts))
                        
                        if abs(closest_target - current_ts) <= duration / len(group):
                            frame_array = np.frombuffer(frame_data, np.uint8)
                            frame = frame_array.reshape((height, width, 3))
                            frames[closest_target] = frame
                            
                        frame_count += 1
                    
                    process.terminate()
                    
                except Exception as e:
                    logger.warning(f"Error processing timestamp group: {e}")
                    continue
            
            return frames
            
        except Exception as e:
            logger.error(f"Optimized seeking failed: {e}")
            return {}

    def get_video_frames_and_audio_paths(self, video_url: str, smart_sampling=True):
        """
        Optimized processing method with single-pass frame extraction.
        """
        try:
            if not self.service:
                self.get_drive_service()
            
            # Extract file ID from URL
            if '/file/d/' in video_url:
                file_id = video_url.split('/file/d/')[1].split('/')[0]
            elif '/id=' in video_url:
                file_id = video_url.split('/id=')[1].split('&')[0]
            else:
                error_msg = "Invalid Google Drive URL format"
                logger.error(error_msg)
                raise ValueError(error_msg)

            logger.info(f"Starting optimized video processing for file ID: {file_id}")
            
            # Download video
            temp_video = os.path.join(self.temp_dir, 'temp_video.mp4')
            
            request = self.service.files().get_media(fileId=file_id, acknowledgeAbuse=True)
            file_content = io.BytesIO()
            downloader = MediaIoBaseDownload(file_content, request)
            done = False
            
            while not done:
                status, done = downloader.next_chunk()
                if not done:
                    progress = int(status.progress() * 100)
                    if progress % 20 == 0:
                        logger.info(f"Download progress: {progress}%")

            # Write to file
            file_content.seek(0)
            with open(temp_video, 'wb') as f:
                f.write(file_content.getvalue())

            logger.info("Video downloaded, analyzing metadata")
            
            # Get video metadata
            probe = ffmpeg.probe(temp_video)
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            
            fps = float(video_info['r_frame_rate'].split('/')[0]) / float(video_info['r_frame_rate'].split('/')[1])
            duration = float(probe['format']['duration'])
            width = int(video_info['width'])
            height = int(video_info['height'])
            
            metadata = {
                'duration': duration,
                'fps': fps,
                'width': width,
                'height': height,
                'total_frames': int(duration * fps)
            }
            
            logger.info(f"Video metadata: {duration:.1f}s, {fps:.1f} fps, {width}x{height}")
            
            # Smart sampling strategy
            sampling_interval = int(os.getenv("FRAME_SAMPLING_INTERVAL", "60"))
            
            if smart_sampling and duration > 60:
                # Sample more densely at the beginning and end, sparse in middle
                start_samples = list(range(0, min(30, int(duration//3)), 2))
                middle_samples = list(range(30, int(duration*2//3), sampling_interval*2))
                end_samples = list(range(max(int(duration*2//3), 30), int(duration), 2))
                timestamps = start_samples + middle_samples + end_samples
                timestamps = [t for t in timestamps if t < duration]
            else:
                timestamps = list(range(0, int(duration), sampling_interval))
            
            # Remove duplicates and sort
            timestamps = sorted(list(set(timestamps)))
            
            logger.info(f"Extracting {len(timestamps)} frames using single-pass method")
            
            # Use single-pass extraction (much faster!)
            frames = self._extract_all_frames_single_pass(temp_video, timestamps, width, height)
            
            # Extract audio in parallel
            logger.info("Extracting audio")
            audio_path = os.path.join(self.temp_dir, 'extracted_audio.wav')

            stream = ffmpeg.input(temp_video)
            stream = ffmpeg.output(stream, audio_path,
                                 acodec='pcm_s16le',
                                 ar='16000',
                                 ac='1',
                                 preset='ultrafast')

            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            logger.info(f"Audio extracted to: {audio_path}")
            
            # Clean up video file
            if os.path.exists(temp_video):
                os.unlink(temp_video)
            
            successful_frames = len([f for f in frames.values() if f is not None])
            logger.info(f"Processing completed: {successful_frames}/{len(timestamps)} frames extracted")
            
            return {
                'frames': frames,
                'audio_path': audio_path,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error in optimized video processing: {e}", exc_info=True)
            raise Exception(f"Failed to process video: {str(e)}")

    def cleanup(self):
        """Clean up temporary directory and files"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            logger.info(f"Cleaning up temporary directory: {self.temp_dir}")
            try:
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                self.temp_dir = None
                logger.info("Cleanup completed successfully")
            except Exception as e:
                logger.warning(f"Error during cleanup: {e}")

    def __del__(self):
        """Cleanup when object is destroyed"""
        self.cleanup()