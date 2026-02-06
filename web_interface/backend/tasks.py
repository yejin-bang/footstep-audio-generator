from celery import Celery
from pathlib import Path
import sys
import os

# Add your project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import UPLOAD_DIR, RESULTS_DIR, REDIS_URL

# Import your actual pipeline
from src.pipeline.main_pipeline import FootstepAudioPipeline, PipelineConfig

# Initialize Celery
celery_app = Celery(
    "footstep_tasks",
    broker=REDIS_URL,
    backend=REDIS_URL
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)

@celery_app.task(bind=True)
def process_footstep_video(self, job_id: str, video_filename: str, mode: str = "automatic"):
    """
    Background task to process video and generate footstep audio using your real pipeline

    Args:
        job_id: Unique job identifier
        video_filename: Name of uploaded video file (e.g., "abc123.mp4")
        mode: 'automatic' or 'manual'
    """
    try:
        import time

        # Paths
        input_video_path = str(UPLOAD_DIR / video_filename)
        output_audio_path = str(RESULTS_DIR / f"{job_id}_footsteps.wav")

        # Configure your pipeline
        config = PipelineConfig(
            backend="mock",  # Your audio backend
            output_dir=str(RESULTS_DIR),
            save_intermediates=True,
            merge_video=True  # This will create the final video with audio
        )

        # Initialize pipeline
        pipeline = FootstepAudioPipeline(config)

        start_time = time.time()

        # ====================================================================
        # STEP 1: Footstep Detection (includes video validation)
        # ====================================================================
        detection_results = pipeline.footstep_detector.process_video(
            str(input_video_path), verbose=False
        )

        # Video validated by detector
        self.update_state(state='PROCESSING', meta={
            'video_validated': True,
            'video_info': detection_results['video_info']
        })

        # Footsteps detected
        num_footsteps = len(detection_results['heel_strike_detections'])
        if num_footsteps == 0:
            raise Exception("No footsteps detected in video")

        self.update_state(state='PROCESSING', meta={
            'video_validated': True,
            'pose_detected': True,
            'num_footsteps': num_footsteps,
            'video_info': detection_results['video_info'],
            'processing_time_seconds': time.time() - start_time
        })

        # ====================================================================
        # STEP 2: Scene Analysis
        # ====================================================================
        scene_results = pipeline.scene_analyzer.analyze_from_detection_results(
            detection_results
        )
        audio_prompt = scene_results[0]['prompt']

        self.update_state(state='PROCESSING', meta={
            'video_validated': True,
            'pose_detected': True,
            'scene_analyzed': True,
            'num_footsteps': num_footsteps,
            'audio_prompt': audio_prompt,
            'video_info': detection_results['video_info'],
            'processing_time_seconds': time.time() - start_time
        })

        # ====================================================================
        # STEP 3: Audio Generation
        # ====================================================================
        audio_variations = pipeline._generate_audio_variations(audio_prompt)

        if not audio_variations:
            raise Exception("Audio generation failed")

        self.update_state(state='PROCESSING', meta={
            'video_validated': True,
            'pose_detected': True,
            'scene_analyzed': True,
            'audio_generated': True,
            'num_footsteps': num_footsteps,
            'audio_prompt': audio_prompt,
            'video_info': detection_results['video_info'],
            'processing_time_seconds': time.time() - start_time
        })

        # ====================================================================
        # STEP 4: Spatial Audio Processing
        # ====================================================================
        spatial_data = pipeline._prepare_spatial_data(detection_results)
        video_info = detection_results['video_info']

        final_audio = pipeline._create_final_audio(
            audio_variations,
            spatial_data,
            video_info,
            Path(output_audio_path)
        )

        self.update_state(state='PROCESSING', meta={
            'video_validated': True,
            'pose_detected': True,
            'scene_analyzed': True,
            'audio_generated': True,
            'spatial_processed': True,
            'num_footsteps': num_footsteps,
            'audio_prompt': audio_prompt,
            'video_info': detection_results['video_info'],
            'processing_time_seconds': time.time() - start_time
        })

        # ====================================================================
        # STEP 5: Video Merging
        # ====================================================================
        merged_video_path = None
        if config.merge_video:
            from src.utils.video_merger import merge_audio_video, check_ffmpeg_installed

            if check_ffmpeg_installed():
                success, result = merge_audio_video(
                    video_path=str(input_video_path),
                    audio_path=str(output_audio_path),
                    output_path=str(RESULTS_DIR / f"{job_id}_with_footsteps.mp4")
                )

                if success:
                    merged_video_path = result

        processing_time = time.time() - start_time

        # Return final results with all stage flags
        return {
            'status': 'completed',
            'job_id': job_id,
            'video_validated': True,
            'pose_detected': True,
            'scene_analyzed': True,
            'audio_generated': True,
            'spatial_processed': True,
            'video_merged': True,
            'output_audio_path': str(output_audio_path),
            'merged_video_path': str(merged_video_path) if merged_video_path else None,
            'num_footsteps': num_footsteps,
            'processing_time_seconds': processing_time,
            'video_info': detection_results['video_info'],
            'audio_prompt': audio_prompt,
            'message': 'Footstep generation completed successfully!'
        }

    except Exception as e:
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise