import { useState, useEffect } from 'react';
import { Loader2, CheckCircle2, XCircle, Download, Video, Footprints, Eye, Music, Volume2, FileVideo } from 'lucide-react';
import ResultViewer from './ResultViewer';

function ProcessingStatus({ jobId, taskId }) {
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!taskId) return;

    // Poll for status every 2 seconds using taskId (Celery task ID)
    const pollInterval = setInterval(async () => {
      try {
        const response = await fetch(`http://localhost:8000/status/${taskId}`);
        const data = await response.json();

        setStatus(data);

        // Stop polling if completed or failed
        if (data.status === 'completed' || data.status === 'failed') {
          clearInterval(pollInterval);
        }
      } catch (err) {
        setError(err.message);
        clearInterval(pollInterval);
      }
    }, 2000);

    // Cleanup on unmount
    return () => clearInterval(pollInterval);
  }, [taskId]);

  if (!status) {
    return (
      <div className="max-w-4xl mx-auto p-6 bg-white rounded-lg shadow-sm">
        <div className="flex items-center gap-3">
          <Loader2 className="h-6 w-6 animate-spin text-blue-500" />
          <p className="text-gray-700">Starting processing...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-4xl mx-auto p-6 bg-red-50 border border-red-200 rounded-lg">
        <div className="flex items-center gap-3 mb-2">
          <XCircle className="h-6 w-6 text-red-500" />
          <h3 className="font-semibold text-red-800">Error</h3>
        </div>
        <p className="text-red-700">{error}</p>
      </div>
    );
  }

  // Define stages with icons
  const stages = [
    { 
      key: 'video_validated', 
      label: 'Video Validation',
      icon: Video,
      description: 'Checking video format'
    },
    { 
      key: 'pose_detected', 
      label: 'Footstep Detection',
      icon: Footprints,
      description: 'Analyzing movement'
    },
    { 
      key: 'scene_analyzed', 
      label: 'Scene Analysis',
      icon: Eye,
      description: 'Understanding environment'
    },
    { 
      key: 'audio_generated', 
      label: 'Audio Generation',
      icon: Music,
      description: 'Creating footstep sounds'
    },
    { 
      key: 'spatial_processed', 
      label: 'Spatial Audio',
      icon: Volume2,
      description: 'Adding 3D positioning'
    },
    { 
      key: 'video_merged', 
      label: 'Exporting Video',
      icon: FileVideo,
      description: 'Finalizing output'
    },
  ];

  // Calculate current stage index
  const getCurrentStageIndex = () => {
    if (status.status === 'completed') return stages.length;
    
    for (let i = stages.length - 1; i >= 0; i--) {
      if (status.result && status.result[stages[i].key]) {
        return i + 1;
      }
    }
    return 0;
  };

  const currentStageIndex = getCurrentStageIndex();
  const progressPercentage = (currentStageIndex / stages.length) * 100;

  // Show ResultViewer when completed
  if (status.status === 'completed' && status.result) {
    return <ResultViewer result={status.result} jobId={jobId} />;
  }

  return (
    <div className="max-w-4xl mx-auto">
      {/* Status Card */}
      <div className="bg-white rounded-lg shadow-sm p-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">Processing Your Video</h2>
            <p className="text-gray-500 mt-1">
              {status.status === 'processing' && `Step ${currentStageIndex} of ${stages.length}`}
              {status.status === 'failed' && 'Processing failed'}
            </p>
          </div>
          <span className={`
            px-4 py-2 rounded-full text-sm font-semibold
            ${status.status === 'processing' ? 'bg-blue-100 text-blue-700' : ''}
            ${status.status === 'failed' ? 'bg-red-100 text-red-700' : ''}
          `}>
            {status.status === 'processing' && '⏳ Processing'}
            {status.status === 'failed' && '✗ Failed'}
          </span>
        </div>

        {/* Progress Bar */}
        <div className="mb-12">
          <div className="relative">
            {/* Background line */}
            <div className="absolute top-6 left-0 right-0 h-1 bg-gray-200 rounded-full" />

            {/* Progress line */}
            <div
              className="absolute top-6 left-0 h-1 bg-blue-500 rounded-full transition-all duration-500 ease-out"
              style={{ width: `${progressPercentage}%` }}
            />

            {/* Stage indicators */}
            <div className="relative flex justify-between">
              {stages.map((stage, index) => {
                const isComplete = index < currentStageIndex;
                const isCurrent = index === currentStageIndex && status.status === 'processing';
                const Icon = stage.icon;

                return (
                  <div key={stage.key} className="flex flex-col items-center" style={{ width: '120px' }}>
                    {/* Icon circle */}
                    <div className={`
                      relative z-10 flex items-center justify-center w-12 h-12 rounded-full border-2 transition-all duration-300
                      ${isComplete ? 'bg-blue-500 border-blue-500' : ''}
                      ${isCurrent ? 'bg-white border-blue-500 shadow-lg' : ''}
                      ${!isComplete && !isCurrent ? 'bg-white border-gray-300' : ''}
                    `}>
                      {isComplete ? (
                        <CheckCircle2 className="h-6 w-6 text-white" />
                      ) : isCurrent ? (
                        <Icon className="h-5 w-5 text-blue-500 animate-pulse" />
                      ) : (
                        <Icon className="h-5 w-5 text-gray-400" />
                      )}
                    </div>

                    {/* Label */}
                    <div className="mt-3 text-center">
                      <p className={`
                        text-xs font-medium leading-tight
                        ${isComplete ? 'text-blue-600' : ''}
                        ${isCurrent ? 'text-blue-600' : ''}
                        ${!isComplete && !isCurrent ? 'text-gray-400' : ''}
                      `}>
                        {stage.label}
                      </p>
                      {isCurrent && (
                        <p className="text-xs text-gray-500 mt-1">
                          {stage.description}
                        </p>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

      </div>

      {/* Failed State */}
      {status.status === 'failed' && (
        <div className="mt-6 bg-red-50 border border-red-200 rounded-lg p-6">
          <div className="flex items-center gap-3 mb-3">
            <XCircle className="h-7 w-7 text-red-600" />
            <h3 className="font-bold text-red-900 text-lg">Processing Failed</h3>
          </div>
          <p className="text-red-700 mb-4">{status.error || 'An unknown error occurred during processing'}</p>
          <p className="text-sm text-red-600">Please try uploading your video again or contact support if the issue persists.</p>
        </div>
      )}
    </div>
  );
}

export default ProcessingStatus;