import { useState } from 'react';
import { Download, Play, Volume2, Footprints, Clock, CheckCircle2 } from 'lucide-react';

function ResultViewer({ result, jobId }) {
  const [isDownloading, setIsDownloading] = useState(false);

  // Debug: Log the result to see what data we're receiving
  console.log('ResultViewer received result:', result);
  console.log('jobId:', jobId);

  const handleDownload = async (type) => {
    setIsDownloading(true);
    
    try {
      const endpoint = type === 'video'
        ? `http://localhost:8000/download/${jobId}/video`
        : `http://localhost:8000/download/${jobId}/audio`;
      
      const response = await fetch(endpoint);
      
      if (!response.ok) {
        throw new Error('Download failed');
      }

      // Get filename from Content-Disposition header or use default
      const contentDisposition = response.headers.get('Content-Disposition');
      let filename = type === 'video' ? 'footsteps_video.mp4' : 'footsteps_audio.wav';
      
      if (contentDisposition) {
        const filenameMatch = contentDisposition.match(/filename="?(.+)"?/);
        if (filenameMatch) {
          filename = filenameMatch[1];
        }
      }

      // Download the file
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
      
    } catch (error) {
      alert('Download failed: ' + error.message);
    } finally {
      setIsDownloading(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto">
      {/* Success Banner */}
      <div className="bg-gradient-to-r from-green-50 to-emerald-50 border border-green-200 rounded-lg p-6 mb-6">
        <div className="flex items-center gap-3 mb-4">
          <CheckCircle2 className="h-8 w-8 text-green-600" />
          <div>
            <h2 className="text-2xl font-bold text-green-900">Processing Complete!</h2>
            <p className="text-green-700">Your video with AI-generated footsteps is ready</p>
          </div>
        </div>

        {/* Quick Stats */}
        <div className="grid grid-cols-3 gap-4 mt-4">
          <div className="bg-white bg-opacity-60 rounded-lg p-3 text-center">
            <div className="flex items-center justify-center gap-2 mb-1">
              <Footprints className="h-4 w-4 text-green-600" />
              <span className="text-2xl font-bold text-gray-900">
                {result.num_footsteps || 0}
              </span>
            </div>
            <p className="text-xs text-gray-600">Detected Footsteps</p>
          </div>
          
          <div className="bg-white bg-opacity-60 rounded-lg p-3 text-center">
            <div className="flex items-center justify-center gap-2 mb-1">
              <Clock className="h-4 w-4 text-green-600" />
              <span className="text-2xl font-bold text-gray-900">
                {result.processing_time_seconds ? result.processing_time_seconds.toFixed(1) + 's' : 'N/A'}
              </span>
            </div>
            <p className="text-xs text-gray-600">Processing Time</p>
          </div>

          <div className="bg-white bg-opacity-60 rounded-lg p-3 text-center">
            <div className="flex items-center justify-center gap-2 mb-1">
              <Play className="h-4 w-4 text-green-600" />
              <span className="text-2xl font-bold text-gray-900">
                {result.video_info?.duration ? result.video_info.duration.toFixed(1) + 's' : 'N/A'}
              </span>
            </div>
            <p className="text-xs text-gray-600">Video Duration</p>
          </div>
        </div>
      </div>

      {/* Generated Video Preview */}
      <div className="max-w-4xl mx-auto mb-6">
        <div className="bg-white rounded-lg shadow-sm overflow-hidden border-2 border-blue-200">
          <div className="bg-gradient-to-r from-blue-50 to-blue-100 px-4 py-3 border-b border-blue-200">
            <h3 className="font-semibold text-blue-900 flex items-center gap-2">
              <Volume2 className="h-4 w-4 text-blue-600" />
              Video with AI-Generated Footsteps
            </h3>
            <p className="text-xs text-blue-700 mt-1">Spatialized audio with depth and panning</p>
          </div>
          <div className="p-4">
            <video
              src={`http://localhost:8000/preview/${jobId}/generated`}
              controls
              className="w-full rounded bg-black"
              style={{ maxHeight: '500px' }}
            />
          </div>
        </div>
      </div>

      {/* Audio Details */}
      {result.audio_prompt && (
        <div className="bg-white rounded-lg shadow-sm p-6 mb-6">
          <h3 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
            <Volume2 className="h-5 w-5 text-blue-500" />
            Generated Audio Description
          </h3>
          <p className="text-gray-700 bg-gray-50 p-4 rounded-lg">
            "{result.audio_prompt}"
          </p>
        </div>
      )}

      {/* Download Buttons */}
      <div className="bg-white rounded-lg shadow-sm p-6">
        <h3 className="font-semibold text-gray-900 mb-4">Download Your Files</h3>
        <div className="flex flex-col sm:flex-row gap-3">
          {/* Download Video */}
          <button
            onClick={() => handleDownload('video')}
            disabled={isDownloading}
            className="flex-1 flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-medium py-3 px-6 rounded-lg transition-all hover:shadow-md active:scale-95 disabled:cursor-not-allowed"
          >
            <Download className="h-5 w-5" />
            {isDownloading ? 'Downloading...' : 'Video File with Generated Footsteps'}
          </button>

          {/* Download Audio Only */}
          <button
            onClick={() => handleDownload('audio')}
            disabled={isDownloading}
            className="flex-1 flex items-center justify-center gap-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-400 text-white font-medium py-3 px-6 rounded-lg transition-all hover:shadow-md active:scale-95 disabled:cursor-not-allowed"
          >
            <Download className="h-5 w-5" />
            {isDownloading ? 'Downloading...' : 'Audio File (.wav)'}
          </button>
        </div>
        
        <p className="text-xs text-gray-500 mt-3 text-center">
          Files will be downloaded to your default Downloads folder
        </p>
      </div>
    </div>
  );
}

export default ResultViewer;