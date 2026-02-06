import { useState } from 'react';
import { Upload, Film, X, AlertCircle, Loader2 } from 'lucide-react';

function VideoUpload({ onUploadComplete }) {
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [videoPreview, setVideoPreview] = useState(null);
  const [error, setError] = useState(null);
  const [isUploading, setIsUploading] = useState(false);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    
    const file = e.dataTransfer.files[0];
    handleFile(file);
  };

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    handleFile(file);
  };

  const handleFile = (file) => {
    setError(null); // Clear previous errors

    // Validate file type
    if (!file.type.startsWith('video/')) {
      setError('Please upload a video file (MP4, MOV, AVI, etc.)');
      return;
    }

    // Validate file size (max 100MB)
    const maxSize = 100 * 1024 * 1024; // 100MB
    if (file.size > maxSize) {
      setError('File too large. Maximum size is 100MB');
      return;
    }

    setSelectedFile(file);

    // Create video preview
    const videoUrl = URL.createObjectURL(file);
    setVideoPreview(videoUrl);
  };

  const clearSelection = () => {
    setSelectedFile(null);
    setError(null);
    if (videoPreview) {
      URL.revokeObjectURL(videoPreview);
      setVideoPreview(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setIsUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        onUploadComplete({ jobId: data.job_id, taskId: data.task_id });
      } else {
        setError(data.detail || 'Upload failed. Please try again.');
        setIsUploading(false);
      }
    } catch (_err) {
      setError('Network error. Please check if the server is running.');
      setIsUploading(false);
    }
  };

  return (
    <div className="w-full max-w-2xl mx-auto">
      {/* Error Message */}
      {error && (
        <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start gap-3">
          <AlertCircle className="h-5 w-5 text-red-500 flex-shrink-0 mt-0.5" />
          <div>
            <p className="font-medium text-red-800">Upload Error</p>
            <p className="text-sm text-red-600 mt-1">{error}</p>
          </div>
        </div>
      )}

      {!selectedFile ? (
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={`
            border-2 border-dashed rounded-lg p-12 text-center cursor-pointer
            transition-all duration-200
            ${isDragging 
              ? 'border-blue-500 bg-blue-50 scale-105' 
              : 'border-gray-300 hover:border-gray-400 hover:bg-gray-50'
            }
          `}
          onClick={() => document.getElementById('file-input').click()}
        >
          <Upload className={`mx-auto h-12 w-12 mb-4 transition-colors ${isDragging ? 'text-blue-500' : 'text-gray-400'}`} />
          <p className="text-lg font-medium text-gray-700 mb-2">
            {isDragging ? 'Drop your video here' : 'Drag & drop your video'}
          </p>
          <p className="text-sm text-gray-500 mb-4">
            or click to browse
          </p>
          <p className="text-xs text-gray-400">
            Supports MP4, MOV, AVI • Maximum 100MB
          </p>
          <input
            id="file-input"
            type="file"
            accept="video/*"
            onChange={handleFileSelect}
            className="hidden"
          />
        </div>
      ) : (
        <div className="border rounded-lg overflow-hidden shadow-sm">
          <div className="bg-gray-50 p-4 border-b flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Film className="h-5 w-5 text-blue-500" />
              <div>
                <p className="font-medium text-gray-900">{selectedFile.name}</p>
                <p className="text-sm text-gray-500">
                  {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
            </div>
            <button
              onClick={clearSelection}
              disabled={isUploading}
              className="text-gray-400 hover:text-gray-600 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <X className="h-5 w-5" />
            </button>
          </div>
          
          {videoPreview && (
            <video
              src={videoPreview}
              controls
              className="w-full max-h-96 bg-black"
            />
          )}
          
          <div className="p-4 bg-white">
            <button
              onClick={handleUpload}
              disabled={isUploading}
              className={`
                w-full font-medium py-3 px-4 rounded-lg transition-all
                ${isUploading 
                  ? 'bg-gray-400 cursor-not-allowed' 
                  : 'bg-blue-500 hover:bg-blue-600 hover:shadow-md active:scale-95'
                }
                text-white
              `}
            >
              {isUploading ? (
                <span className="flex items-center justify-center gap-2">
                  <Loader2 className="h-5 w-5 animate-spin" />
                  Uploading...
                </span>
              ) : (
                'Generate Footstep Audio'
              )}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default VideoUpload;