import { useState } from 'react';
import VideoUpload from './components/VideoUpload';
import ProcessingStatus from './components/ProcessingStatus';

function App() {
  const [ids, setIds] = useState(null);

  const handleUploadComplete = (uploadIds) => {
    setIds(uploadIds);
  };

  const handleReset = () => {
    setIds(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12">
        {/* Header */}
        <div className="text-center mb-8 sm:mb-12">
          <h1 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-gray-900 mb-3 sm:mb-4">
            Footstep Audio Generator
          </h1>
          <p className="text-base sm:text-lg text-gray-600 max-w-2xl mx-auto px-4">
            Upload a video and generate realistic footstep sounds using AI
          </p>
        </div>

        {/* Upload or Progress */}
        {!ids ? (
          <VideoUpload onUploadComplete={handleUploadComplete} />
        ) : (
          <>
            <ProcessingStatus jobId={ids.jobId} taskId={ids.taskId} />

            {/* Reset Button */}
            <div className="mt-6 text-center">
              <button
                onClick={handleReset}
                className="text-blue-600 hover:text-blue-700 font-medium text-sm sm:text-base"
              >
                ← Upload Another Video
              </button>
            </div>
          </>
        )}

        {/* Footer */}
        <footer className="mt-16 text-center text-sm text-gray-500">
          <p>Powered by MediaPipe, CLIP, and Stable Audio</p>
        </footer>
      </div>
    </div>
  );
}

export default App;