import brainScanImg from "figma:asset/8fcad39013e77c5b6b11f7d12cbbea5e6a1ef958";

export function BrainScanViewer() {
  return (
    <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-100">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-gray-900">Brain Scan Analysis</h3>
        <span className="text-sm text-gray-500">MRI • T1-Weighted</span>
      </div>

      <div className="relative aspect-[16/9] bg-gray-50 rounded overflow-hidden">
        <img
          src={brainScanImg}
          alt="Brain MRI Scan"
          className="w-full h-full object-contain"
        />
      </div>

      <div className="mt-4 grid grid-cols-3 gap-4">
        <div className="text-center">
          <div className="text-2xl text-gray-900 mb-1">Normal</div>
          <div className="text-xs text-gray-500">Overall Status</div>
        </div>
        <div className="text-center">
          <div className="text-2xl text-gray-900 mb-1">1.2 T</div>
          <div className="text-xs text-gray-500">Field Strength</div>
        </div>
        <div className="text-center">
          <div className="text-2xl text-gray-900 mb-1">256×256</div>
          <div className="text-xs text-gray-500">Resolution</div>
        </div>
      </div>
    </div>
  );
}
