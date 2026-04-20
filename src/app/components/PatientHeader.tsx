import { User, Calendar, FileText } from "lucide-react";

export function PatientHeader() {
  return (
    <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-100">
      <div className="flex items-start justify-between">
        <div className="flex items-start gap-4">
          <div className="w-16 h-16 rounded-full bg-gray-100 flex items-center justify-center">
            <User className="w-8 h-8 text-gray-400" strokeWidth={1.5} />
          </div>
          <div>
            <h2 className="text-xl text-gray-900 mb-1">Sarah Mitchell</h2>
            <div className="flex items-center gap-4 text-sm text-gray-500">
              <span>Female • 42 years</span>
              <span>•</span>
              <span>ID: NM-2026-4829</span>
            </div>
            <div className="flex items-center gap-2 mt-2">
              <Calendar className="w-4 h-4 text-gray-400" strokeWidth={1.5} />
              <span className="text-sm text-gray-500">Last Visit: April 12, 2026</span>
            </div>
          </div>
        </div>
        <button className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
          <FileText className="w-4 h-4" strokeWidth={1.5} />
          View History
        </button>
      </div>
    </div>
  );
}
