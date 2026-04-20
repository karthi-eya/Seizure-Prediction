import { Clock } from "lucide-react";

interface TimelineEvent {
  time: string;
  title: string;
  description: string;
  type: "scan" | "note" | "prescription";
}

export function ActivityTimeline() {
  const events: TimelineEvent[] = [
    {
      time: "09:24 AM",
      title: "MRI Scan Completed",
      description: "T1-weighted brain scan • 256×256 resolution",
      type: "scan"
    },
    {
      time: "09:18 AM",
      title: "EEG Recording Started",
      description: "32-channel recording • Alpha wave analysis",
      type: "scan"
    },
    {
      time: "09:05 AM",
      title: "Clinical Notes Added",
      description: "Patient reports reduced headache frequency",
      type: "note"
    },
    {
      time: "08:52 AM",
      title: "Prescription Updated",
      description: "Dosage adjustment for current medication",
      type: "prescription"
    }
  ];

  const typeColors = {
    scan: "bg-blue-100 text-blue-700",
    note: "bg-gray-100 text-gray-700",
    prescription: "bg-purple-100 text-purple-700"
  };

  return (
    <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-100">
      <h3 className="text-gray-900 mb-4">Activity Timeline</h3>
      <div className="space-y-4">
        {events.map((event, index) => (
          <div key={index} className="flex gap-4">
            <div className="flex flex-col items-center">
              <div className={`w-8 h-8 rounded-full flex items-center justify-center ${typeColors[event.type]}`}>
                <Clock className="w-4 h-4" strokeWidth={1.5} />
              </div>
              {index < events.length - 1 && (
                <div className="w-px h-full bg-gray-200 mt-2" />
              )}
            </div>
            <div className="flex-1 pb-4">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-sm text-gray-900">{event.title}</span>
                <span className="text-xs text-gray-400">{event.time}</span>
              </div>
              <p className="text-sm text-gray-500">{event.description}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
