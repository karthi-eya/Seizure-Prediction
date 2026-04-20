import { BrainScanViewer } from "./components/BrainScanViewer";
import { WaveformGraph } from "./components/WaveformGraph";
import { MetricCard } from "./components/MetricCard";
import { PatientHeader } from "./components/PatientHeader";
import { ActivityTimeline } from "./components/ActivityTimeline";
import { Sidebar } from "./components/Sidebar";

export default function App() {
  return (
    <div className="min-h-screen bg-gray-50 flex">
      <Sidebar />
      <div className="flex-1 max-h-screen overflow-y-auto">
        <div className="max-w-[1600px] w-full p-8 space-y-6">
          <PatientHeader />

        <div className="grid grid-cols-4 gap-6">
          <MetricCard
            title="Cerebral Activity"
            value="87.4"
            unit="%"
            status="normal"
            icon="brain"
          />
          <MetricCard
            title="Heart Rate"
            value="72"
            unit="bpm"
            status="normal"
            icon="heart"
          />
          <MetricCard
            title="Alpha Waves"
            value="8.2"
            unit="Hz"
            status="normal"
            icon="activity"
          />
          <MetricCard
            title="Neural Response"
            value="142"
            unit="ms"
            status="elevated"
            icon="zap"
          />
        </div>

        <div className="grid grid-cols-3 gap-6">
          <div className="col-span-2">
            <BrainScanViewer />
          </div>
          <div>
            <ActivityTimeline />
          </div>
        </div>

        <div className="space-y-4">
          <WaveformGraph
            title="EEG - Alpha Rhythm"
            unit="8-13 Hz"
            color="#3b82f6"
          />
          <WaveformGraph
            title="EEG - Beta Rhythm"
            unit="13-30 Hz"
            color="#8b5cf6"
          />
          <WaveformGraph
            title="Heart Rate Variability"
            unit="ms"
            color="#10b981"
          />
        </div>
        </div>
      </div>
    </div>
  );
}