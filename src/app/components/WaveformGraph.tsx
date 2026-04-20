import { useEffect, useRef } from "react";

interface WaveformGraphProps {
  title: string;
  unit: string;
  color?: string;
}

export function WaveformGraph({ title, unit, color = "#3b82f6" }: WaveformGraphProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const centerY = height / 2;

    ctx.clearRect(0, 0, width, height);

    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();

    for (let x = 0; x < width; x++) {
      const frequency = 0.02;
      const amplitude = 30;
      const noise = Math.random() * 3 - 1.5;
      const y = centerY + Math.sin(x * frequency) * amplitude + noise;

      if (x === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }

    ctx.stroke();

    ctx.strokeStyle = "#e5e7eb";
    ctx.lineWidth = 1;
    for (let y = 0; y < height; y += 30) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }
  }, [color]);

  return (
    <div className="bg-white rounded-lg shadow-sm p-4 border border-gray-100">
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-sm text-gray-900">{title}</h4>
        <span className="text-xs text-gray-500">{unit}</span>
      </div>
      <canvas
        ref={canvasRef}
        width={600}
        height={120}
        className="w-full h-[120px] bg-gray-50 rounded"
      />
    </div>
  );
}
