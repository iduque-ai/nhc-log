import React, { useEffect, useReducer, useMemo, useRef } from 'react';
import { LogEntry, LogLevel } from '../types.ts';
import { formatDuration } from '../utils/helpers.ts';

interface SummaryDashboardProps {
  data: LogEntry[];
}

const StatCard: React.FC<{ title: string; value: string | number }> = ({ title, value }) => (
    <div className="bg-gray-800 p-2 rounded-lg border border-gray-700 text-center">
        <p className="text-xl font-bold text-white">{value}</p>
        <p className="text-xs text-gray-400">{title}</p>
    </div>
);

const ChartCard: React.FC<React.PropsWithChildren<{ title: string; className?: string }>> = ({ title, children, className = '' }) => (
    <div className={`bg-gray-800 p-2 rounded-lg border border-gray-700 ${className}`}>
        <h3 className="text-sm font-semibold text-gray-200 mb-2">{title}</h3>
        <div className="h-56 relative">
            {children}
        </div>
    </div>
);

const LoadingCharts: React.FC = () => (
    <div className="p-2 grid grid-cols-1 lg:grid-cols-2 gap-2">
        <ChartCard title="Logs by Level"><div className="flex items-center justify-center h-full text-gray-400 text-xs">Loading Chart...</div></ChartCard>
        <ChartCard title="Top 10 Daemons"><div className="flex items-center justify-center h-full text-gray-400 text-xs">Loading Chart...</div></ChartCard>
        <ChartCard title="Top 10 Functions" className="lg:col-span-2"><div className="flex items-center justify-center h-full text-gray-400 text-xs">Loading Chart...</div></ChartCard>
    </div>
);

const levelColorMap: Record<LogLevel, string> = {
  [LogLevel.DEBUG]: '#475569',
  [LogLevel.INFO]: '#3b82f6',
  [LogLevel.NOTICE]: '#0ea5e9',
  [LogLevel.VERBOSE]: '#14b8a6',
  [LogLevel.WARNING]: '#f59e0b',
  [LogLevel.ERROR]: '#ef4444',
  [LogLevel.CRITICAL]: '#a855f7',
  [LogLevel.UNKNOWN]: '#9ca3af',
};

export const SummaryDashboard: React.FC<SummaryDashboardProps> = ({ data }) => {
  const [, forceUpdate] = useReducer(x => x + 1, 0);

  useEffect(() => {
    if (window.Chart) return;
    const timer = setInterval(() => {
      if (window.Chart) {
        clearInterval(timer);
        forceUpdate();
      }
    }, 100);
    return () => clearInterval(timer);
  }, []);

  const {
    levelCounts,
    topDaemons,
    topFunctions,
    totalLogs,
    errorRate,
    timeSpan,
    uniqueDaemonCount
  } = useMemo(() => {
    if (data.length === 0) {
      return { levelCounts: [], topDaemons: [], topFunctions: [], totalLogs: 0, errorRate: '0.00%', timeSpan: 'N/A', uniqueDaemonCount: 0 };
    }

    const levels: Record<string, number> = {};
    const daemons: Record<string, number> = {};
    const functions: Record<string, number> = {};

    let errorCount = 0;
    for (const log of data) {
      levels[log.level] = (levels[log.level] || 0) + 1;
      
      if (log.daemon && log.daemon.toLowerCase() !== 'unknown') {
        daemons[log.daemon] = (daemons[log.daemon] || 0) + 1;
      }
      
      if (log.functionName && log.functionName.toLowerCase() !== 'unknown') {
        functions[log.functionName] = (functions[log.functionName] || 0) + 1;
      }

      if (log.level === LogLevel.ERROR || log.level === LogLevel.CRITICAL) {
        errorCount++;
      }
    }

    const levelOrder = [LogLevel.CRITICAL, LogLevel.ERROR, LogLevel.WARNING, LogLevel.NOTICE, LogLevel.INFO, LogLevel.VERBOSE, LogLevel.DEBUG, LogLevel.UNKNOWN];
    const levelData = Object.entries(levels)
        .map(([name, count]) => ({ name, count }))
        .sort((a, b) => levelOrder.indexOf(a.name as LogLevel) - levelOrder.indexOf(b.name as LogLevel));

    const daemonData = Object.entries(daemons)
        .map(([name, count]) => ({ name, count }))
        .sort((a, b) => b.count - a.count)
        .slice(0, 10);

    const functionData = Object.entries(functions)
        .map(([name, count]) => ({ name, count }))
        .sort((a, b) => b.count - a.count)
        .slice(0, 10);

    const firstLogTime = data[0].timestamp.getTime();
    const lastLogTime = data[data.length - 1].timestamp.getTime();
    const duration = lastLogTime - firstLogTime;
    
    return { 
      levelCounts: levelData, 
      topDaemons: daemonData, 
      topFunctions: functionData,
      totalLogs: data.length,
      errorRate: data.length > 0 ? ((errorCount / data.length) * 100).toFixed(2) + '%' : '0.00%',
      timeSpan: formatDuration(duration),
      uniqueDaemonCount: Object.keys(daemons).length,
    };
  }, [data]);

  const levelChartRef = useRef<HTMLCanvasElement>(null);
  const daemonChartRef = useRef<HTMLCanvasElement>(null);
  const functionChartRef = useRef<HTMLCanvasElement>(null);
  const chartInstances = useRef<any>({});

  useEffect(() => {
    if (!window.Chart || !levelChartRef.current || !daemonChartRef.current || !functionChartRef.current) {
        return;
    }
    
    const { Chart } = window;
    
    const commonTooltipOptions = {
        backgroundColor: '#1f2937',
        titleColor: '#e5e7eb',
        bodyColor: '#d1d5db',
        borderColor: '#374151',
        borderWidth: 1,
        titleFont: { size: 10 },
        bodyFont: { size: 10 },
        padding: 6
    };
    
    const commonScaleOptions = {
        x: { ticks: { color: '#9ca3af', font: { size: 10 } }, grid: { color: '#374151' } },
        y: { ticks: { color: '#9ca3af', font: { size: 10 } }, grid: { display: false } }
    };

    const createOrUpdateChart = (key: string, ref: React.RefObject<HTMLCanvasElement>, type: any, data: any, options: any) => {
        if (chartInstances.current[key]) chartInstances.current[key].destroy();
        const ctx = ref.current?.getContext('2d');
        if (ctx) chartInstances.current[key] = new Chart(ctx, { type, data, options });
    };

    createOrUpdateChart('levelChart', levelChartRef, 'doughnut', {
        labels: levelCounts.map(d => d.name),
        datasets: [{
            label: 'Count',
            data: levelCounts.map(d => d.count),
            backgroundColor: levelCounts.map(d => levelColorMap[d.name as LogLevel]),
            borderColor: '#1e293b',
            borderWidth: 2,
        }]
    }, {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { position: 'right', labels: { color: '#9ca3af', font: { size: 10 }, boxWidth: 10 } },
            tooltip: commonTooltipOptions,
        }
    });

    createOrUpdateChart('daemonChart', daemonChartRef, 'bar', {
        labels: topDaemons.map(d => d.name),
        datasets: [{ label: 'Count', data: topDaemons.map(d => d.count), backgroundColor: '#10b981' }]
    }, {
        indexAxis: 'y', responsive: true, maintainAspectRatio: false,
        scales: commonScaleOptions,
        plugins: { legend: { display: false }, tooltip: commonTooltipOptions }
    });

    createOrUpdateChart('functionChart', functionChartRef, 'bar', {
        labels: topFunctions.map(d => d.name),
        datasets: [{ label: 'Count', data: topFunctions.map(d => d.count), backgroundColor: '#f97316' }]
    }, {
        indexAxis: 'y', responsive: true, maintainAspectRatio: false,
        scales: commonScaleOptions,
        plugins: { legend: { display: false }, tooltip: commonTooltipOptions }
    });

    return () => {
        Object.values(chartInstances.current).forEach((chart: any) => chart.destroy());
    };
  }, [levelCounts, topDaemons, topFunctions]);

  if (!window.Chart) {
    return <LoadingCharts />;
  }
  
  const NoDataMessage = () => <div className="flex items-center justify-center h-full text-gray-400 text-xs">No data to display for the current filters.</div>;

  return (
    <div className="p-2 space-y-2">
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-2">
          <StatCard title="Total Logs" value={totalLogs.toLocaleString()} />
          <StatCard title="Error Rate" value={errorRate} />
          <StatCard title="Time Span" value={timeSpan} />
          <StatCard title="Unique Daemons" value={uniqueDaemonCount.toLocaleString()} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-2">
          <ChartCard title="Logs by Level">
              {levelCounts.length > 0 ? <canvas ref={levelChartRef}></canvas> : <NoDataMessage />}
          </ChartCard>
          
          <ChartCard title="Top 10 Daemons">
              {topDaemons.length > 0 ? <canvas ref={daemonChartRef}></canvas> : <NoDataMessage />}
          </ChartCard>

          <ChartCard title="Top 10 Functions" className="lg:col-span-2">
              {topFunctions.length > 0 ? <canvas ref={functionChartRef}></canvas> : <NoDataMessage />}
          </ChartCard>
      </div>
    </div>
  );
};