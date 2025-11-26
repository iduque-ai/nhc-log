import React, { useState, useEffect } from 'react';
import { LogEntry } from '../types.ts';
import { SummaryDashboard } from './SummaryDashboard.tsx';
import { LogTable } from './LogTable.tsx';

interface LogViewerProps {
    logs: LogEntry[];
    totalCount: number;
    selectedTimezone: string;
    totalDaemonCount: number;
    keywordQueries: string[];
    enableKeywordHighlight: boolean;
    onRowDoubleClick?: (log: LogEntry) => void;
    onKeywordClick?: (keyword: string) => void;
    scrollToLogId?: number | null;
    onScrollComplete?: () => void;
    currentPage: number;
    onPageChange: (page: number) => void;
    scrollTop?: number;
    onScrollChange?: (scrollTop: number) => void;
}

export const LogViewer: React.FC<LogViewerProps> = ({ 
    logs, 
    totalCount, 
    selectedTimezone, 
    totalDaemonCount, 
    keywordQueries, 
    enableKeywordHighlight,
    onRowDoubleClick,
    onKeywordClick,
    scrollToLogId,
    onScrollComplete,
    currentPage,
    onPageChange,
    scrollTop,
    onScrollChange
}) => {
    const [activeView, setActiveView] = useState<'data' | 'summary'>('data');

    // When the user switches to a new top-level tab, reset the view to the default 'data' view.
    useEffect(() => {
        setActiveView('data');
    }, [logs]);

    return (
        <div className="flex flex-col h-full bg-gray-900">
            <div className="flex-shrink-0 border-b border-gray-700 bg-gray-800">
                <nav className="flex space-x-1 p-1" aria-label="Log Views">
                    <button
                        onClick={() => setActiveView('summary')}
                        className={`py-2 px-4 rounded-md text-sm font-medium transition-colors ${activeView === 'summary' ? 'bg-gray-700 text-white' : 'text-gray-400 hover:bg-gray-600'}`}
                        aria-current={activeView === 'summary' ? 'page' : undefined}
                    >
                        Summary Statistics
                    </button>
                    <button
                        onClick={() => setActiveView('data')}
                        className={`py-2 px-4 rounded-md text-sm font-medium transition-colors ${activeView === 'data' ? 'bg-gray-700 text-white' : 'text-gray-400 hover:bg-gray-600'}`}
                        aria-current={activeView === 'data' ? 'page' : undefined}
                    >
                        Log Data
                    </button>
                </nav>
            </div>
            <div className="flex-grow overflow-auto">
               {activeView === 'summary' 
                    ? <SummaryDashboard data={logs} />
                    : <LogTable 
                        data={logs} 
                        totalCount={totalCount} 
                        selectedTimezone={selectedTimezone} 
                        totalDaemonCount={totalDaemonCount}
                        keywordQueries={keywordQueries}
                        enableKeywordHighlight={enableKeywordHighlight}
                        onRowDoubleClick={onRowDoubleClick}
                        onKeywordClick={onKeywordClick}
                        scrollToLogId={scrollToLogId}
                        onScrollComplete={onScrollComplete}
                        currentPage={currentPage}
                        onPageChange={onPageChange}
                        scrollTop={scrollTop}
                        onScrollChange={onScrollChange}
                      />
               }
            </div>
        </div>
    );
};