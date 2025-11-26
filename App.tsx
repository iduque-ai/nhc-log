import React, { useState, useMemo, useCallback, useEffect, useRef } from 'react';
import JSZip from 'jszip';
import { ungzip } from 'pako';
import { FileUpload } from './components/FileUpload.tsx';
import { Sidebar } from './components/Sidebar.tsx';
import { LogViewer } from './components/LogViewer.tsx';
import { LogEntry, FilterState, LogTab, LogLevel, FileInfo } from './types.ts';
import { evaluateKeywordQuery } from './utils/helpers.ts';

const ASTERISK_LOG_REGEX = /^(?<timestamp>\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+(?<hostname>\S+)\s+(?<daemon>[^\[:]+)(?:\[(?<pid>\d+)\])?:\s+(?<level>\w+)(?:\[\d+\])?:\s+(?<module>[^:]+:\d+)\s+in\s+(?<functionName>[^:]+):\s+(?<message>.*)$/;

const NIKO_LOG_REGEX = /^(?<timestamp>\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+(?<hostname>\S+)\s+(?<daemon>[^\[:]+)(?:\[(?<pid>\d+)\])?:\s+(?<level>[A-Z]+)\s+(?<module>\S+)\s+-\s+(?<message>.*)$/;

const LOG_LINE_REGEX = /^(?<timestamp>\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+(?<hostname>\S+)\s+(?<daemon>\S+?)(?:\[(?<pid>\d+)\])?:\s+(?:\[(?<level>\w+)\])?\s*(?:\[(?<module>[^\]]+)\])?(?::)?\s*(?<message>.*)$/;

// Updated to support both "Sep 11..." and ISO "2025-09-11..." timestamps
const SYSLOG_FILE_REGEX = /^(?<timestamp>\S+(?:\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)?)\s+(?<hostname>\S+)\s+(?<message>.*)$/;

const determineLevelFromMessage = (message: string): LogLevel => {
    const upperMsg = message.toUpperCase();
    if (upperMsg.includes('CRITICAL')) return LogLevel.CRITICAL;
    if (upperMsg.includes('ERROR') || upperMsg.includes('ERR')) return LogLevel.ERROR;
    if (upperMsg.includes('WARNING') || upperMsg.includes('WARN')) return LogLevel.WARNING;
    if (upperMsg.includes('NOTICE')) return LogLevel.NOTICE;
    if (upperMsg.includes('INFO')) return LogLevel.INFO;
    if (upperMsg.includes('DEBUG')) return LogLevel.DEBUG;
    if (upperMsg.includes('VERBOSE')) return LogLevel.VERBOSE;
    return LogLevel.UNKNOWN;
};

const parseTimestamp = (timestampStr: string): Date | null => {
    // Handle timestamps without a year by assuming the current year.
    // Per user request, log times are in UTC, so append 'UTC' to parse correctly.
    // Check if it's a syslog style (Sep 11 ...) or ISO style (2024-...)
    
    let parsedTimestamp;
    
    if (/^\w{3}\s+\d/.test(timestampStr)) {
         const parsed = new Date(`${timestampStr} ${new Date().getFullYear()} UTC`);
         if (!isNaN(parsed.getTime())) return parsed;
    }

    if (/^\d+$/.test(timestampStr)) {
        // Numeric timestamp is likely epoch milliseconds, which is already UTC.
        parsedTimestamp = new Date(parseInt(timestampStr, 10));
    } else {
        // For string timestamps, attempt to parse as UTC if no timezone is specified.
        if (!/Z|[+-]\d{2}(:?\d{2})?$/.test(timestampStr)) {
            // For ISO-like strings (containing 'T'), appending 'Z' is the standard for UTC.
            // For other formats, appending ' UTC' often works.
            const utcTimestampString = timestampStr.includes('T')
                ? `${timestampStr}Z`
                : `${timestampStr} UTC`;
            parsedTimestamp = new Date(utcTimestampString);

            // If parsing as UTC fails, fall back to default browser parsing
            if (isNaN(parsedTimestamp.getTime())) {
                parsedTimestamp = new Date(timestampStr);
            }
        } else {
            // Timestamp already has timezone info.
            parsedTimestamp = new Date(timestampStr);
        }
    }

    if (isNaN(parsedTimestamp.getTime())) return null;
    return parsedTimestamp;
}

const extractDaemonFromFileName = (fileName: string): string => {
    const parts = fileName.split('.');
    // Look for a segment that matches YYYYMMDD (6-8 digits) OR YYYY-MM-DD
    const dateIndex = parts.findIndex(part => /^\d{6,8}$|^\d{4}-\d{2}-\d{2}$/.test(part));
    
    if (dateIndex !== -1 && dateIndex + 1 < parts.length) {
        return parts[dateIndex + 1];
    }
    return '';
};

const parseLogLine = (line: string, id: number, fileName: string): LogEntry | null => {
  if (!line || line.trim() === '') return null;

  // 1. Special Handling for .syslog files
  if (fileName.includes('.syslog')) {
      const syslogMatch = line.match(SYSLOG_FILE_REGEX);
      if (syslogMatch?.groups) {
          const { timestamp, hostname, message } = syslogMatch.groups;
          const parsedTimestamp = parseTimestamp(timestamp);
          if (!parsedTimestamp) return null;

          return {
              id,
              timestamp: parsedTimestamp,
              hostname,
              daemon: extractDaemonFromFileName(fileName),
              pid: 0,
              level: determineLevelFromMessage(message),
              module: 'unknown',
              functionName: 'unknown',
              message: message.trim(),
          };
      }
      // If regex fails, fall through or return null? Fall through in case other regexes catch it.
  }

  // 2. Asterisk Format
  const asteriskMatch = line.match(ASTERISK_LOG_REGEX);
  if (asteriskMatch?.groups) {
    const { timestamp, hostname, daemon, pid, level, module, functionName, message } = asteriskMatch.groups;
    const logLevel = Object.values(LogLevel).includes(level?.toUpperCase() as LogLevel)
      ? level.toUpperCase() as LogLevel
      : LogLevel.UNKNOWN;

    const parsedTimestamp = parseTimestamp(timestamp);
    if (!parsedTimestamp) return null;

    return {
      id,
      timestamp: parsedTimestamp,
      hostname,
      daemon: daemon.trim(),
      pid: pid ? parseInt(pid, 10) : 0,
      level: logLevel,
      module: module || 'unknown',
      functionName: functionName || 'unknown',
      message: message.trim(),
    };
  }

  // 3. Niko/Coco Format (Daemon[PID]: LEVEL Module - Message)
  const nikoMatch = line.match(NIKO_LOG_REGEX);
  if (nikoMatch?.groups) {
      const { timestamp, hostname, daemon, pid, level, module, message } = nikoMatch.groups;
      const logLevel = Object.values(LogLevel).includes(level?.toUpperCase() as LogLevel)
      ? level.toUpperCase() as LogLevel
      : LogLevel.UNKNOWN;
      
      const parsedTimestamp = parseTimestamp(timestamp);
      if (!parsedTimestamp) return null;

      return {
          id,
          timestamp: parsedTimestamp,
          hostname,
          daemon: daemon.trim(),
          pid: pid ? parseInt(pid, 10) : 0,
          level: logLevel,
          module: module || 'unknown',
          functionName: 'unknown',
          message: message.trim(),
      };
  }

  // 4. Standard Generic Log Line
  const match = line.match(LOG_LINE_REGEX);
  if (match?.groups) {
    const { timestamp, hostname, daemon, pid, level, module, message } = match.groups;
    const logLevel = Object.values(LogLevel).includes(level?.toUpperCase() as LogLevel)
      ? level.toUpperCase() as LogLevel
      : LogLevel.UNKNOWN;

    const parsedTimestamp = parseTimestamp(timestamp);
    if (!parsedTimestamp) return null;

    let cleanedMessage = message.trim();
    let parsedFunctionName = 'unknown';

    // Regex to find function name at the end of the message, e.g., (my_func) or [in file.c (my_func)]
    const functionRegex = /(?:\((?<funcName1>[a-zA-Z0-9_]+)\)|\[in\s+.*:\d+\s+\((?<funcName2>[a-zA-Z0-9_]+)\)\])\s*$/;
    const functionNameMatch = cleanedMessage.match(functionRegex);
    
    if (functionNameMatch?.groups) {
        parsedFunctionName = functionNameMatch.groups.funcName1 || functionNameMatch.groups.funcName2 || 'unknown';
        cleanedMessage = cleanedMessage.replace(functionRegex, '').trim();
    }

    let extractedDaemon = daemon.trim();
    if (!extractedDaemon || extractedDaemon.toLowerCase() === 'unknown') {
        const fromFile = extractDaemonFromFileName(fileName);
        if (fromFile) extractedDaemon = fromFile;
    }

    return {
      id,
      timestamp: parsedTimestamp,
      hostname,
      daemon: extractedDaemon,
      pid: pid ? parseInt(pid, 10) : 0,
      level: logLevel,
      module: module || 'unknown',
      functionName: parsedFunctionName,
      message: cleanedMessage,
    };
  }
  
  // 5. Simple/Fallback Match (Handles standard syslog without explicit Level column)
  // Updated regex to support both "Sep 11..." and ISO timestamps
  const simpleMatch = line.match(/^(?<timestamp>\S+(?:\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)?)\s+(?<hostname>\S+)\s+(?<daemon>[^:\s]+)(?:\[(?<pid>\d+)\])?:\s+(?<message>.*)$/);
  
  if (simpleMatch?.groups) {
    const { timestamp, hostname, daemon, pid, message } = simpleMatch.groups;
    
    const parsedTimestamp = parseTimestamp(timestamp);
    if (!parsedTimestamp) return null;
    
    let extractedDaemon = daemon.trim();
    // Fallback if daemon is 'unknown' or empty
    if (!extractedDaemon || extractedDaemon.toLowerCase() === 'unknown') {
        const fromFile = extractDaemonFromFileName(fileName);
        if (fromFile) extractedDaemon = fromFile;
    }

    return {
      id,
      timestamp: parsedTimestamp,
      hostname,
      daemon: extractedDaemon,
      pid: pid ? parseInt(pid, 10) : 0,
      level: LogLevel.UNKNOWN,
      module: 'unknown',
      functionName: 'unknown',
      message: message.trim(),
    };
  }
  
  return null;
};

const INITIAL_FILTER_STATE: FilterState = {
  selectedLevels: [],
  selectedDaemons: [],
  selectedModules: [],
  selectedFunctionNames: [],
  dateRange: [null, null],
  keywordQueries: [],
  enableKeywordHighlight: false,
  deviceId: '',
};

const INITIAL_VIEW_STATE = {
  logsPerPage: 500,
  searchQuery: '',
  searchMatchCase: false,
  searchMatchWholeWord: false,
  searchUseRegex: false,
};

const App: React.FC = () => {
  const [fileInfos, setFileInfos] = useState<FileInfo[]>([]);
  const [tabs, setTabs] = useState<LogTab[]>([]);
  const [activeTabId, setActiveTabId] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedTimezone, setSelectedTimezone] = useState<string>('UTC');
  const [editingTabId, setEditingTabId] = useState<number | null>(null);
  const [editingTabName, setEditingTabName] = useState('');
  const [scrollToLogId, setScrollToLogId] = useState<number | null>(null);
  const [currentKeywordInput, setCurrentKeywordInput] = useState('');
  const fileIdCounter = useRef(0);
  const logIdCounter = useRef(0);

  const processFiles = async (files: File[]): Promise<FileInfo[]> => {
    const newFileInfos: FileInfo[] = [];

    for (const file of files) {
      if (file.name.endsWith('.zip')) {
        const zip = new JSZip();
        const content = await zip.loadAsync(file);
        const zipEntries: JSZip.JSZipObject[] = [];
        content.forEach((_, zipEntry) => {
          if (!zipEntry.dir && !zipEntry.name.startsWith('__MACOSX/')) {
            zipEntries.push(zipEntry);
          }
        });
        
        for (const zipEntry of zipEntries) {
            const textContent = await zipEntry.async('string');
            const parsedLogs = textContent
                .split('\n')
                .map((line) => parseLogLine(line, logIdCounter.current++, zipEntry.name))
                .filter((log): log is LogEntry => log !== null);
            
            newFileInfos.push({
                id: `file-${fileIdCounter.current++}`,
                name: zipEntry.name,
                size: new Blob([textContent]).size,
                logs: parsedLogs
            });
        }
      } else if (file.name.endsWith('.gz')) {
        try {
            const buffer = await file.arrayBuffer();
            // Explicitly cast to string to avoid TypeScript 'any' inference errors
            const textContent = ungzip(new Uint8Array(buffer), { to: 'string' }) as string;
            const parsedLogs = textContent
                .split('\n')
                .map((line: string) => parseLogLine(line, logIdCounter.current++, file.name))
                .filter((log): log is LogEntry => log !== null);
            
            newFileInfos.push({
                id: `file-${fileIdCounter.current++}`,
                name: file.name,
                size: new Blob([textContent]).size,
                logs: parsedLogs
            });
        } catch (e) {
            console.error(`Error processing .gz file ${file.name}:`, e);
            alert(`Failed to decompress ${file.name}. It might be corrupted or not a valid gzip file.`);
        }
      } else {
        const textContent = await file.text();
        const parsedLogs = textContent
          .split('\n')
          .map((line) => parseLogLine(line, logIdCounter.current++, file.name))
          .filter((log): log is LogEntry => log !== null);
        
        newFileInfos.push({
            id: `file-${fileIdCounter.current++}`,
            name: file.name,
            size: file.size,
            logs: parsedLogs
        });
      }
    }
    return newFileInfos;
  };

  const handleInitialUpload = async (files: File[]) => {
    setIsLoading(true);
    logIdCounter.current = 0; // Reset for new uploads
    const newFileInfos = await processFiles(files);
    
    const allLogs = newFileInfos.flatMap(f => f.logs);
    let minDate: Date | null = null;
    let maxDate: Date | null = null;

    if (allLogs.length > 0) {
      // Create a temporary sorted array just to find the date range
      const sortedForDateRange = [...allLogs].sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());
      minDate = sortedForDateRange[0].timestamp;
      maxDate = sortedForDateRange[sortedForDateRange.length - 1].timestamp;
    }

    setFileInfos(newFileInfos);

    const allLogsTabId = Date.now();
    const initialFiltersForTabs: FilterState = { 
      ...INITIAL_FILTER_STATE,
      dateRange: [minDate, maxDate],
    };

    const allLogsTab: LogTab = {
      id: allLogsTabId,
      name: 'All Logs',
      filters: { ...initialFiltersForTabs },
      currentPage: 1,
      scrollTop: 0,
      ...INITIAL_VIEW_STATE,
    };

    const tab1Id = allLogsTabId + 1; // Ensure unique ID
    const clonedFilters = JSON.parse(JSON.stringify(initialFiltersForTabs));

    // Rehydrate date strings to Date objects after cloning via JSON.
    if (clonedFilters.dateRange) {
      clonedFilters.dateRange = [
        clonedFilters.dateRange[0] ? new Date(clonedFilters.dateRange[0]) : null,
        clonedFilters.dateRange[1] ? new Date(clonedFilters.dateRange[1]) : null
      ];
    }

    const tab1: LogTab = {
        id: tab1Id,
        name: 'Tab 1',
        filters: clonedFilters,
        fixedFilters: clonedFilters,
        currentPage: 1,
        scrollTop: 0,
        ...INITIAL_VIEW_STATE,
    };
    
    setTabs([allLogsTab, tab1]);
    setActiveTabId(tab1Id); // Make "Tab 1" active
    setIsLoading(false);
  };
  
  const handleAppendFiles = async (files: File[]) => {
    setIsLoading(true);
    const newFileInfos = await processFiles(files);
    setFileInfos(prev => [...prev, ...newFileInfos]);
    setIsLoading(false);
  };

  const handleRemoveFile = (fileIdToRemove: string) => {
    const newFileInfos = fileInfos.filter(f => f.id !== fileIdToRemove);
    if (newFileInfos.length === 0) {
      setFileInfos([]);
      setTabs([]);
      setActiveTabId(null);
    } else {
      setFileInfos(newFileInfos);
    }
  };

  const baseLogs = useMemo(() => {
    const allLogs = fileInfos.flatMap(f => f.logs);
    // Sort logs chronologically. This is the canonical order for the entire dataset.
    allLogs.sort((a, b) => {
        const timeDiff = a.timestamp.getTime() - b.timestamp.getTime();
        if (timeDiff !== 0) return timeDiff;
        // If timestamps are identical, use the original parse-order ID as a tie-breaker
        // to ensure a stable sort, as the browser's .sort() is not guaranteed to be stable.
        return a.id - b.id;
    });
    
    // The logs now have their original, stable parse-time ID and are sorted chronologically.
    // This sorted array is the canonical source of truth for all views.
    return allLogs;
  }, [fileInfos]);

  const activeTab = useMemo(() => tabs.find(tab => tab.id === activeTabId), [tabs, activeTabId]);

  const handleFilterChange = useCallback((filters: Partial<FilterState>) => {
    if (activeTabId === null) return;
    setTabs(currentTabs => currentTabs.map(tab =>
      tab.id === activeTabId ? { ...tab, filters: { ...tab.filters, ...filters }, currentPage: 1, scrollTop: 0 } : tab
    ));
  }, [activeTabId]);
  
  const handlePageChange = useCallback((page: number) => {
    if (activeTabId === null) return;
    setTabs(currentTabs => currentTabs.map(tab =>
      tab.id === activeTabId ? { ...tab, currentPage: page, scrollTop: 0 } : tab
    ));
  }, [activeTabId]);

  const handleScrollChange = useCallback((scrollTop: number) => {
    if (activeTabId === null) return;
    setTabs(currentTabs =>
        currentTabs.map(tab =>
            tab.id === activeTabId ? { ...tab, scrollTop } : tab
        )
    );
  }, [activeTabId]);

  const handleTabViewStateChange = useCallback((changes: Partial<LogTab>) => {
    if (activeTabId === null) return;
    setTabs(currentTabs =>
        currentTabs.map(tab =>
            tab.id === activeTabId ? { ...tab, ...changes } : tab
        )
    );
  }, [activeTabId]);

  const handleLogsPerPageChange = useCallback((newLogsPerPage: number) => {
    if (activeTabId === null) return;
    setTabs(currentTabs =>
      currentTabs.map(tab => {
        if (tab.id === activeTabId) {
            const currentLogsPerPage = tab.logsPerPage || 500;
            const currentPage = tab.currentPage || 1;
            // Calculate index of the first log on the current page (0-based)
            const firstLogIndex = (currentPage - 1) * currentLogsPerPage;
            // Calculate which page this log will be on with the new page size
            const newPage = Math.floor(firstLogIndex / newLogsPerPage) + 1;
            
            return { 
                ...tab, 
                logsPerPage: newLogsPerPage, 
                currentPage: newPage,
                scrollTop: 0 // Reset scroll because exact pixel position will change
            };
        }
        return tab;
      })
    );
  }, [activeTabId]);

  const handleNewTab = () => {
    const activeTab = tabs.find(tab => tab.id === activeTabId);
    // The new fixed filters are a deep copy of the active tab's current filters.
    const newFixedFilters = activeTab ? JSON.parse(JSON.stringify(activeTab.filters)) : { ...INITIAL_FILTER_STATE };
    
    // FIX: Rehydrate date strings to Date objects after cloning via JSON.
    if (newFixedFilters.dateRange) {
      newFixedFilters.dateRange = [
        newFixedFilters.dateRange[0] ? new Date(newFixedFilters.dateRange[0]) : null,
        newFixedFilters.dateRange[1] ? new Date(newFixedFilters.dateRange[1]) : null
      ];
    }

    const newTabId = Date.now();
    const tabNumbers = tabs
      .map(t => {
        const match = t.name.match(/^Tab (\d+)$/);
        return match ? parseInt(match[1], 10) : 0;
      });
    const nextTabNumber = Math.max(0, ...tabNumbers) + 1;

    const newTab: LogTab = {
      id: newTabId,
      name: `Tab ${nextTabNumber}`,
      filters: newFixedFilters,
      fixedFilters: newFixedFilters,
      currentPage: 1,
      scrollTop: 0,
      ...INITIAL_VIEW_STATE,
    };
    setTabs([...tabs, newTab]);
    setActiveTabId(newTabId);
  };
  
  const handleCloseTab = (tabIdToClose: number) => {
    // Prevent closing the first ("All Logs") tab
    if (tabIdToClose === tabs[0]?.id) return;
    
    const tabIndex = tabs.findIndex(tab => tab.id === tabIdToClose);
    if (tabIndex === -1) return;
    const newTabs = tabs.filter(tab => tab.id !== tabIdToClose);
    if (activeTabId === tabIdToClose) {
      const newActiveIndex = Math.max(0, tabIndex - 1);
      setActiveTabId(newTabs[newActiveIndex].id);
    }
    setTabs(newTabs);
  };
  
  const handleRenameTab = (tabId: number, newName: string) => {
    if (!newName.trim()) {
      setEditingTabId(null);
      return;
    }
    setTabs(currentTabs =>
      currentTabs.map(tab => (tab.id === tabId ? { ...tab, name: newName.trim() } : tab))
    );
    setEditingTabId(null);
  };

  const handleRowDoubleClick = useCallback((log: LogEntry) => {
    const allLogsTab = tabs[0];
    if (allLogsTab) {
        setActiveTabId(allLogsTab.id);
        setScrollToLogId(log.id);
    }
  }, [tabs]);

  const handleScrollComplete = useCallback(() => {
    setScrollToLogId(null);
  }, []);

  const handleKeywordClick = useCallback((keyword: string) => {
    setCurrentKeywordInput(prev => (prev ? prev.trim() + ' ' : '') + keyword);
  }, []);

  const { allDaemons, allModules, allFunctionNames } = useMemo(() => {
    const daemons = new Set<string>();
    const modules = new Set<string>();
    const functionNames = new Set<string>();
    baseLogs.forEach(log => {
      if (log.daemon) daemons.add(log.daemon);
      if (log.module) modules.add(log.module);
      if (log.functionName) functionNames.add(log.functionName);
    });
    return {
      allDaemons: Array.from(daemons).sort(),
      allModules: Array.from(modules).sort(),
      allFunctionNames: Array.from(functionNames).sort(),
    };
  }, [baseLogs]);

  const activeTabFilters = activeTab?.filters;
  const filteredLogs = useMemo(() => {
    if (!activeTabFilters) return [];
    
    return baseLogs.filter(log => {
      const { selectedLevels, selectedDaemons, selectedModules, selectedFunctionNames, dateRange, keywordQueries } = activeTabFilters;
      if (selectedLevels.length > 0 && !selectedLevels.includes(log.level)) return false;
      if (selectedDaemons.length > 0 && !selectedDaemons.includes(log.daemon)) return false;
      if (selectedModules.length > 0 && !selectedModules.includes(log.module)) return false;
      if (selectedFunctionNames.length > 0 && !selectedFunctionNames.includes(log.functionName)) return false;
      if (keywordQueries.length > 0 && !keywordQueries.every(q => evaluateKeywordQuery(q, log.message))) return false;
      const logTime = log.timestamp.getTime();
      if (dateRange[0] && logTime < dateRange[0].getTime()) return false;
      if (dateRange[1] && logTime > dateRange[1].getTime()) return false;
      return true;
    });
  }, [
      baseLogs, 
      activeTabFilters?.selectedLevels, 
      activeTabFilters?.selectedDaemons, 
      activeTabFilters?.selectedModules,
      activeTabFilters?.selectedFunctionNames,
      activeTabFilters?.dateRange?.[0],
      activeTabFilters?.dateRange?.[1],
      activeTabFilters?.keywordQueries
  ]);

  useEffect(() => {
    setTabs(currentTabs => {
      if (currentTabs.length === 0) return currentTabs;
  
      let tabsNeedUpdate = false;
      const updatedTabs = currentTabs.map(tab => {
        const { filters } = tab;
  
        // If only one option exists, it becomes the selection. Otherwise, filter existing selections.
        // Defensive check: use empty array fallback for potential undefined filters during state transitions
        const newSelectedDaemons = allDaemons.length === 1 ? [...allDaemons] : (filters.selectedDaemons || []).filter(d => allDaemons.includes(d));
        const newSelectedModules = allModules.length === 1 ? [...allModules] : (filters.selectedModules || []).filter(m => allModules.includes(m));
        const newSelectedFunctionNames = allFunctionNames.length === 1 ? [...allFunctionNames] : (filters.selectedFunctionNames || []).filter(f => allFunctionNames.includes(f));
        
        // Using stringify for simple array comparison
        if (
          JSON.stringify(newSelectedDaemons) !== JSON.stringify(filters.selectedDaemons || []) ||
          JSON.stringify(newSelectedModules) !== JSON.stringify(filters.selectedModules || []) ||
          JSON.stringify(newSelectedFunctionNames) !== JSON.stringify(filters.selectedFunctionNames || [])
        ) {
          tabsNeedUpdate = true;
          return {
            ...tab,
            filters: {
              ...filters,
              selectedDaemons: newSelectedDaemons,
              selectedModules: newSelectedModules,
              selectedFunctionNames: newSelectedFunctionNames,
            },
          };
        }
  
        return tab;
      });
  
      return tabsNeedUpdate ? updatedTabs : currentTabs;
    });
  }, [allDaemons, allModules, allFunctionNames]);

  return (
    <div className="flex h-screen bg-gray-800 text-white font-sans overflow-hidden">
      {fileInfos.length > 0 && activeTab ? (
        <>
          <Sidebar
            filterState={activeTab.filters}
            fixedFilters={activeTab.fixedFilters}
            onFilterChange={handleFilterChange}
            allDaemons={allDaemons}
            allModules={allModules}
            allFunctionNames={allFunctionNames}
            onNewTab={handleNewTab}
            isLoading={isLoading}
            fileInfos={fileInfos}
            onAppendFiles={handleAppendFiles}
            onRemoveFile={handleRemoveFile}
            selectedTimezone={selectedTimezone}
            setSelectedTimezone={setSelectedTimezone}
            filtersDisabled={activeTab.id === tabs[0]?.id}
            currentKeywordInput={currentKeywordInput}
            onCurrentKeywordInputChange={setCurrentKeywordInput}
          />
          <main className="flex-1 flex flex-col min-w-0">
            <div className="flex-shrink-0 bg-gray-900 border-b border-gray-700 flex items-center justify-between">
              <div className="flex overflow-x-auto">
                {tabs.map((tab, index) => {
                  const isAllLogsTab = index === 0;
                  const isActive = activeTabId === tab.id;

                  let tabClasses = 'flex items-center px-4 py-2 border-r border-gray-700 flex-shrink-0 whitespace-nowrap transition-colors ';
                  if (isActive) {
                    tabClasses += 'bg-gray-800 text-white';
                  } else {
                    if (isAllLogsTab) {
                      tabClasses += 'bg-gray-950 text-gray-400 hover:bg-gray-700 hover:text-white';
                    } else {
                      tabClasses += 'bg-gray-900 text-white hover:bg-gray-700';
                    }
                  }
                  tabClasses += isAllLogsTab ? ' cursor-default' : ' cursor-pointer';

                  return (
                    <div
                      key={tab.id}
                      onClick={() => editingTabId !== tab.id && setActiveTabId(tab.id)}
                      onDoubleClick={() => {
                        if (isAllLogsTab) return;
                        setEditingTabId(tab.id);
                        setEditingTabName(tab.name);
                      }}
                      className={tabClasses}
                    >
                      {isAllLogsTab && (
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-2 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                          <path strokeLinecap="round" strokeLinejoin="round" d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4" />
                        </svg>
                      )}
                      {editingTabId === tab.id ? (
                        <input
                          type="text"
                          value={editingTabName}
                          onChange={e => setEditingTabName(e.target.value)}
                          onBlur={() => handleRenameTab(tab.id, editingTabName)}
                          onKeyDown={e => {
                            if (e.key === 'Enter') handleRenameTab(tab.id, editingTabName);
                            if (e.key === 'Escape') setEditingTabId(null);
                          }}
                          autoFocus
                          onFocus={e => e.target.select()}
                          className="bg-gray-700 text-white text-sm border border-gray-600 focus:ring-1 focus:ring-blue-500 rounded-sm px-1 mr-2 max-w-xs"
                        />
                      ) : (
                        <span className="text-sm mr-2 truncate max-w-xs" title={tab.name}>{tab.name}</span>
                      )}
                      {!isAllLogsTab && <button onClick={(e) => { e.stopPropagation(); handleCloseTab(tab.id); }} className="text-gray-500 hover:text-white flex-shrink-0">&times;</button>}
                    </div>
                  );
                })}
              </div>
            </div>
            <div className="flex-grow min-h-0">
              <LogViewer
                logs={filteredLogs}
                totalCount={baseLogs.length}
                selectedTimezone={selectedTimezone}
                totalDaemonCount={allDaemons.length}
                keywordQueries={activeTab.filters.keywordQueries}
                enableKeywordHighlight={activeTab.filters.enableKeywordHighlight}
                onRowDoubleClick={activeTab.id !== tabs[0]?.id ? handleRowDoubleClick : undefined}
                onKeywordClick={handleKeywordClick}
                scrollToLogId={activeTab.id === tabs[0]?.id ? scrollToLogId : null}
                onScrollComplete={handleScrollComplete}
                currentPage={activeTab.currentPage || 1}
                onPageChange={handlePageChange}
                scrollTop={activeTab.scrollTop}
                onScrollChange={handleScrollChange}
                // View State Persistence
                tabId={activeTab.id}
                logsPerPage={activeTab.logsPerPage || 500}
                onLogsPerPageChange={handleLogsPerPageChange}
                searchQuery={activeTab.searchQuery || ''}
                onSearchQueryChange={(s) => handleTabViewStateChange({ searchQuery: s })}
                searchMatchCase={activeTab.searchMatchCase || false}
                onSearchMatchCaseChange={(b) => handleTabViewStateChange({ searchMatchCase: b })}
                searchMatchWholeWord={activeTab.searchMatchWholeWord || false}
                onSearchMatchWholeWordChange={(b) => handleTabViewStateChange({ searchMatchWholeWord: b })}
                searchUseRegex={activeTab.searchUseRegex || false}
                onSearchUseRegexChange={(b) => handleTabViewStateChange({ searchUseRegex: b })}
              />
            </div>
          </main>
        </>
      ) : (
        <main className="flex-1 flex flex-col min-w-0">
          <div className="flex-grow flex flex-col items-center justify-center p-4">
            <h1 className="text-3xl font-bold mb-4">NHC Log Viewer</h1>
            <p className="text-gray-400 mb-8">Upload one or more log files to get started.</p>
            <FileUpload onUpload={handleInitialUpload} isLoading={isLoading} />
          </div>
        </main>
      )}
    </div>
  );
};

export default App;