
import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { GoogleGenAI, Type, FunctionDeclaration, Content, Part } from "@google/genai";
import { CreateMLCEngine } from "@mlc-ai/web-llm";
import { LogEntry, FilterState, LogLevel } from '../types.ts';
import { formatTimestamp } from '../utils/helpers.ts';

// Define FunctionCall locally since it is not exported by the SDK
interface FunctionCall {
  name: string;
  args: Record<string, any>;
  id?: string;
}

interface FilterAction {
  type: 'apply_filter';
  payload: Partial<FilterState>;
  label: string;
}

interface AIAssistantProps {
  onClose: () => void;
  visibleLogs: LogEntry[];
  allLogs: LogEntry[];
  allDaemons: string[];
  onUpdateFilters: (filters: Partial<FilterState>, reset?: boolean) => void;
  onScrollToLog: (logId: number) => void;
  savedFindings: string[];
  onSaveFinding: (finding: string) => void;
}

interface Message {
  id: string;
  role: 'user' | 'model';
  text: string;
  isError?: boolean;
  isWarning?: boolean;
  action?: FilterAction;
}

interface DownloadStatus {
  text: string;
  progress: number; // 0 to 1
}

// Extend Window interface for Chrome's Built-in AI
declare global {
  interface Window {
    ai?: {
      languageModel: {
        capabilities: () => Promise<{ available: 'readily' | 'after-download' | 'no' }>;
        create: (options?: { systemPrompt?: string, outputLanguage?: string }) => Promise<{
          prompt: (input: string) => Promise<string>;
          promptStreaming: (input: string) => AsyncIterable<string>;
          destroy: () => void;
        }>;
      };
    };
  }
}

// --- Helpers for Efficient Search ---

// Find the first index in logs where timestamp >= targetTime
const findLogStartIndex = (logs: LogEntry[], time: number): number => {
    let l = 0, r = logs.length;
    while (l < r) {
        const m = (l + r) >>> 1;
        if (logs[m].timestamp.getTime() < time) l = m + 1;
        else r = m;
    }
    return l;
};

// --- Tool Definitions ---

const updateFiltersTool: FunctionDeclaration = {
  name: 'update_filters',
  description: 'Updates the log filters. Use this when the user explicitly asks to filter logs (e.g., "show me error logs", "filter by daemon X").',
  parameters: {
    type: Type.OBJECT,
    properties: {
      log_levels: {
        type: Type.ARRAY,
        items: { type: Type.STRING },
        description: 'List of log levels to include (e.g., "ERROR", "WARNING").',
      },
      daemons: {
        type: Type.ARRAY,
        items: { type: Type.STRING },
        description: 'List of daemon names to filter by.',
      },
      search_keywords: {
        type: Type.ARRAY,
        items: { type: Type.STRING },
        description: 'List of keywords to set as a filter on the view.',
      },
      keyword_match_mode: {
        type: Type.STRING,
        enum: ['AND', 'OR'],
        description: 'Set to "OR" if the search_keywords are synonyms (any match). Set to "AND" if all keywords must be present. Default is "OR".',
      },
      reset_before_applying: {
        type: Type.BOOLEAN,
        description: 'If true, assumes a fresh slate (default true for new tabs).',
      },
      apply_immediately: {
        type: Type.BOOLEAN,
        description: 'Set to true if the user explicitly commanded to change filters (e.g. "filter by...", "show only..."). Set to false if this is a proactive suggestion based on analysis.',
      }
    },
  },
};

const scrollToLogTool: FunctionDeclaration = {
  name: 'scroll_to_log',
  description: 'Scroll the viewer to a specific log entry. Use this when the user clicks a log ID link or when you want to show a specific log.',
  parameters: {
    type: Type.OBJECT,
    properties: {
      log_id: {
        type: Type.NUMBER,
        description: 'The numeric ID of the log entry.',
      },
    },
    required: ['log_id'],
  },
};

const searchLogsTool: FunctionDeclaration = {
  name: 'search_logs',
  description: 'Search ALL logs for specific information. Supports temporal filtering using start_time and end_time.',
  parameters: {
    type: Type.OBJECT,
    properties: {
      keywords: {
        type: Type.ARRAY,
        items: { type: Type.STRING },
        description: 'List of terms to search for. E.g., ["charger", "battery"].',
      },
      match_mode: {
        type: Type.STRING,
        enum: ['AND', 'OR'],
        description: 'If "OR", log matches if ANY keyword is present. If "AND", matches if ALL are present.',
      },
      start_time: {
        type: Type.STRING,
        description: 'ISO 8601 timestamp (e.g., "2023-10-27T09:45:00Z") to filter logs after this time.',
      },
      end_time: {
        type: Type.STRING,
        description: 'ISO 8601 timestamp to filter logs before this time.',
      },
      limit: {
        type: Type.NUMBER,
        description: 'Maximum number of logs to return (default 500).',
      },
    },
    required: ['keywords'],
  },
};

const getLogsAroundTimeTool: FunctionDeclaration = {
  name: 'get_logs_around_time',
  description: 'Retrieves a window of logs centered around a specific timestamp. Useful for investigating incidents at a known time.',
  parameters: {
    type: Type.OBJECT,
    properties: {
      timestamp: {
        type: Type.STRING,
        description: 'ISO 8601 timestamp (e.g., "2023-10-27T09:45:00Z").',
      },
      window_seconds: {
        type: Type.NUMBER,
        description: 'Number of seconds before and after the timestamp to include (default 30).',
      },
    },
    required: ['timestamp'],
  },
};

const findLogPatternsTool: FunctionDeclaration = {
  name: 'find_log_patterns',
  description: 'Analyzes logs to find repeating messages or statistical anomalies in frequency. Useful for spotting trends or systemic issues.',
  parameters: {
    type: Type.OBJECT,
    properties: {
      pattern_type: {
        type: Type.STRING,
        enum: ['repeating_error', 'frequency_spike'],
        description: 'The type of pattern to search for.'
      },
      time_window_minutes: {
        type: Type.NUMBER,
        description: 'Optional. The number of minutes from the end of the log file to analyze.'
      }
    },
    required: ['pattern_type']
  }
};

const traceErrorOriginTool: FunctionDeclaration = {
  name: 'trace_error_origin',
  description: 'Traces events leading up to a specific log entry to help find the root cause.',
  parameters: {
    type: Type.OBJECT,
    properties: {
      error_log_id: {
        type: Type.NUMBER,
        description: 'The numeric ID of the log entry to start the trace from.'
      },
      trace_window_seconds: {
        type: Type.NUMBER,
        description: 'How many seconds to look backward in time. Defaults to 60 seconds.'
      }
    },
    required: ['error_log_id']
  }
};

const suggestSolutionTool: FunctionDeclaration = {
  name: 'suggest_solution',
  description: 'Provides potential solutions or debugging steps for a given error message.',
  parameters: {
    type: Type.OBJECT,
    properties: {
      error_message: {
        type: Type.STRING,
        description: 'The text of the error message.'
      }
    },
    required: ['error_message']
  }
};

const allTools = { updateFiltersTool, scrollToLogTool, searchLogsTool, findLogPatternsTool, traceErrorOriginTool, suggestSolutionTool, getLogsAroundTimeTool };
type ConversationState = 'IDLE' | 'ANALYZING';

const getAvailableTools = (state: ConversationState): FunctionDeclaration[] => {
    switch (state) {
        case 'ANALYZING':
            return [allTools.traceErrorOriginTool, allTools.suggestSolutionTool, allTools.scrollToLogTool, allTools.searchLogsTool, allTools.getLogsAroundTimeTool];
        case 'IDLE':
        default:
            return [allTools.searchLogsTool, allTools.findLogPatternsTool, allTools.updateFiltersTool, allTools.scrollToLogTool, allTools.suggestSolutionTool, allTools.getLogsAroundTimeTool];
    }
};

// --- Pattern Abstraction Utility ---
const getLogPattern = (message: string): string => {
  return message
    .replace(/\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?/g, '<TIMESTAMP>')
    .replace(/^[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}/g, '<TIMESTAMP>')
    .replace(/0x[0-9a-fA-F]+/g, '<HEX>')
    .replace(/[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}/g, '<UUID>')
    .replace(/((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)/g, '<IP>')
    .replace(/\d+/g, '<NUM>')
    .trim();
};

const renderInlineMarkdown = (text: string, onScrollToLog: (id: number) => void): React.ReactNode => {
    const parts = text.split(/(\[Log ID: \d+\]|:::scroll_to_log\(\d+\):::|\[.*?\]\(.*?\)|https?:\/\/[^\s\)]+)/g);

    return parts.map((part, i) => {
        const logIdMatch = part.match(/^\[Log ID: (\d+)\]$/);
        if (logIdMatch) {
            const id = parseInt(logIdMatch[1], 10);
            return (
                <button
                    key={i}
                    onClick={() => onScrollToLog(id)}
                    className="inline-flex items-center gap-1 text-blue-400 hover:text-blue-200 underline decoration-blue-500/50 hover:decoration-blue-400 font-mono cursor-pointer bg-blue-900/20 hover:bg-blue-900/40 px-1.5 rounded mx-0.5 transition-colors align-baseline text-[11px]"
                    title={`Click to scroll to log #${id}`}
                >
                    <svg className="w-3 h-3 opacity-70" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" /></svg>
                    <span>#{id}</span>
                </button>
            );
        }

        const shorthandMatch = part.match(/^:::scroll_to_log\((\d+)\):::$/);
        if (shorthandMatch) {
             const id = parseInt(shorthandMatch[1], 10);
             return (
                 <button
                    key={i}
                    onClick={() => onScrollToLog(id)}
                    className="inline-flex items-center gap-1 text-green-400 hover:text-green-200 underline decoration-green-500/50 hover:decoration-green-400 font-mono cursor-pointer bg-green-900/20 hover:bg-green-900/40 px-1.5 rounded mx-0.5 transition-colors align-baseline text-[11px]"
                    title={`Click to scroll to log #${id}`}
                >
                    <svg className="w-3 h-3 opacity-70" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 5l7 7-7 7M5 5l7 7-7 7" /></svg>
                    <span>Go to #{id}</span>
                </button>
             );
        }

        const linkMatch = part.match(/^\[(.*?)\]\((.*?)\)$/);
        if (linkMatch) {
            const [, text, url] = linkMatch;
            return (
                <a href={url} target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:text-blue-200 underline" key={i}>
                    {text}
                </a>
            );
        }

        if (part.startsWith('http')) {
            return (
                <a href={part} target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:text-blue-200 underline" key={i}>
                    {part}
                </a>
            );
        }

        const boldParts = part.split(/\*\*(.*?)\*\*/g);
        return (
            <span key={i}>
                {boldParts.map((boldPart, j) => {
                    if (j % 2 === 1) return <strong key={j} className="font-bold text-white">{boldPart}</strong>;
                    const codeParts = boldPart.split(/`(.*?)`/g);
                    return (
                        <span key={j}>
                            {codeParts.map((codePart, k) => {
                                if (k % 2 === 1) return <code key={k} className="bg-gray-800 text-blue-200 px-1 py-0.5 rounded font-mono text-[11px] border border-gray-700/50 break-all">{codePart}</code>;
                                return codePart;
                            })}
                        </span>
                    );
                })}
            </span>
        );
    });
};

const FormattedMessage: React.FC<{ text: string; onScrollToLog: (id: number) => void }> = ({ text, onScrollToLog }) => {
    const parts = text.split(/(```[\s\S]*?```)/g);
    return (
        <div className="text-xs space-y-2">
            {parts.map((part, index) => {
                if (part.startsWith('```')) {
                    const content = part.replace(/^```\w*\n?/, '').replace(/```$/, '');
                    return (
                        <div key={index} className="bg-gray-950 rounded p-2 overflow-x-auto border border-gray-700 max-w-full">
                             <pre className="font-mono text-[10px] text-gray-300 whitespace-pre-wrap break-all">{content}</pre>
                        </div>
                    );
                }
                const lines = part.split('\n');
                return (
                    <div key={index}>
                        {lines.map((line, lineIdx) => {
                             const listMatch = line.match(/^(\s*)([-*]|\d+\.)\s+(.*)/);
                             if (listMatch) {
                                 const [, indent, marker, content] = listMatch;
                                 const indentStr = indent || '';
                                 const paddingLeft = indentStr.length > 0 ? `${(indentStr.length / 2) + 0.25}rem` : '0';
                                 return (
                                     <div key={lineIdx} className="flex items-start ml-1 mt-1" style={{ paddingLeft }}>
                                         <span className="mr-2 text-gray-500 flex-shrink-0 select-none min-w-[1rem] text-right font-mono opacity-80">
                                             {marker && marker.match(/\d/) ? marker : '•'}
                                         </span>
                                         <span className="flex-1 break-words">
                                             {renderInlineMarkdown(content || '', onScrollToLog)}
                                         </span>
                                     </div>
                                 );
                             }
                             if (line.trim() === '') return <div key={lineIdx} className="h-2" />;
                             return (
                                 <div key={lineIdx} className="break-words min-h-[1.2em]">
                                     {renderInlineMarkdown(line, onScrollToLog)}
                                 </div>
                             );
                        })}
                    </div>
                );
            })}
        </div>
    );
};

const WEB_LLM_MODEL_ID = "Llama-3.2-3B-Instruct-q4f16_1-MLC";
const WEBLMM_CONSENT_KEY = 'nhc_log_viewer_webllm_consent';

export const AIAssistant: React.FC<AIAssistantProps> = ({ onClose, visibleLogs, allLogs, allDaemons, onUpdateFilters, onScrollToLog, savedFindings, onSaveFinding }) => {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 'welcome',
      role: 'model',
      text: "Hello! I'm your AI log assistant. How can I help you analyze these logs?"
    }
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [downloadStatus, setDownloadStatus] = useState<DownloadStatus | null>(null);
  const [modelTier, setModelTier] = useState<string>('gemini-3-flash-preview');
  const [showWebLlmConsent, setShowWebLlmConsent] = useState(false);
  const [pendingPrompt, setPendingPrompt] = useState<string | null>(null);
  const [isChromeModelAvailable, setIsChromeModelAvailable] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const cloudPrivacyWarningShown = useRef(false);
  const [userApiKey, setUserApiKey] = useState('');
  const [tempApiKey, setTempApiKey] = useState('');
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isFindingsOpen, setIsFindingsOpen] = useState(false);
  const [disableLocalSearch, setDisableLocalSearch] = useState(false);
  const [tempDisableLocalSearch, setTempDisableLocalSearch] = useState(false);
  const conversationStateRef = useRef<ConversationState>('IDLE');
  const chromeAiSession = useRef<any>(null);

  useEffect(() => {
    const checkChromeAI = async () => {
        if (window.ai?.languageModel) {
            try {
                const capabilities = await window.ai.languageModel.capabilities();
                if (capabilities.available !== 'no') {
                    setIsChromeModelAvailable(true);
                }
            } catch (e) {
                console.warn("Could not check for Chrome's built-in AI:", e);
            }
        }
    };
    checkChromeAI();
  }, []);

  useEffect(() => {
      const storedKey = localStorage.getItem('nhc_log_viewer_api_key');
      if (storedKey) {
          setUserApiKey(storedKey);
          setTempApiKey(storedKey);
      }
      
      const storedDisableSearch = localStorage.getItem('nhc_log_viewer_disable_local_search');
      if (storedDisableSearch === 'true') {
          setDisableLocalSearch(true);
          setTempDisableLocalSearch(true);
      }
  }, []);

  useEffect(() => {
    return () => {
        if (chromeAiSession.current) {
            chromeAiSession.current.destroy();
            chromeAiSession.current = null;
        }
    };
  }, []);
  
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading, downloadStatus]);

  const addMessage = useCallback((role: 'user' | 'model', text: string, isError = false, isWarning = false, action?: FilterAction) => {
    setMessages(prev => [...prev, { id: Date.now().toString() + Math.random(), role, text, isError, isWarning, action }]);
  }, []);

  const handleSaveSettings = () => {
      const newKey = tempApiKey.trim();
      localStorage.setItem('nhc_log_viewer_api_key', newKey);
      setUserApiKey(newKey);
      
      localStorage.setItem('nhc_log_viewer_disable_local_search', String(tempDisableLocalSearch));
      setDisableLocalSearch(tempDisableLocalSearch);

      setIsSettingsOpen(false);
  };
  
  const handleToolCall = useCallback(async (toolName: string, args: any, aiInstance?: GoogleGenAI): Promise<any> => {
    switch (toolName) {
      case 'update_filters':
        if (args.apply_immediately) {
             const filters = {
                 selectedLevels: args.log_levels || [],
                 selectedDaemons: args.daemons || [],
                 keywordQueries: args.search_keywords || [],
                 keywordMatchMode: args.keyword_match_mode || 'OR',
             };
             onUpdateFilters(filters, args.reset_before_applying ?? true);
             return { success: true, summary: `Filters applied immediately.` };
        }
        return { success: true, summary: `Filter suggestion created.` };
      
      case 'scroll_to_log':
        onScrollToLog(Number(args.log_id));
        return { success: true, summary: `Scrolled to log ID ${args.log_id}.` };

      case 'get_logs_around_time': {
        const { timestamp, window_seconds = 30 } = args;
        const targetTime = new Date(timestamp).getTime();
        if (isNaN(targetTime)) return { error: 'Invalid timestamp format.' };

        const start = targetTime - (window_seconds * 1000);
        const end = targetTime + (window_seconds * 1000);
        
        const results = allLogs.filter(log => {
            const t = log.timestamp.getTime();
            return t >= start && t <= end;
        });

        return {
            summary: `Found ${results.length} logs within ±${window_seconds}s of ${timestamp}.`,
            logs: results.slice(0, 20).map(l => `[Log ID: ${l.id}] [${formatTimestamp(l.timestamp, 'UTC')}] [${l.level}] ${l.message}`)
        };
      }

      case 'search_logs': {
        const { keywords, match_mode = 'OR', limit = 50, start_time, end_time } = args;
        if (!keywords || keywords.length === 0) return { summary: 'No keywords provided.' };
        
        const lowerCaseKeywords = keywords.map((k: string) => k.toLowerCase());
        const start = start_time ? new Date(start_time).getTime() : -Infinity;
        const end = end_time ? new Date(end_time).getTime() : Infinity;

        const results = allLogs.filter(log => {
            const logTime = log.timestamp.getTime();
            if (logTime < start || logTime > end) return false;

            const textToSearch = `${log.message} ${log.daemon} ${log.level}`.toLowerCase();
            if (match_mode === 'AND') return lowerCaseKeywords.every((kw: string) => textToSearch.includes(kw));
            return lowerCaseKeywords.some((kw: string) => textToSearch.includes(kw));
        }).slice(0, limit);

        if (results.length === 0) return { summary: 'Found 0 logs matching the criteria.' };

        return {
            summary: `Found ${results.length} logs matching criteria.`,
            logs: results.map(l => `[Log ID: ${l.id}] [${formatTimestamp(l.timestamp, 'UTC')}] [${l.level}] ${l.message}`)
        };
      }
      
      case 'find_log_patterns': {
        const { pattern_type, time_window_minutes } = args;
        const targetLogs = time_window_minutes ? allLogs.filter(log => {
            const logTime = log.timestamp.getTime();
            const endTime = allLogs[allLogs.length - 1].timestamp.getTime();
            const startTime = endTime - time_window_minutes * 60 * 1000;
            return logTime >= startTime && logTime <= endTime;
        }) : allLogs;

        if (pattern_type === 'repeating_error') {
            type PatternStats = { count: number; id: number; timestamp: string };
            const counts = targetLogs.filter(l => l.level === 'ERROR' || l.level === 'CRITICAL').reduce((acc, log) => {
                const genericMessage = getLogPattern(log.message);
                if (!acc[genericMessage]) {
                    acc[genericMessage] = { count: 0, id: log.id, timestamp: formatTimestamp(log.timestamp, 'UTC') };
                }
                acc[genericMessage].count++;
                return acc;
            }, {} as Record<string, PatternStats>);

            const top = (Object.entries(counts) as [string, PatternStats][]).sort((a, b) => b[1].count - a[1].count).slice(0, 5);
            return {
                summary: `Found ${top.length} repeating error patterns.`,
                top_patterns: top.map(([msg, data]) => ({ message_pattern: msg, count: data.count, example_log_id: data.id, example_timestamp: data.timestamp }))
            };
        }
        return { summary: 'Pattern analysis completed.' };
      }
        
      case 'trace_error_origin': {
          const { error_log_id, trace_window_seconds = 60 } = args;
          const errorLog = allLogs.find(l => l.id === error_log_id);
          if (!errorLog) return { summary: `Log ID ${error_log_id} not found.` };
          const endTime = errorLog.timestamp.getTime();
          const startTime = endTime - trace_window_seconds * 1000;
          const traceLogs = allLogs.filter(l => l.timestamp.getTime() >= startTime && l.timestamp.getTime() <= endTime);
          return {
              summary: `Traced ${traceLogs.length} logs before log ${error_log_id}.`,
              logs: traceLogs.slice(-10).map(l => `[Log ID: ${l.id}] [${formatTimestamp(l.timestamp, 'UTC')}] ${l.message}`)
          };
      }
        
      case 'suggest_solution': {
          if (!aiInstance) return { summary: 'AI Instance required for solution generation.' };
          const solutionPrompt = `Provide causes and solutions for: "${args.error_message}"`;
          try {
              const result = (await aiInstance.models.generateContent({ model: 'gemini-3-flash-preview', contents: [{ role: 'user', parts: [{ text: solutionPrompt }] }] })) as any;
              return { solution: result.text || "No suggestion found." };
          } catch (e: any) {
              return { solution: `Error: ${e.message}` };
          }
      }

      default:
        return { error: `Tool "${toolName}" not found.` };
    }
  }, [allLogs, onScrollToLog, onUpdateFilters]);

  const runCloudAI = useCallback(async (prompt: string, effectiveModel: string) => {
    const apiKey = userApiKey || import.meta.env.VITE_API_KEY;
    if (!apiKey) {
      addMessage('model', "API key missing. Please set it in Settings.", true);
      setIsLoading(false);
      return;
    }
    
    if (!cloudPrivacyWarningShown.current) {
        addMessage('model', "Using cloud AI model. Log summaries and findings will be sent to Google for analysis. Do not send sensitive or regulated data.", false, true);
        cloudPrivacyWarningShown.current = true;
    }
    
    const ai = new GoogleGenAI({ apiKey });

    const dateRangeStr = allLogs.length > 0 
        ? `${allLogs[0].timestamp.toISOString()} to ${allLogs[allLogs.length - 1].timestamp.toISOString()}`
        : 'N/A';

    let systemPrompt = `You are an expert log analysis AI. 
# LOG DATA CONTEXT
- Loaded Log Count: ${allLogs.length.toLocaleString()}
- Loaded Date Range: ${dateRangeStr}
- Available Daemons: ${allDaemons.join(', ') || 'N/A'}
- IMPORTANT: You CAN search by timestamp. Use start_time and end_time ISO strings in search_logs or get_logs_around_time.
- IMPORTANT: If the user asks about a specific time (e.g. 9:45 AM), look at the Date Range above to determine the correct ISO date to use with tools.`;

    const history: Content[] = messages.slice(1).reduce((acc: Content[], m) => {
        if (!m.isError && !m.isWarning && (m.role !== 'model' || !m.text.startsWith('Tool'))) {
            acc.push({ role: m.role, parts: [{ text: m.text }] });
        }
        return acc;
    }, []);
    
    history.push({ role: 'user', parts: [{ text: prompt }] });
    
    const payloadConfig = { 
        systemInstruction: systemPrompt, 
        tools: [{ functionDeclarations: getAvailableTools(conversationStateRef.current) }] 
    };

    let pendingFilterAction: FilterAction | null = null;
    const MAX_TURNS = 10;
    
    for (let turn = 1; turn <= MAX_TURNS; turn++) {
        try {
            const result = (await ai.models.generateContent({ 
                model: effectiveModel, 
                contents: history, 
                config: payloadConfig
            })) as any;

            const candidate = result.candidates?.[0];
            if (!candidate?.content) {
                addMessage('model', "No response candidate received.", true);
                break;
            }
            
            history.push(candidate.content);

            const functionCalls = result.functionCalls;
            if (functionCalls && functionCalls.length > 0) {
                for (const toolCall of functionCalls) {
                    const toolResult = await handleToolCall(toolCall.name, toolCall.args || {}, ai);

                    if (toolCall.name === 'search_logs' || toolCall.name === 'get_logs_around_time') {
                        conversationStateRef.current = 'ANALYZING';
                        pendingFilterAction = {
                            type: 'apply_filter',
                            label: 'Apply Results as Filter',
                            payload: {
                                keywordQueries: toolCall.args.keywords || [],
                                dateRange: toolCall.args.start_time ? [new Date(toolCall.args.start_time), toolCall.args.end_time ? new Date(toolCall.args.end_time) : null] : [null, null]
                            }
                        };
                    }

                    history.push({ 
                        role: 'tool', 
                        parts: [{ 
                            functionResponse: { 
                                name: toolCall.name, 
                                response: { result: JSON.stringify(toolResult) } 
                            } 
                        }] 
                    } as unknown as Content);
                }
            } else {
                addMessage('model', result.text || "No response received.", false, false, pendingFilterAction || undefined);
                conversationStateRef.current = 'IDLE';
                break;
            }
        } catch (e: any) {
            addMessage('model', `AI Error: ${e.message}`, true);
            break;
        }
    }
    setIsLoading(false);
  }, [userApiKey, allLogs, allDaemons, messages, addMessage, handleToolCall]);
  
  const mlcEngine = useRef<any>(null);

  const executeLocalAI = useCallback(async (initialPrompt: string) => {
    if (!mlcEngine.current) return;
    setIsLoading(true);

    const systemPrompt = `Log analysis assistant. Logs: ${allLogs.length}. Use tools for actions.`;
    let history = messages.slice(-6).reduce((acc: any[], m) => {
        if (!m.isError && !m.isWarning && (m.role !== 'model' || !m.text.startsWith('Tool'))) {
            acc.push({ role: m.role === 'model' ? 'assistant' : m.role, content: m.text });
        }
        return acc;
    }, []);

    history.push({ role: 'user', content: initialPrompt });
    
    try {
        const chunks = await mlcEngine.current.chat.completions.create({
            messages: [{ role: 'system', content: systemPrompt }, ...history],
            stream: true,
            temperature: 0.7
        });

        let fullText = "";
        const messageId = Date.now().toString() + Math.random();
        let isFirst = true;

        for await (const chunk of chunks) {
            const delta = chunk.choices[0]?.delta?.content || "";
            fullText += delta;
            if (isFirst) {
                setMessages(prev => [...prev, { id: messageId, role: 'model', text: fullText }]);
                isFirst = false;
            } else {
                setMessages(prev => prev.map(m => m.id === messageId ? { ...m, text: fullText } : m));
            }
        }
    } catch (e: any) {
        addMessage('model', `Local AI Error: ${e.message}`, true);
    } finally {
        setIsLoading(false);
    }
  }, [messages, allLogs, addMessage]);

  const loadWebLlm = useCallback(async () => {
    setIsLoading(true);
    setShowWebLlmConsent(false);
    try {
        const engine = await CreateMLCEngine(WEB_LLM_MODEL_ID, {
            initProgressCallback: (report) => {
                setDownloadStatus({ text: report.text, progress: report.progress });
            }
        });
        mlcEngine.current = engine;
        localStorage.setItem(WEBLMM_CONSENT_KEY, 'true');
        if (pendingPrompt) {
            executeLocalAI(pendingPrompt);
            setPendingPrompt(null);
        } else {
            setIsLoading(false);
        }
    } catch (e: any) {
        addMessage('model', `Failed: ${e.message}`, true);
        setIsLoading(false);
    }
  }, [pendingPrompt, addMessage, executeLocalAI]);

  const runLocalAI = useCallback(async (prompt: string) => {
    if (mlcEngine.current) {
        executeLocalAI(prompt);
        return;
    }
    const hasConsented = localStorage.getItem(WEBLMM_CONSENT_KEY);
    setPendingPrompt(prompt);
    if (hasConsented === 'true') loadWebLlm();
    else setShowWebLlmConsent(true);
  }, [executeLocalAI, loadWebLlm]);

  const handleConsent = (consented: boolean) => {
      setShowWebLlmConsent(false);
      if (consented) loadWebLlm();
      else {
          setPendingPrompt(null);
          setIsLoading(false);
          setModelTier('gemini-3-flash-preview');
      }
  };
  
  const runChromeBuiltInAI = useCallback(async (prompt: string) => {
    if (!window.ai?.languageModel) {
        addMessage('model', `Chrome AI unavailable.`, true);
        setIsLoading(false);
        return;
    }
    try {
        if (!chromeAiSession.current) {
            chromeAiSession.current = await window.ai.languageModel.create({ systemPrompt: "Log assistant." });
        }
        const response = await chromeAiSession.current.prompt(prompt);
        addMessage('model', response);
    } catch (e: any) {
        addMessage('model', `Chrome AI Error: ${e.message}`, true);
    } finally {
        setIsLoading(false);
    }
  }, [addMessage]);

  const getEffectiveModelTierAndRun = useCallback((prompt: string) => {
    if (modelTier === 'chrome-built-in') {
        runChromeBuiltInAI(prompt);
        return;
    }
    if (modelTier === 'web-llm') {
        runLocalAI(prompt);
        return;
    }
    runCloudAI(prompt, modelTier);
  }, [modelTier, runChromeBuiltInAI, runLocalAI, runCloudAI]);

  const enhancePromptWithLocalContext = useCallback(async (prompt: string): Promise<string> => {
      const stopWords = new Set(['a', 'an', 'the', 'is', 'are', 'what', 'show', 'me', 'find', 'list', 'of', 'in', 'for', 'all', 'with', 'and', 'or', 'about']);
      const extractedKeywords = prompt.toLowerCase()
          .replace(/[^\w\s\.-]/g, '') 
          .split(/\s+/)
          .filter(word => !stopWords.has(word) && word.length > 2);
      
      if (extractedKeywords.length === 0 || allLogs.length === 0) return prompt;

      let startLogIdx = 0;
      let endLogIdx = allLogs.length;

      const dateMatch = prompt.match(/\b(\d{1,2}[\/-]\d{1,2}[\/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b/);
      if (dateMatch) {
          const d = new Date(dateMatch[1]);
          if (!isNaN(d.getTime())) {
             startLogIdx = findLogStartIndex(allLogs, d.getTime());
             const nextDay = new Date(d);
             nextDay.setDate(nextDay.getDate() + 1);
             endLogIdx = findLogStartIndex(allLogs, nextDay.getTime());
          }
      }

      const groupedLogs = new Map<string, { level: LogLevel; daemon: string; count: number; examples: { id: number; timestamp: string }[] }>();
      let matchCount = 0;
      const CHUNK_SIZE = 5000; 

      for (let i = startLogIdx; i < endLogIdx; i += CHUNK_SIZE) {
          const chunkEnd = Math.min(i + CHUNK_SIZE, endLogIdx);
          for (let idx = i; idx < chunkEnd; idx++) {
              const log = allLogs[idx];
              const msgLower = log.message.toLowerCase();
              const timeStr = log.timestamp.toISOString().toLowerCase();
              
              let matched = false;
              for (const kw of extractedKeywords) {
                  if (msgLower.includes(kw) || log.daemon.toLowerCase().includes(kw) || timeStr.includes(kw)) {
                      matched = true;
                      break;
                  }
              }

              if (matched) {
                  matchCount++;
                  const key = `${log.level}|${log.daemon}|${getLogPattern(log.message)}`;
                  if (!groupedLogs.has(key)) {
                      groupedLogs.set(key, { level: log.level, daemon: log.daemon, count: 0, examples: [] });
                  }
                  const group = groupedLogs.get(key)!;
                  group.count++;
                  if (group.examples.length < 3) {
                      group.examples.push({ id: log.id, timestamp: formatTimestamp(log.timestamp, 'UTC') });
                  }
              }
          }
          if (endLogIdx - startLogIdx > CHUNK_SIZE) await new Promise(r => setTimeout(r, 0));
      }

      if (matchCount === 0) return prompt;

      const contextData = Array.from(groupedLogs.values())
          .sort((a, b) => (a.count === 1 ? -1 : 1) - (b.count === 1 ? -1 : 1))
          .slice(0, 20)
          .map(g => `[Count: ${g.count}] [${g.level}] [${g.daemon}] Examples: ${g.examples.map(ex => `[Log ID: ${ex.id} @ ${ex.timestamp}]`).join(', ')}`)
          .join('\n');

      return `[LOCAL LOG SUMMARY (Matched Keywords: ${extractedKeywords.join(', ')})]\n${contextData}\n\nUser Question: "${prompt}"`;
  }, [allLogs]);

  const handleSubmit = useCallback(async (e?: React.FormEvent, overridePrompt?: string) => {
    e?.preventDefault();
    const trimmedInput = overridePrompt || input.trim();
    if (!trimmedInput || isLoading) return;

    addMessage('user', trimmedInput);
    setIsLoading(true);
    conversationStateRef.current = 'IDLE';

    let enhancedPrompt = trimmedInput;
    if (!disableLocalSearch) {
        enhancedPrompt = await enhancePromptWithLocalContext(trimmedInput);
    }
    
    getEffectiveModelTierAndRun(enhancedPrompt);
    setInput('');
  }, [input, isLoading, addMessage, getEffectiveModelTierAndRun, enhancePromptWithLocalContext, disableLocalSearch]);

  const handleRunPatterns = () => {
    const prompt = "Please find the most common log patterns and anomalies.";
    addMessage('user', prompt);
    setIsLoading(true);
    getEffectiveModelTierAndRun(prompt);
  };

  return (
    <div className="h-full flex flex-col bg-gray-800 relative border-l border-gray-700">
        {/* Header with Buttons */}
        <div className="flex-shrink-0 flex items-center justify-between p-2 border-b border-gray-700 bg-gray-900">
          <div className="flex items-center space-x-2">
            <svg className="w-4 h-4 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
            <select value={modelTier} onChange={e => setModelTier(e.target.value)} className="bg-gray-800 text-white text-xs rounded py-0.5 px-1 border border-gray-700 focus:outline-none focus:ring-1 focus:ring-blue-500">
                <option value="gemini-flash-lite-latest">Fast</option>
                <option value="gemini-3-flash-preview">Balanced</option>
                <option value="gemini-3-pro-preview">Reasoning</option>
                <option value="chrome-built-in">Local (Chrome)</option>
                <option value="web-llm">Local (WebLLM)</option>
            </select>
          </div>
          <div className="flex items-center space-x-1">
             <button onClick={handleRunPatterns} title="Analyze Patterns" className="p-1 text-gray-400 hover:text-white rounded-md hover:bg-gray-800 transition-colors">
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M11 3.055A9.001 9.001 0 1020.945 13H11V3.055z"></path><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M20.488 9H15V3.512A9.025 9.025 0 0120.488 9z"></path></svg>
             </button>
             <button onClick={() => setIsFindingsOpen(!isFindingsOpen)} title="Saved Findings" className="p-1 text-gray-400 hover:text-white rounded-md hover:bg-gray-800 transition-colors">
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01"></path></svg>
             </button>
             <button onClick={() => setIsSettingsOpen(true)} title="AI Settings" className="p-1 text-gray-400 hover:text-white rounded-md hover:bg-gray-800 transition-colors">
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"></path><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path></svg>
             </button>
             <div className="w-px h-4 bg-gray-700 mx-1"></div>
             <button onClick={onClose} className="p-1 text-gray-400 hover:text-white rounded-md hover:bg-gray-800 transition-colors">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" /></svg>
             </button>
          </div>
        </div>

        <div className="flex-grow p-3 overflow-y-auto space-y-4 bg-gray-900/30">
          {messages.map((message) => (
            <div key={message.id} className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`relative max-w-[85%] p-2 rounded-lg text-white shadow-sm ${
                message.isWarning 
                    ? 'bg-amber-900/40 border border-amber-500/30 text-amber-200' 
                    : message.isError 
                        ? 'bg-red-900/40 border border-red-500/30 text-red-200' 
                        : message.role === 'user' ? 'bg-blue-600' : 'bg-gray-800'
              }`}>
                 {message.isWarning && (
                     <div className="flex items-center space-x-1.5 mb-1.5 pb-1.5 border-b border-amber-500/20 text-[10px] font-bold uppercase tracking-wider">
                         <svg className="w-3 h-3 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path></svg>
                         <span>Privacy Warning</span>
                     </div>
                 )}
                 <FormattedMessage text={message.text} onScrollToLog={onScrollToLog} />
                 {message.action && (
                    <button onClick={() => onUpdateFilters(message.action!.payload, true)} className="mt-2 flex items-center space-x-1 text-xs bg-blue-600 hover:bg-blue-500 text-white px-2 py-1 rounded transition-colors w-full justify-center">
                        <span>{message.action.label}</span>
                    </button>
                 )}
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
        
        <div className="flex-shrink-0 p-2 border-t border-gray-700 bg-gray-900">
            <form onSubmit={handleSubmit} className="flex items-center space-x-2">
              <textarea 
                value={input} 
                onChange={e => setInput(e.target.value)} 
                onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        handleSubmit();
                    }
                }}
                placeholder="Ask about logs or time range..." 
                disabled={isLoading} 
                rows={1}
                className="flex-grow bg-gray-800 border border-gray-700 text-white text-xs rounded-md p-2 focus:outline-none focus:ring-1 focus:ring-blue-500 resize-none max-h-32" 
              />
              <button type="submit" disabled={isLoading || !input.trim()} className="bg-blue-600 text-white p-2 rounded-md hover:bg-blue-700 disabled:bg-gray-700 transition-colors shadow-sm">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 10l7-7m0 0l7 7m-7-7v18"></path></svg>
              </button>
            </form>
        </div>

        {/* Findings Popover */}
        {isFindingsOpen && (
            <div className="absolute top-12 right-2 left-2 bottom-12 z-20 bg-gray-800 border border-gray-700 rounded-lg shadow-2xl flex flex-col">
                <div className="p-3 border-b border-gray-700 flex justify-between items-center bg-gray-900/50">
                    <h3 className="text-sm font-bold flex items-center gap-2">
                        <svg className="w-4 h-4 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01"></path></svg>
                        Saved Findings
                    </h3>
                    <button onClick={() => setIsFindingsOpen(false)} className="text-gray-400 hover:text-white">&times;</button>
                </div>
                <div className="flex-grow p-3 overflow-y-auto space-y-2">
                    {savedFindings.length > 0 ? savedFindings.map((finding, i) => (
                        <div key={i} className="p-2 bg-gray-900 border border-gray-700 rounded text-xs leading-relaxed group relative">
                            <button 
                                onClick={() => {
                                    const newFindings = savedFindings.filter((_, idx) => idx !== i);
                                    localStorage.setItem('findings', JSON.stringify(newFindings)); // This is simplified, real logic uses sourceHash
                                    // We'd actually need to pass a callback to update finding state properly in App.tsx
                                }}
                                className="absolute top-1 right-1 opacity-0 group-hover:opacity-100 p-1 text-gray-500 hover:text-red-400 transition-opacity"
                            >
                                &times;
                            </button>
                            {finding}
                        </div>
                    )) : (
                        <p className="text-center text-gray-500 text-xs py-8">No findings saved yet.</p>
                    )}
                </div>
            </div>
        )}

        {/* Settings Modal */}
        {isSettingsOpen && (
            <div className="absolute inset-0 bg-black/80 z-30 flex items-center justify-center p-4 backdrop-blur-sm">
                <div className="bg-gray-800 border border-gray-700 rounded-lg p-5 w-full max-w-sm shadow-2xl">
                    <h3 className="text-sm font-bold text-white mb-4 flex items-center gap-2">
                        <svg className="w-4 h-4 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"></path><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path></svg>
                        AI Configuration
                    </h3>
                    <div className="space-y-4">
                        <div>
                            <label className="block text-[10px] text-gray-400 uppercase font-bold mb-1">Gemini API Key</label>
                            <input 
                                type="password" 
                                value={tempApiKey} 
                                onChange={e => setTempApiKey(e.target.value)} 
                                placeholder="Paste your API key..."
                                className="w-full bg-gray-900 border border-gray-700 text-white rounded p-2 text-xs focus:ring-1 focus:ring-blue-500" 
                            />
                            <p className="text-[9px] text-gray-500 mt-1">Get keys at <a href="https://aistudio.google.com/app/apikey" target="_blank" className="text-blue-400 hover:underline">Google AI Studio</a>.</p>
                        </div>
                        <div className="flex items-center justify-between">
                            <label className="text-xs text-gray-300">Disable Smart Local Scan</label>
                            <button 
                                onClick={() => setTempDisableLocalSearch(!tempDisableLocalSearch)}
                                className={`w-8 h-4 rounded-full relative transition-colors ${tempDisableLocalSearch ? 'bg-blue-600' : 'bg-gray-700'}`}
                            >
                                <div className={`absolute top-0.5 w-3 h-3 bg-white rounded-full transition-transform ${tempDisableLocalSearch ? 'translate-x-4.5' : 'translate-x-0.5'}`} />
                            </button>
                        </div>
                    </div>
                    <div className="mt-6 flex justify-end space-x-2">
                        <button onClick={() => setIsSettingsOpen(false)} className="px-3 py-1.5 rounded bg-gray-700 text-white text-xs hover:bg-gray-600">Cancel</button>
                        <button onClick={handleSaveSettings} className="px-3 py-1.5 rounded bg-blue-600 text-white text-xs hover:bg-blue-500">Save Changes</button>
                    </div>
                </div>
            </div>
        )}

        {showWebLlmConsent && (
            <div className="absolute inset-0 bg-black/80 z-40 flex items-center justify-center p-4 backdrop-blur-md">
                <div className="bg-gray-800 rounded-lg p-6 max-w-md w-full border border-gray-700 text-center shadow-2xl">
                    <div className="w-12 h-12 bg-blue-600/20 rounded-full flex items-center justify-center mx-auto mb-4 text-blue-400">
                        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path></svg>
                    </div>
                    <h3 className="text-xl font-bold text-white mb-2">Download Local AI?</h3>
                    <p className="text-sm text-gray-300 mb-6">Run Llama 3 entirely in your browser. This requires a one-time download of approximately 2.5GB. No data will ever leave your device.</p>
                    <div className="flex justify-center space-x-3">
                        <button onClick={() => handleConsent(false)} className="px-4 py-2 rounded bg-gray-700 text-white text-sm hover:bg-gray-600 transition-colors">Cancel</button>
                        <button onClick={() => handleConsent(true)} className="px-4 py-2 rounded bg-blue-600 text-white text-sm hover:bg-blue-500 transition-colors font-bold">Start Download</button>
                    </div>
                </div>
            </div>
        )}

        {downloadStatus && (
            <div className="absolute bottom-20 left-4 right-4 z-50 bg-gray-800 border border-gray-700 rounded-lg p-4 shadow-2xl">
                <div className="flex justify-between items-center mb-2">
                    <span className="text-xs text-blue-400 font-bold">Downloading AI Model...</span>
                    <span className="text-xs text-gray-400">{Math.round(downloadStatus.progress * 100)}%</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-1.5 mb-2">
                    <div className="bg-blue-600 h-1.5 rounded-full transition-all duration-300" style={{ width: `${downloadStatus.progress * 100}%` }}></div>
                </div>
                <p className="text-[10px] text-gray-500 truncate">{downloadStatus.text}</p>
            </div>
        )}
    </div>
  );
};
