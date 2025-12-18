
import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { GoogleGenAI, Type, FunctionDeclaration, Content, Part } from "@google/genai";
import { CreateMLCEngine } from "@mlc-ai/web-llm";
import { LogEntry, FilterState, LogLevel } from '../types.ts';

// Define FunctionCall locally since it is not exported by the SDK
interface FunctionCall {
  name: string;
  args: Record<string, any>;
  id?: string;
}

// Local definition to match SDK response structure or augment it
interface GenerateContentResponse {
  text?: string | undefined;
  functionCalls?: FunctionCall[];
  candidates?: { content?: Content }[];
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

// Find the first index where value >= target (Lower Bound)
const lowerBound = (arr: number[], value: number): number => {
    let l = 0, r = arr.length;
    while (l < r) {
        const m = (l + r) >>> 1;
        if (arr[m] < value) l = m + 1;
        else r = m;
    }
    return l;
};

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
  description: 'Updates the log filters. Use this when the user explicitly asks to filter logs (e.g., "show me error logs", "filter by daemon X"). If the user is just asking for analysis and you think filtering would help, set apply_immediately to false to show a suggestion button.',
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
  description: 'Search ALL logs for specific information. Supports temporal filtering.',
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
        description: 'Optional ISO 8601 timestamp (e.g., "2023-10-27T09:45:00Z") to filter logs after this time.',
      },
      end_time: {
        type: Type.STRING,
        description: 'Optional ISO 8601 timestamp to filter logs before this time.',
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
  description: 'Analyzes logs to find repeating messages or statistical anomalies in frequency. Useful for spotting trends or systemic issues. Returns a summary of findings.',
  parameters: {
    type: Type.OBJECT,
    properties: {
      pattern_type: {
        type: Type.STRING,
        enum: ['repeating_error', 'frequency_spike'],
        description: 'The type of pattern to search for: "repeating_error" finds the most common error messages, "frequency_spike" finds time intervals with an unusually high number of logs.'
      },
      time_window_minutes: {
        type: Type.NUMBER,
        description: 'Optional. The number of minutes from the end of the log file to analyze. Defaults to the entire log file if not provided.'
      }
    },
    required: ['pattern_type']
  }
};

const traceErrorOriginTool: FunctionDeclaration = {
  name: 'trace_error_origin',
  description: 'Traces events leading up to a specific log entry to help find the root cause. It looks backwards in time from the given log ID. Returns a summary of the trace.',
  parameters: {
    type: Type.OBJECT,
    properties: {
      error_log_id: {
        type: Type.NUMBER,
        description: 'The numeric ID of the log entry to start the trace from.'
      },
      trace_window_seconds: {
        type: Type.NUMBER,
        description: 'How many seconds to look backward in time from the error log\'s timestamp. Defaults to 60 seconds.'
      }
    },
    required: ['error_log_id']
  }
};

const suggestSolutionTool: FunctionDeclaration = {
  name: 'suggest_solution',
  description: 'Provides potential solutions or debugging steps for a given error message. This tool is for getting advice, not for searching logs. Only use this when the user explicitly asks for a solution.',
  parameters: {
    type: Type.OBJECT,
    properties: {
      error_message: {
        type: Type.STRING,
        description: 'The text of the error message to get a solution for.'
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

const MODEL_CONFIG = {
    'gemini-3-pro-preview': { name: 'Reasoning', rpm: 2 },
    'gemini-3-flash-preview': { name: 'Balanced', rpm: 10 },
    'gemini-flash-lite-latest': { name: 'Fast', rpm: 15 },
    'chrome-built-in': { name: 'Local (Chrome)', rpm: Infinity },
    'web-llm': { name: 'Local (WebLLM)', rpm: Infinity },
};

// --- Pattern Abstraction Utility ---
const getLogPattern = (message: string): string => {
  return message
    .replace(/\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?/g, '<TIMESTAMP>')
    // Syslog timestamp pattern (e.g., "Sep 11 12:34:56")
    .replace(/^[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}/g, '<TIMESTAMP>')
    .replace(/0x[0-9a-fA-F]+/g, '<HEX>')
    .replace(/[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}/g, '<UUID>')
    .replace(/((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)/g, '<IP>')
    .replace(/\d+/g, '<NUM>')
    .trim();
};

// --- Formatted Message Component ---

const renderInlineMarkdown = (text: string, onScrollToLog: (id: number) => void): React.ReactNode => {
    // Regex matches:
    // 1. [Log ID: 123] -> Clickable
    // 2. :::scroll_to_log(123)::: -> Clickable (Safety net for local hallucinations)
    // 3. [text](url) -> Link
    // 4. http(s)://... -> Link
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
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [userApiKey, setUserApiKey] = useState('');
  const [tempApiKey, setTempApiKey] = useState('');
  const conversationStateRef = useRef<ConversationState>('IDLE');
  const lastPromptRef = useRef<string | null>(null);
  const apiRequestTimestampsRef = useRef<Record<string, number[]>>({});
  const chromeAiSession = useRef<any>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [disableLocalSearch, setDisableLocalSearch] = useState(false);
  const [tempDisableLocalSearch, setTempDisableLocalSearch] = useState(false);

  // --- 1. Efficient Indexing using useMemo ---
  const logIndex = useMemo(() => {
    const levels: Record<string, number[]> = {};
    const daemons: Record<string, number[]> = {};
    const len = allLogs.length;
    for (let i = 0; i < len; i++) {
        const log = allLogs[i];
        if (!levels[log.level]) levels[log.level] = [];
        levels[log.level].push(i);
        const d = (log.daemon || '').toLowerCase();
        if (d) {
            if (!daemons[d]) daemons[d] = [];
            daemons[d].push(i);
        }
    }
    return { levels, daemons };
  }, [allLogs]);

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
            console.log('[AI] Destroying Chrome AI session on component unmount.');
            chromeAiSession.current.destroy();
            chromeAiSession.current = null;
        }
    };
  }, []);

  const handleSaveSettings = () => {
      const newKey = tempApiKey.trim();
      localStorage.setItem('nhc_log_viewer_api_key', newKey);
      setUserApiKey(newKey);
      
      localStorage.setItem('nhc_log_viewer_disable_local_search', String(tempDisableLocalSearch));
      setDisableLocalSearch(tempDisableLocalSearch);

      setIsSettingsOpen(false);
      if (newKey && lastPromptRef.current) {
          addMessage('model', "Settings saved. Retrying your last request...", false);
          handleSubmit(undefined, lastPromptRef.current);
          lastPromptRef.current = null;
      }
  };
  
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading, downloadStatus]);

  const addMessage = useCallback((role: 'user' | 'model', text: string, isError = false, isWarning = false, action?: FilterAction) => {
    setMessages(prev => [...prev, { id: Date.now().toString() + Math.random(), role, text, isError, isWarning, action }]);
  }, []);
  
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
             return { success: true, summary: `Filters applied immediately as requested.` };
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
            logs: results.slice(0, 50).map(l => `[Log ID: ${l.id}] [${l.level}] ${l.message}`)
        };
      }

      case 'search_logs': {
        const { keywords, match_mode = 'OR', limit = 500, start_time, end_time } = args;
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
            example_log_ids: results.slice(0, 5).map(l => l.id)
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
            type PatternStats = { count: number; id: number; };
            const counts = targetLogs.filter(l => l.level === 'ERROR' || l.level === 'CRITICAL').reduce((acc, log) => {
                const genericMessage = log.message.replace(/\d+/g, 'N');
                if (!acc[genericMessage]) {
                    acc[genericMessage] = { count: 0, id: log.id };
                }
                acc[genericMessage].count++;
                return acc;
            }, {} as Record<string, PatternStats>);

            const top = Object.entries(counts).sort((a: [string, PatternStats], b: [string, PatternStats]) => b[1].count - a[1].count).slice(0, 5);
            return {
                summary: `Found ${top.length} repeating error patterns.`,
                top_patterns: top.map(([msg, data]: [string, PatternStats]) => ({ message_pattern: msg, count: data.count, example_log_id: data.id }))
            };
        }
        return { summary: 'Pattern type not implemented.' };
      }
        
      case 'trace_error_origin': {
          const { error_log_id, trace_window_seconds = 60 } = args;
          const errorLog = allLogs.find(l => l.id === error_log_id);
          if (!errorLog) return { summary: `Log ID ${error_log_id} not found.` };
          const endTime = errorLog.timestamp.getTime();
          const startTime = endTime - trace_window_seconds * 1000;
          const traceLogs = allLogs.filter(l => l.timestamp.getTime() >= startTime && l.timestamp.getTime() <= endTime);
          return {
              summary: `Found ${traceLogs.length} logs in the ${trace_window_seconds}s before log ${error_log_id}.`,
              example_log_ids: traceLogs.slice(-5).map(l => l.id)
          };
      }
        
      case 'suggest_solution': {
          if (!aiInstance) return { summary: 'Local AI cannot suggest solution using Cloud Tools.' };
          const solutionPrompt = `Provide a concise list of potential causes and solutions for: "${args.error_message}"`;
          try {
              const result = await aiInstance.models.generateContent({ model: 'gemini-3-flash-preview', contents: [{ role: 'user', parts: [{ text: solutionPrompt }] }] });
              return { solution: result.text || "No solution generated." };
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
      addMessage('model', "API key is not configured.", true);
      setIsLoading(false);
      return;
    }
    
    if (!cloudPrivacyWarningShown.current) {
        addMessage('model', "You are using a cloud-based AI model. Summary data is sent to Google.", false, true);
        cloudPrivacyWarningShown.current = true;
    }
    
    const ai = new GoogleGenAI({ apiKey });

    const dateRangeStr = allLogs.length > 0 
        ? `${allLogs[0].timestamp.toISOString()} to ${allLogs[allLogs.length - 1].timestamp.toISOString()}`
        : 'N/A';

    let systemPrompt = `You are an expert log analysis AI. 
# CONTEXT
- Loaded Log Count: ${allLogs.length.toLocaleString()}
- Loaded Date Range: ${dateRangeStr}
- Available Daemons: ${allDaemons.join(', ') || 'N/A'}`;

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
        let response: GenerateContentResponse;
        try {
            const result = await ai.models.generateContent({ 
                model: effectiveModel, 
                contents: history, 
                config: payloadConfig
            });
            response = { text: result.text, functionCalls: result.functionCalls as FunctionCall[], candidates: result.candidates };
        } catch (e: any) {
            addMessage('model', `AI Error: ${e.message}`, true);
            setIsLoading(false);
            return;
        }

        const functionCalls = response.functionCalls;
        if (functionCalls && functionCalls.length > 0) {
            const toolCall = functionCalls[0];
            history.push({ role: 'model', parts: [{ functionCall: toolCall } as any] });
            
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

            history.push({ role: 'tool', parts: [{ functionResponse: { name: toolCall.name, response: { result: JSON.stringify(toolResult) } } }] } as unknown as Content);
        } else {
            addMessage('model', response.text || "No response.", false, false, pendingFilterAction || undefined);
            conversationStateRef.current = 'IDLE';
            break;
        }
    }
    setIsLoading(false);
  }, [userApiKey, allLogs, allDaemons, messages, addMessage, handleToolCall]);
  
  const mlcEngine = useRef<any>(null);

  const executeLocalAI = useCallback(async (initialPrompt: string) => {
    if (!mlcEngine.current) return;
    setIsLoading(true);

    const systemPrompt = `Helpful log analysis assistant. Logs count: ${allLogs.length}. Use tools for actions.`;
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
        addMessage('model', `Chrome built-in AI not available.`, true);
        setIsLoading(false);
        return;
    }
    try {
        if (!chromeAiSession.current) {
            chromeAiSession.current = await window.ai.languageModel.create({ systemPrompt: "Helpful log assistant." });
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

      // Improved regex for dates (DD/MM/YYYY or YYYY-MM-DD) and times
      const dateMatch = prompt.match(/\b(\d{1,2}[\/-]\d{1,2}[\/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b/);
      const timeMatch = prompt.match(/\b(\d{1,2}:\d{2})\s*(AM|PM|am|pm)?\b/);

      if (dateMatch) {
          const d = new Date(dateMatch[1]);
          if (!isNaN(d.getTime())) {
             startLogIdx = findLogStartIndex(allLogs, d.getTime());
             const nextDay = new Date(d);
             nextDay.setDate(nextDay.getDate() + 1);
             endLogIdx = findLogStartIndex(allLogs, nextDay.getTime());
          }
      }

      const scanCount = endLogIdx - startLogIdx;
      const groupedLogs = new Map<string, { level: LogLevel; daemon: string; count: number; score: number; examples: number[] }>();
      
      let matchCount = 0;
      const CHUNK_SIZE = 5000; 

      for (let i = startLogIdx; i < endLogIdx; i += CHUNK_SIZE) {
          const chunkEnd = Math.min(i + CHUNK_SIZE, endLogIdx);
          for (let idx = i; idx < chunkEnd; idx++) {
              const log = allLogs[idx];
              let score = 0;
              const msgLower = log.message.toLowerCase();
              const timeStr = log.timestamp.toISOString().toLowerCase();
              
              for (const kw of extractedKeywords) {
                  if (msgLower.includes(kw) || log.daemon.toLowerCase().includes(kw) || timeStr.includes(kw)) {
                      score += 1;
                  }
              }

              if (score > 0) {
                  matchCount++;
                  const key = `${log.level}|${log.daemon}|${getLogPattern(log.message)}`;
                  if (!groupedLogs.has(key)) {
                      groupedLogs.set(key, { level: log.level, daemon: log.daemon, count: 0, score: 0, examples: [] });
                  }
                  const group = groupedLogs.get(key)!;
                  group.count++;
                  group.score = Math.max(group.score, score);
                  if (group.examples.length < 3) group.examples.push(log.id);
              }
          }
          if (scanCount > CHUNK_SIZE) await new Promise(r => setTimeout(r, 0));
      }

      if (matchCount === 0) return prompt;

      const contextData = Array.from(groupedLogs.values())
          .sort((a, b) => (a.count === 1 ? -1 : 1) - (b.count === 1 ? -1 : 1)) // Prioritize unique logs
          .slice(0, 20)
          .map(g => `[Count: ${g.count}] [${g.level}] [${g.daemon}] Examples: ${g.examples.map(id => `[Log ID: ${id}]`).join(', ')}`)
          .join('\n');

      return `[LOCAL LOG SUMMARY]\n${contextData}\n\nUser Question: "${prompt}"`;
  }, [allLogs, logIndex]);

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

  return (
    <div className="h-full flex flex-col bg-gray-800 relative">
        <div className="flex-shrink-0 flex items-center justify-between p-2 border-b border-gray-700">
          <div className="flex items-center space-x-2">
            <svg className="w-4 h-4 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
            <h2 className="font-bold text-sm text-white">AI Assistant</h2>
            <select value={modelTier} onChange={e => setModelTier(e.target.value)} className="bg-gray-700 text-white text-xs rounded py-0.5 px-1 border border-gray-600 focus:outline-none">
                <option value="gemini-flash-lite-latest">Fast</option>
                <option value="gemini-3-flash-preview">Balanced</option>
                <option value="gemini-3-pro-preview">Reasoning</option>
                <option value="chrome-built-in">Local (Chrome)</option>
                <option value="web-llm">Local (WebLLM)</option>
            </select>
          </div>
          <button onClick={onClose} className="p-1 text-gray-400 hover:text-white rounded-md hover:bg-gray-700"><svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" /></svg></button>
        </div>

        <div className="flex-grow p-3 overflow-y-auto space-y-4">
          {messages.map((message) => (
            <div key={message.id} className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`relative max-w-[85%] p-2 rounded-lg text-white ${message.role === 'user' ? 'bg-blue-600' : 'bg-gray-700'}`}>
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
        
        <div className="flex-shrink-0 p-2 border-t border-gray-700 bg-gray-800">
            <form onSubmit={handleSubmit} className="flex items-center space-x-2">
              <input value={input} onChange={e => setInput(e.target.value)} placeholder="Ask about your logs..." disabled={isLoading} className="flex-grow bg-gray-700 border border-gray-600 text-white text-xs rounded-md p-2 focus:outline-none focus:ring-1 focus:ring-blue-500" />
              <button type="submit" disabled={isLoading || !input.trim()} className="bg-blue-600 text-white p-2 rounded-md hover:bg-blue-700 disabled:bg-gray-600"><svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 10l7-7m0 0l7 7m-7-7v18"></path></svg></button>
            </form>
        </div>

        {showWebLlmConsent && (
            <div className="absolute inset-0 bg-black/80 z-20 flex items-center justify-center p-4">
                <div className="bg-gray-800 rounded-lg p-6 max-w-md w-full border border-gray-700 text-center">
                    <h3 className="text-xl font-bold text-white mb-2">Download Local AI?</h3>
                    <p className="text-sm text-gray-300 mb-4">Download Llama 3 weights (~2.5GB)? One-time download.</p>
                    <div className="flex justify-center space-x-4">
                        <button onClick={() => handleConsent(false)} className="px-4 py-2 rounded bg-gray-600 text-white text-sm">Cancel</button>
                        <button onClick={() => handleConsent(true)} className="px-4 py-2 rounded bg-blue-600 text-white text-sm">Download</button>
                    </div>
                </div>
            </div>
        )}
    </div>
  );
};
