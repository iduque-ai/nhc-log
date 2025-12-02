
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
  description: 'Search ALL logs for specific information. NOTE: A local search is often performed before this tool is called. Check the context first. Use this tool only if the local context is insufficient or if you need to perform a broader search with different terms.',
  parameters: {
    type: Type.OBJECT,
    properties: {
      keywords: {
        type: Type.ARRAY,
        items: { type: Type.STRING },
        description: 'List of terms to search for. E.g., ["charger", "battery", "2023-10-27"].',
      },
      match_mode: {
        type: Type.STRING,
        enum: ['AND', 'OR'],
        description: 'If "OR", log matches if ANY keyword is present (good for synonyms). If "AND", matches if ALL are present.',
      },
      limit: {
        type: Type.NUMBER,
        description: 'Maximum number of logs to return (default 500).',
      },
    },
    required: ['keywords'],
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

const allTools = { updateFiltersTool, scrollToLogTool, searchLogsTool, findLogPatternsTool, traceErrorOriginTool, suggestSolutionTool };
type ConversationState = 'IDLE' | 'ANALYZING';

const getAvailableTools = (state: ConversationState): FunctionDeclaration[] => {
    switch (state) {
        case 'ANALYZING':
            return [allTools.traceErrorOriginTool, allTools.suggestSolutionTool, allTools.scrollToLogTool, allTools.searchLogsTool];
        case 'IDLE':
        default:
            return [allTools.searchLogsTool, allTools.findLogPatternsTool, allTools.updateFiltersTool, allTools.scrollToLogTool, allTools.suggestSolutionTool];
    }
};

const MODEL_CONFIG = {
    'gemini-2.5-pro': { name: 'Reasoning', rpm: 2 },
    'gemini-2.5-flash': { name: 'Balanced', rpm: 10 },
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
  const [modelTier, setModelTier] = useState<string>('gemini-2.5-flash');
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
        // Instead of executing immediately, we acknowledge the suggestion.
        return { success: true, summary: `Filter suggestion created. User can apply it via the UI.` };
      
      case 'scroll_to_log':
        onScrollToLog(Number(args.log_id));
        return { success: true, summary: `Scrolled to log ID ${args.log_id}.` };

      case 'search_logs': {
        const { keywords, match_mode = 'OR', limit = 500 } = args;
        if (!keywords || keywords.length === 0) return { summary: 'No keywords provided.' };
        
        const lowerCaseKeywords = keywords.map((k: string) => k.toLowerCase());
        const results = allLogs.filter(log => {
            const textToSearch = `${log.message} ${log.timestamp.toISOString()}`.toLowerCase();
            if (match_mode === 'AND') return lowerCaseKeywords.every((kw: string) => textToSearch.includes(kw));
            return lowerCaseKeywords.some((kw: string) => textToSearch.includes(kw));
        }).slice(0, limit);

        if (results.length === 0) return { summary: 'Found 0 logs matching the criteria.' };

        const levelCounts = results.reduce((acc, log) => {
            acc[log.level] = (acc[log.level] || 0) + 1;
            return acc;
        }, {} as Record<string, number>);

        return {
            summary: `Found ${results.length} logs. Levels: ${JSON.stringify(levelCounts)}.`,
            example_log_ids: results.slice(0, 3).map(l => l.id)
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
            if (top.length === 0) return { summary: 'No repeating error patterns found.' };
            return {
                summary: `Found ${top.length} repeating error patterns. The most common one occurred ${top[0][1].count} times.`,
                top_patterns: top.map(([msg, data]: [string, PatternStats]) => ({ message_pattern: msg, count: data.count, example_log_id: data.id }))
            };
        }

        if (pattern_type === 'frequency_spike') {
            const bucketSize = 60 * 1000; // 1 minute
            const buckets: Record<number, number> = {};
            targetLogs.forEach(log => {
                const bucket = Math.floor(log.timestamp.getTime() / bucketSize);
                buckets[bucket] = (buckets[bucket] || 0) + 1;
            });
            const counts = Object.values(buckets);
            if (counts.length < 2) return { summary: 'Not enough data to detect spikes.' };
            const avg = counts.reduce((a, b) => a + b, 0) / counts.length;
            const stdDev = Math.sqrt(counts.map(x => Math.pow(x - avg, 2)).reduce((a, b) => a + b, 0) / counts.length);
            const spikes = Object.entries(buckets).filter(([, count]) => count > avg + 2 * stdDev);
            if (spikes.length === 0) return { summary: 'No significant spikes in log frequency detected.' };
            return {
                summary: `Detected ${spikes.length} spike(s) in log activity. The largest spike had ${Math.max(...spikes.map(s => s[1]))} logs in one minute.`,
                spikes: spikes.map(([bucket, count]) => ({ timestamp: new Date(Number(bucket) * bucketSize).toISOString(), count }))
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
          const levelCounts = traceLogs.reduce((acc, log) => {
              acc[log.level] = (acc[log.level] || 0) + 1;
              return acc;
          }, {} as Record<string, number>);
          return {
              summary: `Found ${traceLogs.length} logs in the ${trace_window_seconds}s before log ${error_log_id}. Levels: ${JSON.stringify(levelCounts)}.`,
              example_log_ids: traceLogs.slice(-5).map(l => l.id)
          };
      }
        
      case 'suggest_solution': {
          if (!aiInstance) return { summary: 'Local AI cannot suggest solution using Cloud Tools. Please answer based on your knowledge.' };
          const solutionPrompt = `Based on the following error message, act as a senior software engineer and provide a concise, actionable list of potential causes and solutions. Error: "${args.error_message}"`;
          try {
              const result = await aiInstance.models.generateContent({ model: 'gemini-2.5-flash', contents: [{ role: 'user', parts: [{ text: solutionPrompt }] }] });
              const text = result.candidates?.[0]?.content?.parts?.map(p => p.text).join('') || "Could not generate a solution.";
              return { solution: text };
          } catch (e: any) {
              return { solution: `An error occurred while generating a solution: ${e.message}` };
          }
      }

      default:
        return { error: `Tool "${toolName}" not found.` };
    }
  }, [allLogs, onScrollToLog, onUpdateFilters]);

  const runCloudAI = useCallback(async (prompt: string, effectiveModel: string) => {
    const apiKey = userApiKey || import.meta.env.VITE_API_KEY;
    if (!apiKey) {
      lastPromptRef.current = prompt;
      addMessage('model', "API key is not configured. Please set one in the settings (⚙️) or get one from [Google AI Studio](https://aistudio.google.com/api-keys).", true);
      setIsLoading(false);
      return;
    }
    
    if (!cloudPrivacyWarningShown.current) {
        addMessage('model', "You are using a cloud-based AI model. A summary of your log data will be sent to Google for analysis. For fully private, on-device analysis, you can switch to a local model.", false, true);
        cloudPrivacyWarningShown.current = true;
    }
    
    const ai = new GoogleGenAI({ apiKey });

    let systemPrompt = `You are an expert AI assistant embedded in a log analysis tool. Your primary goal is to help users understand their logs and identify problems.
# CONTEXT
- Total logs across all files: ${allLogs.length.toLocaleString()}
- Available Daemons: ${allDaemons.join(', ') || 'N/A'}`;

    if (effectiveModel === 'gemini-2.5-pro') {
        systemPrompt += `\n\nIMPORTANT: You are running on a model with strict rate limits. Try to answer the user's question IMMEDIATELY using the provided context. Do NOT call tools like 'search_logs' unless the local context is completely irrelevant or empty.`;
    }

    const history: Content[] = messages.slice(1).reduce((acc: Content[], m) => {
        if (m.isError || m.isWarning) return acc;
        if (m.role === 'model' && (m.text.startsWith('Tool Call:') || m.text.startsWith('Tool Response:'))) {
          // Skip these messages as they are for UI display only
        } else {
            acc.push({ role: m.role, parts: [{ text: m.text }] });
        }
        return acc;
    }, []);
    
    history.push({ role: 'user', parts: [{ text: prompt }] });
    
    const historyChars = JSON.stringify(history).length;
    const payloadConfig = { 
        systemInstruction: systemPrompt, 
        tools: [{ functionDeclarations: getAvailableTools(conversationStateRef.current) }] 
    };

    console.groupCollapsed(`[AI] Calling Gemini with ~${(historyChars / 4).toFixed(0)} tokens`);
    console.log(`[AI] Model: ${effectiveModel}`);
    console.log('[AI] Full Payload:', { contents: history, config: payloadConfig });
    console.groupEnd();
    
    let pendingFilterAction: FilterAction | null = null;
    const MAX_TURNS = 10;
    
    for (let turn = 1; turn <= MAX_TURNS; turn++) {
        console.groupCollapsed(`[AI] Turn ${turn}/${MAX_TURNS}`);
        console.log(`[AI] State: ${conversationStateRef.current}`);

        let response: GenerateContentResponse;
        try {
            const result = await ai.models.generateContent({ 
                model: effectiveModel, 
                contents: history, 
                config: payloadConfig
            });
            
            const responseTextCandidate = result.candidates?.[0]?.content?.parts?.map(p => p.text).join('') || undefined;
            
            response = { text: responseTextCandidate, functionCalls: result.functionCalls as FunctionCall[], candidates: result.candidates };
            
            const responseText = response.text || "";
            const responseChars = responseText.length + JSON.stringify(response.functionCalls || {}).length;
            console.log(`[AI] Received response from Gemini (~${(responseChars / 4).toFixed(0)} tokens)`);
            console.log('[AI] Full Response:', result);

        } catch (e: any) {
            console.error("AI Error:", e);
            let errorMessage = `An error occurred: ${e.message || 'Unknown error'}`;
            try {
                const errorBody = JSON.parse(e.message.replace('ApiError: ', ''));
                if (errorBody.error.details) {
                    const retryInfo = errorBody.error.details.find((d: any) => d['@type'] === 'type.googleapis.com/google.rpc.RetryInfo');
                    if (retryInfo?.retryDelay) {
                        const seconds = Math.ceil(parseFloat(retryInfo.retryDelay));
                        errorMessage = `Rate limit exceeded. Please try again in about ${seconds} seconds.`;
                    }
                    const usageLink = errorBody.error.message.match(/https?:\/\/ai\.dev\/usage\?tab=rate-limit/);
                    if (usageLink) {
                        errorMessage += `\n[Monitor your usage here](${usageLink[0]})`;
                    }
                }
            } catch {}
            addMessage('model', errorMessage, true);
            setIsLoading(false);
            console.groupEnd(); // End turn group on error
            return;
        }

        const functionCalls = response.functionCalls;
        if (functionCalls && functionCalls.length > 0) {
            const toolCall = functionCalls[0]; // Assuming one tool call for now

            if (!toolCall.name) {
                 console.error("AI returned a tool call without a name.");
                 addMessage('model', "Error: AI returned a tool call without a name.", true);
                 break;
            }

            // Capture potential actions to show to the user
            if (toolCall.name === 'search_logs') {
                pendingFilterAction = {
                    type: 'apply_filter',
                    label: 'Apply Search Filters',
                    payload: {
                        keywordQueries: toolCall.args.keywords || [], // Default empty array for action payload
                        keywordMatchMode: toolCall.args.match_mode || 'OR'
                    }
                };
            } else if (toolCall.name === 'update_filters') {
                if (!toolCall.args.apply_immediately) {
                    const labels = [];
                    if (toolCall.args.log_levels?.length) labels.push(toolCall.args.log_levels.join('|'));
                    if (toolCall.args.daemons?.length) labels.push(toolCall.args.daemons.join('|'));
                    if (toolCall.args.search_keywords?.length) labels.push(`"${toolCall.args.search_keywords.join(' ')}"`);
                    
                    pendingFilterAction = {
                        type: 'apply_filter',
                        label: labels.length > 0 ? `Filter: ${labels.join(', ')}` : 'Apply Suggested Filters',
                        payload: {
                            selectedLevels: toolCall.args.log_levels || [],
                            selectedDaemons: toolCall.args.daemons || [],
                            keywordQueries: toolCall.args.search_keywords || [],
                            keywordMatchMode: toolCall.args.keyword_match_mode || 'OR',
                        }
                    };
                }
            } else if (toolCall.name === 'trace_error_origin') {
                 const logId = toolCall.args.error_log_id;
                 const log = allLogs.find(l => l.id === logId);
                 if (log) {
                     const end = log.timestamp;
                     const start = new Date(end.getTime() - (toolCall.args.trace_window_seconds || 60) * 1000);
                     pendingFilterAction = {
                        type: 'apply_filter',
                        label: `Isolate Trace (${toolCall.args.trace_window_seconds || 60}s)`,
                        payload: {
                            dateRange: [start, end]
                        }
                    };
                 }
            }

            // Cast local FunctionCall to unknown, then to any to satisfy the strict Part type requirements of the SDK,
            // or simply ensure property alignment. The SDK's Part type usually accepts an object with `functionCall`.
            history.push({ role: 'model', parts: [{ functionCall: toolCall } as any] });
            
            console.groupCollapsed(`[AI] Executing tool: ${toolCall.name}`);
            console.log('[AI] Arguments:', toolCall.args);
            const toolResult = await handleToolCall(toolCall.name, toolCall.args || {}, ai);
            const toolResultChars = JSON.stringify(toolResult).length;
            console.log(`[AI] Tool responded with ~${(toolResultChars / 4).toFixed(0)} tokens.`);
            console.log('[AI] Tool Result:', toolResult);
            console.groupEnd();

            if (toolCall.name === 'search_logs' && toolResult?.example_log_ids?.length > 0) {
                conversationStateRef.current = 'ANALYZING';
                console.log('[AI State] Transitioning to ANALYZING.');
            }
            history.push({ role: 'tool', parts: [{ functionResponse: { name: toolCall.name, response: { result: JSON.stringify(toolResult) } } }] } as unknown as Content);
        } else {
            const text = response.text || "I'm sorry, I couldn't generate a response.";
            console.log('[AI] Model returned final answer.');
            addMessage('model', text, false, false, pendingFilterAction || undefined);
            conversationStateRef.current = 'IDLE';
            console.groupEnd(); // End turn group
            break;
        }
        console.groupEnd(); // End turn group
    }
    setIsLoading(false);
  }, [userApiKey, allLogs, allDaemons, messages, addMessage, handleToolCall, savedFindings]);
  
  const mlcEngine = useRef<any>(null);

  const executeLocalAI = useCallback(async (initialPrompt: string) => {
    if (!mlcEngine.current) return;
    setIsLoading(true);
    setDownloadStatus(null);

    // Limit daemons context to top 50 to improve prefill speed
    const limitedDaemons = allDaemons.slice(0, 50);
    const daemonContextStr = limitedDaemons.length < allDaemons.length 
        ? `${limitedDaemons.join(', ')}... (+${allDaemons.length - 50} more)` 
        : limitedDaemons.join(', ');

    const localToolsSchema = [
        {
            name: "scroll_to_log",
            description: "Scroll the viewer to a specific log entry.",
            parameters: {
                type: "object",
                properties: { log_id: { type: "number", description: "The ID of the log" } },
                required: ["log_id"]
            }
        },
        {
            name: "update_filters",
            description: "Updates log filters. Set apply_immediately=true for user commands (e.g., 'show error logs'). Set apply_immediately=false if suggesting a filter.",
            parameters: {
                type: "object",
                properties: {
                    daemons: { type: "array", items: { type: "string" } },
                    log_levels: { type: "array", items: { type: "string" } },
                    search_keywords: { type: "array", items: { type: "string" } },
                    apply_immediately: { type: "boolean" }
                }
            }
        },
        {
            name: "search_logs",
            description: "Search all logs for keywords.",
            parameters: {
                type: "object",
                properties: {
                    keywords: { type: "array", items: { type: "string" } },
                    match_mode: { type: "string", enum: ["AND", "OR"] }
                },
                required: ["keywords"]
            }
        },
        {
            name: "find_log_patterns",
            description: "Analyze patterns in the logs.",
            parameters: {
                type: "object",
                properties: {
                    pattern_type: { type: "string", enum: ["repeating_error", "frequency_spike"] },
                    time_window_minutes: { type: "number" }
                },
                required: ["pattern_type"]
            }
        }
    ];

    const toolInstructions = `
# TOOLS
You can control the UI. You must answer the user's question, but if you need to perform an action, you can call a tool.
To call a tool, you MUST use the following format exactly:

<<<TOOL>>>
{
  "name": "tool_name",
  "args": { ... }
}
<<<END>>>

Available Tools (JSON Schema):
${JSON.stringify(localToolsSchema, null, 2)}
`;

    const systemPrompt = `You are a helpful AI assistant embedded in a log analysis tool. Analyze the provided information and answer the user's questions concisely.
# CONTEXT
- Total logs: ${allLogs.length.toLocaleString()}
- Available Daemons: ${daemonContextStr || 'N/A'}
${toolInstructions}`;

    let history = messages.slice(-6).reduce((acc: any[], m) => {
        if (!m.isError && !m.isWarning && (m.role !== 'model' || !m.text.startsWith('Tool'))) {
            const role = m.role === 'model' ? 'assistant' : m.role;
            let cleanContent = m.text.replace(/<<<TOOL>>>[\s\S]*?<<<END>>>/g, '').trim();
            // Handle various tag formats in cleanup
            cleanContent = cleanContent.replace(/<<<[\w_]+>>>[\s\S]*?<<<END>>>?/g, '').trim();
            if (cleanContent) {
                acc.push({ role, content: cleanContent });
            }
        }
        return acc;
    }, []);

    history.push({ role: 'user', content: initialPrompt });
    
    let turn = 0;
    const MAX_TURNS = 10; 
    let pendingFilterAction: FilterAction | null = null;

    try {
        while (turn < MAX_TURNS) {
            turn++;
            const messagesPayload = [{ role: 'system', content: systemPrompt }, ...history];

            const chunks = await mlcEngine.current.chat.completions.create({
                messages: messagesPayload,
                stream: true,
                temperature: 0.7
            });

            let fullText = "";
            let uiText = ""; 
            let isFirstChunk = true;
            let hiddenBuffer = ""; 
            let isCollectingTool = false;
            
            const messageId = Date.now().toString() + Math.random();
            
            for await (const chunk of chunks) {
                const delta = chunk.choices[0]?.delta?.content || "";
                if (delta) {
                    fullText += delta;
                    
                    if (isCollectingTool) {
                        hiddenBuffer += delta;
                    } else {
                        // Check for start of tool call
                        // Check for standard tag or loose tag (e.g. <<<update_filters>>>)
                        const lookahead = (hiddenBuffer + delta);
                        // Improve detection for partial tag starts like "<<" or "<<<"
                        if (delta.includes('<') || delta.includes('{') || delta.includes(':')) {
                            // Heuristic: If we see <<< start, switch to collection mode
                            if (lookahead.match(/<<<[\w_]*$/) || lookahead.match(/{\s*"name"/) || lookahead.match(/<<<[\w_]+>>>/)) {
                                isCollectingTool = true;
                                hiddenBuffer += delta; 
                            } else {
                                if (!isCollectingTool) {
                                    uiText += delta;
                                }
                            }
                        } else {
                            if (!isCollectingTool) {
                                uiText += delta;
                            }
                        }
                    }

                    if (uiText || isFirstChunk) { 
                        if (isFirstChunk) {
                            setMessages(prev => [...prev, { id: messageId, role: 'model', text: uiText, action: pendingFilterAction || undefined }]);
                            isFirstChunk = false;
                            if (pendingFilterAction) pendingFilterAction = null; 
                        } else {
                            if (!isCollectingTool) {
                                setMessages(prev => prev.map(m => m.id === messageId ? { ...m, text: uiText } : m));
                            }
                        }
                    }
                }
            }

            // --- Post-Generation Parsing ---
            
            let jsonStr = "";
            let toolNameFromTag = "";

            // 1. Try standard tag match
            const standardMatch = fullText.match(/<<<TOOL>>>([\s\S]*?)<<<END>>>?/);
            if (standardMatch) {
                jsonStr = standardMatch[1];
            } else {
                // 2. Try tag-as-name match (e.g. <<<update_filters>>> { ... } <<<END>>>)
                const namedTagMatch = fullText.match(/<<<(\w+)>>>([\s\S]*?)<<<END>>>?/);
                if (namedTagMatch) {
                    toolNameFromTag = namedTagMatch[1];
                    jsonStr = namedTagMatch[2];
                } else {
                    // 3. Try raw JSON fallback
                    const jsonMatch = fullText.match(/({[\s\S]*"name"[\s\S]*})/);
                    if (jsonMatch) {
                        jsonStr = jsonMatch[1];
                    }
                }
            }
            
            // Shorthand fallback
            const shorthandMatch = fullText.match(/:::scroll_to_log\((\d+)\):::/);

            if (jsonStr) {
                try {
                    jsonStr = jsonStr.replace(/```json/g, '').replace(/```/g, '').trim();
                    let toolCall;
                    
                    if (toolNameFromTag) {
                        // If we got name from tag, the JSON is likely just the args
                        const parsed = JSON.parse(jsonStr);
                        // Sometimes model puts name in JSON too despite the tag, check for that
                        if (parsed.name && !parsed.args && parsed.name === toolNameFromTag) {
                             toolCall = parsed; // It was full tool object
                        } else if (parsed.name && parsed.args) {
                             toolCall = parsed; // It was full tool object, ignore tag mismatch if any
                        } else {
                             // It was just args
                             toolCall = { name: toolNameFromTag, args: parsed };
                        }
                    } else {
                        toolCall = JSON.parse(jsonStr);
                    }

                    // Defaults for specific tools
                    if (toolCall.name === 'update_filters') {
                        toolCall.args.log_levels = toolCall.args.log_levels || [];
                        toolCall.args.daemons = toolCall.args.daemons || [];
                        toolCall.args.search_keywords = toolCall.args.search_keywords || [];
                        
                        // Check if this is a suggestion vs command
                        if (!toolCall.args.apply_immediately) {
                            const labels = [];
                            if (toolCall.args.log_levels?.length) labels.push(toolCall.args.log_levels.join('|'));
                            if (toolCall.args.daemons?.length) labels.push(toolCall.args.daemons.join('|'));
                            if (toolCall.args.search_keywords?.length) labels.push(`"${toolCall.args.search_keywords.join(' ')}"`);

                            pendingFilterAction = {
                                type: 'apply_filter',
                                label: labels.length > 0 ? `Filter: ${labels.join(', ')}` : 'Apply Suggested Filters',
                                payload: {
                                    selectedLevels: toolCall.args.log_levels,
                                    selectedDaemons: toolCall.args.daemons,
                                    keywordQueries: toolCall.args.search_keywords,
                                    keywordMatchMode: toolCall.args.keyword_match_mode || 'OR',
                                }
                            };
                        }
                    }
                    if (toolCall.name === 'search_logs') {
                        toolCall.args.keywords = toolCall.args.keywords || [];
                        pendingFilterAction = {
                            type: 'apply_filter',
                            label: 'Apply Search Filters',
                            payload: {
                                keywordQueries: toolCall.args.keywords,
                                keywordMatchMode: toolCall.args.match_mode || 'OR'
                            }
                        };
                    }

                    // Update the message with the action AND clean up any leaked tool text
                    setMessages(prev => prev.map(m => m.id === messageId ? { 
                        ...m, 
                        // Strip out tool calls from display text to be safe
                        text: m.text.replace(/<<<.*?>>>[\s\S]*?<<<END>>>?/g, '').trim(),
                        action: pendingFilterAction || undefined 
                    } : m));

                    history.push({ role: 'assistant', content: fullText }); 

                    const result = await handleToolCall(toolCall.name, toolCall.args);
                    const resultStr = JSON.stringify(result);
                    
                    history.push({ role: 'user', content: `Tool Output: ${resultStr}` });
                    continue; // Loop for next turn

                } catch (e) {
                    console.error("Local Tool Parse Error", e);
                    break;
                }
            } else if (shorthandMatch) {
                const logId = parseInt(shorthandMatch[1], 10);
                history.push({ role: 'assistant', content: fullText });
                const result = await handleToolCall('scroll_to_log', { log_id: logId });
                history.push({ role: 'user', content: `Tool Output: ${JSON.stringify(result)}` });
                continue;
            } else {
                break;
            }
        }
    } catch (e: any) {
        console.error("Local AI Execution Error:", e);
        addMessage('model', `Local AI Error: ${e.message}`, true);
    } finally {
        setIsLoading(false);
    }
  }, [messages, allLogs, allDaemons, addMessage, handleToolCall]);

  const loadWebLlm = useCallback(async () => {
    setIsLoading(true);
    setShowWebLlmConsent(false);
    try {
        addMessage('model', 'Initializing local model (Llama 3). This requires downloading ~2.5GB of data to your browser cache. This happens only once.', false, true);
        
        const engine = await CreateMLCEngine(WEB_LLM_MODEL_ID, {
            initProgressCallback: (report) => {
                setDownloadStatus({ text: report.text, progress: report.progress });
            }
        });
        
        mlcEngine.current = engine;
        localStorage.setItem(WEBLMM_CONSENT_KEY, 'true');
        addMessage('model', 'Local model loaded! Processing your request...');
        setDownloadStatus(null); // Clear progress text after load
        
        if (pendingPrompt) {
            executeLocalAI(pendingPrompt);
            setPendingPrompt(null);
        } else {
            setIsLoading(false);
        }
    } catch (e: any) {
        console.error("WebLLM Load Error:", e);
        addMessage('model', `Failed to load local model: ${e.message}`, true);
        setIsLoading(false);
        setDownloadStatus(null);
        setPendingPrompt(null);
    }
  }, [pendingPrompt, addMessage, executeLocalAI]);

  const runLocalAI = useCallback(async (prompt: string) => {
    if (mlcEngine.current) {
        executeLocalAI(prompt);
        return;
    }

    const hasConsented = localStorage.getItem(WEBLMM_CONSENT_KEY);
    setPendingPrompt(prompt);
    
    if (hasConsented === 'true') {
        loadWebLlm();
    } else {
        setShowWebLlmConsent(true);
    }
  }, [executeLocalAI, loadWebLlm]);

  const handleConsent = (consented: boolean) => {
      setShowWebLlmConsent(false);
      if (consented) {
          loadWebLlm();
      } else {
          setPendingPrompt(null);
          setIsLoading(false);
          setModelTier('gemini-2.5-flash'); // Fallback to default
          addMessage('model', 'Switched back to Gemini 2.5 Flash.');
      }
  };
  
  const runChromeBuiltInAI = useCallback(async (prompt: string) => {
    if (!window.ai?.languageModel) {
        addMessage('model', 
            `**Chrome built-in AI is not available.**
            
To use this feature, you must:
1. Use Chrome Canary or Dev channel (v127 or higher).
2. Go to \`chrome://flags\`.
3. Enable **"Enforce on-device model inclusion"**.
4. Enable **"Prompt API for Gemini Nano"**.
5. Relaunch Chrome.`, 
            true
        );
        setIsLoading(false);
        return;
    }

    try {
        const capabilities = await window.ai.languageModel.capabilities();
        if (capabilities.available === 'no') {
             addMessage('model', '**Chrome AI model is not downloaded.** Please verify your browser configuration in `chrome://flags` or check internet connection.', true);
             setIsLoading(false);
             return;
        }

        if (!chromeAiSession.current) {
            console.log('[AI] Creating new Chrome AI session.');
            const systemPrompt = `You are a helpful AI assistant embedded in a log analysis tool. Analyze the provided information and answer the user's questions concisely. You do not have tools to search or filter logs.
# CONTEXT
- Total logs across all files: ${allLogs.length.toLocaleString()}
- Available Daemons: ${allDaemons.join(', ') || 'N/A'}`;
            chromeAiSession.current = await window.ai.languageModel.create({ systemPrompt });
        }
        
        const payloadSize = prompt.length;
        console.groupCollapsed(`[AI] Calling Chrome Built-in AI with ~${(payloadSize / 4).toFixed(0)} tokens`);
        console.log('[AI] Prompt:', prompt);
        console.groupEnd();

        const response = await chromeAiSession.current.prompt(prompt);
        
        const responseSize = response.length;
        console.groupCollapsed(`[AI] Received response from Chrome AI with ~${(responseSize / 4).toFixed(0)} tokens`);
        console.log('[AI] Response:', response);
        console.groupEnd();
        
        addMessage('model', response);
    } catch (e: any) {
        console.error("Chrome AI Error:", e);
        addMessage('model', `An error occurred with the Chrome AI: ${e.message}`, true);
        if (chromeAiSession.current) {
            chromeAiSession.current.destroy();
            chromeAiSession.current = null;
        }
    } finally {
        setIsLoading(false);
    }
  }, [addMessage, allLogs, allDaemons]);

  const getEffectiveModelTierAndRun = useCallback((prompt: string) => {
    if (modelTier === 'chrome-built-in') {
        runChromeBuiltInAI(prompt);
        return;
    }
    if (modelTier === 'web-llm') {
        runLocalAI(prompt);
        return;
    }

    const now = Date.now();
    const oneMinuteAgo = now - 60000;

    // Clean up old timestamps
    Object.keys(apiRequestTimestampsRef.current).forEach(model => {
        apiRequestTimestampsRef.current[model] = apiRequestTimestampsRef.current[model].filter(ts => ts > oneMinuteAgo);
    });

    const getRequestCount = (model: string) => apiRequestTimestampsRef.current[model]?.length || 0;
    
    let effectiveModel = modelTier;
    let fallbackMessage = '';

    const tiers: (keyof typeof MODEL_CONFIG)[] = ['gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-flash-lite-latest'];
    const currentTierIndex = tiers.indexOf(modelTier as any);

    if (currentTierIndex !== -1) {
        for (let i = currentTierIndex; i < tiers.length; i++) {
            const tier = tiers[i];
            const limit = MODEL_CONFIG[tier].rpm;
            const count = getRequestCount(tier);
            
            console.log(`[Rate Governor] Checking ${tier}: ${count} requests / ${limit} RPM limit.`);

            if (count < limit) {
                effectiveModel = tier;
                if (tier !== modelTier) {
                    fallbackMessage = `**Notice:** The '${MODEL_CONFIG[modelTier as keyof typeof MODEL_CONFIG].name}' model is busy. Using '${MODEL_CONFIG[effectiveModel as keyof typeof MODEL_CONFIG].name}' for this request.`;
                }
                break;
            }
            if (i === tiers.length - 1) { // Last tier is also busy
                addMessage('model', "All AI models are currently busy due to rate limits. Please wait a moment before trying again.", true);
                setIsLoading(false);
                return;
            }
        }
    }

    if (fallbackMessage) {
        addMessage('model', fallbackMessage, false, true);
    }
    
    // Log the request
    if (!apiRequestTimestampsRef.current[effectiveModel]) {
        apiRequestTimestampsRef.current[effectiveModel] = [];
    }
    apiRequestTimestampsRef.current[effectiveModel].push(now);

    runCloudAI(prompt, effectiveModel);
  }, [modelTier, addMessage, runChromeBuiltInAI, runLocalAI, runCloudAI]);

  // SMART LOCAL SEARCH WITH PATTERN GROUPING AND OPTIMIZATION
  const enhancePromptWithLocalContext = useCallback(async (prompt: string): Promise<string> => {
      const stopWords = new Set([
        'a', 'an', 'the', 'is', 'are', 'what', 'show', 'me', 'find', 'list', 'of', 'in', 'for', 'all', 'with', 'and', 'or', 'about', 
        'when', 'did', 'started', 'start', 'to', 'do', 'does', 'why', 'how', 'where', 'can', 'you', 'i', 'my', 'please', 'logs', 'log'
      ]);
      const extractedKeywords = prompt.toLowerCase()
          // Retain word chars, whitespace, dots, hyphens (e.g. 192.168.1.1, my-daemon)
          .replace(/[^\w\s\.-]/g, '') 
          .split(/\s+/)
          .filter(word => !stopWords.has(word) && word.length > 2);
      
      // If we have no significant keywords or no logs, skip optimization
      if (extractedKeywords.length === 0 || allLogs.length === 0) {
          return prompt;
      }

      // --- Filter by Index (Structural Search) ---
      let candidateIndices: number[] | null = null;
      const structuralKeywords = new Set<string>();
      const levelKeys = Object.keys(logIndex.levels);

      // Helper to find intersection of sorted arrays (indices are naturally sorted in logIndex)
      const intersectSorted = (a: number[], b: number[]) => {
          const res = [];
          let i = 0, j = 0;
          while (i < a.length && j < b.length) {
              if (a[i] < b[j]) i++;
              else if (a[i] > b[j]) j++;
              else {
                  res.push(a[i]);
                  i++; j++;
              }
          }
          return res;
      };

      for (const kw of extractedKeywords) {
          // Check for Level match (e.g. "error", "warnings")
          const levelMatch = levelKeys.find(k => k.toLowerCase() === kw || k.toLowerCase() + 's' === kw);
          if (levelMatch) {
              structuralKeywords.add(kw);
              const indices = logIndex.levels[levelMatch];
              candidateIndices = candidateIndices ? intersectSorted(candidateIndices, indices) : indices;
              continue;
          }
          // Check for Daemon match
          if (logIndex.daemons[kw]) {
              structuralKeywords.add(kw);
              const indices = logIndex.daemons[kw];
              candidateIndices = candidateIndices ? intersectSorted(candidateIndices, indices) : indices;
              continue;
          }
      }

      // If we found structural matches but they intersected to zero, no logs match.
      if (candidateIndices !== null && candidateIndices.length === 0) {
          return prompt;
      }

      const contentKeywords = extractedKeywords.filter(k => !structuralKeywords.has(k));
      const hasLevelKeyword = extractedKeywords.some(kw => Object.keys(logIndex.levels).some(l => l.toLowerCase() === kw));
      
      // --- Timestamp Optimization (Binary Search) ---
      let startLogIdx = 0;
      let endLogIdx = allLogs.length;

      const dateMatch = prompt.match(/\b(\d{4}-\d{2}-\d{2})\b/);
      if (dateMatch) {
          const d = new Date(dateMatch[1]);
          if (!isNaN(d.getTime())) {
             startLogIdx = findLogStartIndex(allLogs, d.getTime());
             const nextDay = new Date(d);
             nextDay.setDate(nextDay.getDate() + 1);
             endLogIdx = findLogStartIndex(allLogs, nextDay.getTime());
          }
      }

      // Determine scan boundaries
      // If we have structural indices, we intersect them with the time range.
      // If not, we just scan the time range.
      let loopStart = 0;
      let loopEnd = 0;
      let useCandidates = false;

      if (candidateIndices) {
           useCandidates = true;
           // Filter sorted candidate indices to be within [startLogIdx, endLogIdx)
           // We use binary search on the indices array itself to find the subset quickly.
           loopStart = lowerBound(candidateIndices, startLogIdx);
           loopEnd = lowerBound(candidateIndices, endLogIdx);
      } else {
           loopStart = startLogIdx;
           loopEnd = endLogIdx;
      }
      
      const scanCount = loopEnd - loopStart;
      const shouldScan = (scanCount > 0) && (contentKeywords.length > 0 || useCandidates || startLogIdx > 0 || endLogIdx < allLogs.length);
      
      if (!shouldScan) {
           return prompt;
      }

      // Combined Scanning and Grouping Loop
      const groupedLogs = new Map<string, { 
          pattern: string; 
          level: LogLevel; 
          daemon: string; 
          count: number; 
          score: number; 
          examples: number[] 
      }>();
      
      let matchCount = 0;
      const CHUNK_SIZE = 5000; 

      // Pre-calculate keyword types for weighting inside the hot loop
      const keywordWeights = contentKeywords.map(kw => {
          let weight = 1; // Base score
          // If keyword matches a known daemon name (but wasn't used as a hard filter), boost it
          if (logIndex.daemons[kw]) weight = 3; 
          return { kw, weight };
      });

      for (let i = loopStart; i < loopEnd; i += CHUNK_SIZE) {
          const chunkEnd = Math.min(i + CHUNK_SIZE, loopEnd);
          
          for (let j = i; j < chunkEnd; j++) {
              const idx = useCandidates && candidateIndices ? candidateIndices[j] : j;
              const log = allLogs[idx];
              let score = 0;
              
              if (contentKeywords.length > 0) {
                  const msgLower = log.message.toLowerCase();
                  const daemonLower = log.daemon.toLowerCase();
                  
                  for (const { kw, weight } of keywordWeights) {
                      if (msgLower.includes(kw) || daemonLower.includes(kw) || log.level.toLowerCase().includes(kw)) {
                          score += weight;
                      }
                  }
              } else {
                  // No content keywords, but matched structural filter (e.g. "show errors")
                  score = 1; 
              }

              // Severity Boost: If user didn't explicitly ask for a specific level, boost errors/criticals
              // This ensures "what's wrong?" bubbles up errors even if query is neutral.
              if (score > 0 && !hasLevelKeyword) {
                  if (log.level === LogLevel.ERROR || log.level === LogLevel.CRITICAL) score += 0.5;
                  else if (log.level === LogLevel.WARNING) score += 0.1;
              }
              
              if (score > 0) {
                  matchCount++;
                  const pattern = getLogPattern(log.message);
                  // Key by Pattern + Daemon + Level to differentiate similar messages from different sources
                  const key = `${log.level}|${log.daemon}|${pattern}`;
                  
                  if (!groupedLogs.has(key)) {
                      groupedLogs.set(key, {
                          pattern,
                          level: log.level,
                          daemon: log.daemon,
                          count: 0,
                          score: 0,
                          examples: []
                      });
                  }
                  
                  const group = groupedLogs.get(key)!;
                  group.count++;
                  group.score = Math.max(group.score, score);
                  
                  if (group.examples.length < 3) {
                      group.examples.push(log.id);
                  }
              }
          }
          // Yield to event loop if processing a large set
          if (scanCount > CHUNK_SIZE) await new Promise(r => setTimeout(r, 0));
      }

      if (matchCount === 0) {
           return prompt;
      }

      // Sort groups by Relevance first, then heavily bias towards RARITY.
      // High frequency logs are usually noise. Low frequency logs are usually interesting.
      const groupsArray = Array.from(groupedLogs.values());
      
      // Apply Rarity Boost
      groupsArray.forEach(g => {
          if (g.count === 1) g.score += 3.0;
          else if (g.count < 5) g.score += 2.0;
          else if (g.count < 20) g.score += 1.0;
          else if (g.count > 1000) g.score -= 1.0; // Penalty for spam
      });

      const sortedGroups = groupsArray.sort((a, b) => {
          // Strict score sorting (which now includes rarity bias)
          return b.score - a.score;
      });

      // Select top groups to fit in context (increased from 20 to 30 to include more variety)
      const topGroups = sortedGroups.slice(0, 30);
      
      const contextData = topGroups.map(g => 
          `[Count: ${g.count}] [${g.level}] [${g.daemon}] Pattern: "${g.pattern}" (Example IDs: ${g.examples.map(id => `[Log ID: ${id}]`).join(', ')})`
      ).join('\n');

      const systemNote = `
[SYSTEM CONTEXT - LOCAL SEARCH RESULTS]
The user's query matched ${matchCount} local logs. Here are the most relevant log groups found, prioritized by relevance and uniqueness (rarity):

${contextData}

[INSTRUCTION]
Use the data above to answer the user's question directly. 
- Do NOT explicitly mention "I found log patterns" or "local search results" unless the user asks how you know.
- Use the "Example IDs" to provide citations (e.g. [Log ID: 123]).
- If this data is sufficient, answer the question. If not, use tools like 'search_logs' for a broader search.
`;
      
      return `${systemNote}\n\nUser Question: "${prompt}"`;
  }, [allLogs, addMessage, logIndex]);

  const handleSubmit = useCallback(async (e?: React.FormEvent, overridePrompt?: string) => {
    e?.preventDefault();
    const trimmedInput = overridePrompt || input.trim();
    if (!trimmedInput || isLoading) return;

    addMessage('user', trimmedInput);
    setIsLoading(true);
    conversationStateRef.current = 'IDLE'; // Reset state for new prompt

    // Perform smart local context enhancement unless disabled
    let enhancedPrompt = trimmedInput;
    if (!disableLocalSearch) {
        enhancedPrompt = await enhancePromptWithLocalContext(trimmedInput);
    } else {
        console.log('[AI] Local search optimization disabled by user. Sending raw prompt.');
    }
    
    getEffectiveModelTierAndRun(enhancedPrompt);

    setInput('');
    if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
        textareaRef.current.style.overflowY = 'hidden';
    }
  }, [input, isLoading, addMessage, getEffectiveModelTierAndRun, enhancePromptWithLocalContext, disableLocalSearch]);

  const handleQuickAction = (prompt: string) => {
    setInput(prompt);
    // Use a timeout to ensure the state updates before submitting
    setTimeout(() => {
        handleSubmit(undefined, prompt);
    }, 50);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      setInput(e.target.value);
      const textarea = e.currentTarget;
      textarea.style.height = 'auto'; // Reset height to recalculate based on content
      const scrollHeight = textarea.scrollHeight;
      const maxHeight = 200; // Approx 8-9 lines

      if (scrollHeight > maxHeight) {
          textarea.style.height = `${maxHeight}px`;
          textarea.style.overflowY = 'auto';
      } else {
          textarea.style.height = `${scrollHeight}px`;
          textarea.style.overflowY = 'hidden';
      }
  };
  
  return (
    <div className="h-full flex flex-col bg-gray-800 relative">
        {/* Header */}
        <div className="flex-shrink-0 flex items-center justify-between p-2 border-b border-gray-700">
          <div className="flex items-center space-x-2">
            <svg className="w-4 h-4 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
            <h2 className="font-bold text-sm text-white">AI Assistant</h2>
            <select value={modelTier} onChange={e => setModelTier(e.target.value)} className="bg-gray-700 text-white text-xs rounded py-0.5 px-1 border border-gray-600 focus:ring-1 focus:ring-blue-500 focus:outline-none">
                <option value="gemini-flash-lite-latest">Fast</option>
                <option value="gemini-2.5-flash">Balanced</option>
                <option value="gemini-2.5-pro">Reasoning</option>
                <option value="chrome-built-in">Local (Chrome)</option>
                <option value="web-llm">Local (WebLLM)</option>
            </select>
          </div>
          <div className="flex items-center space-x-1">
            <button onClick={() => setIsSettingsOpen(!isSettingsOpen)} className="p-1 text-gray-400 hover:text-white rounded-md hover:bg-gray-700" title="Settings"><svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /></svg></button>
            <button onClick={() => setMessages([messages[0]])} className="p-1 text-gray-400 hover:text-white rounded-md hover:bg-gray-700" title="Reset Chat"><svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg></button>
            <button onClick={onClose} className="p-1 text-gray-400 hover:text-white rounded-md hover:bg-gray-700" aria-label="Close AI Assistant"><svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" /></svg></button>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="flex-shrink-0 p-2 border-b border-gray-700 grid grid-cols-2 gap-2">
            <button onClick={() => handleQuickAction("Summarize the key events by searching the entire log file.")} disabled={isLoading} className="p-2 bg-gray-700/50 hover:bg-gray-700 rounded-md text-xs text-gray-300 transition-colors disabled:opacity-50">Summarize View</button>
            <button onClick={() => handleQuickAction("Find all errors in the logs and summarize them.")} disabled={isLoading} className="p-2 bg-gray-700/50 hover:bg-gray-700 rounded-md text-xs text-gray-300 transition-colors disabled:opacity-50">Analyze Errors</button>
            <button onClick={() => handleQuickAction("Find the most critical error and suggest a solution.")} disabled={isLoading} className="p-2 bg-gray-700/50 hover:bg-gray-700 rounded-md text-xs text-gray-300 transition-colors disabled:opacity-50">Suggest Solution</button>
            <button onClick={() => handleQuickAction("Explain your capabilities and provide examples of what I can ask.")} disabled={isLoading} className="p-2 bg-gray-700/50 hover:bg-gray-700 rounded-md text-xs text-gray-300 transition-colors disabled:opacity-50">Capabilities</button>
        </div>

        <div className="flex-grow p-3 overflow-y-auto space-y-4">
          {messages.map((message) => (
            <div key={message.id} className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`relative max-w-[85%] p-2 rounded-lg text-white ${message.role === 'user' ? 'bg-blue-600' : (message.isError ? 'bg-red-800' : (message.isWarning ? 'bg-yellow-800/80' : 'bg-gray-700'))}`}>
                 {message.role === 'model' && !message.isError && !message.isWarning && savedFindings.includes(message.text) ? (
                    <div className="absolute top-0 right-0 flex -translate-y-1/2 translate-x-1/2" title="Finding Saved"><div className="p-0.5 rounded-full bg-green-600 text-white"><svg className="w-2 h-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="3" d="M5 13l4 4L19 7"></path></svg></div></div>
                 ) : (message.role === 'model' && !message.isError && !message.isWarning &&
                    <button onClick={() => onSaveFinding(message.text)} className="absolute top-0 right-0 flex -translate-y-1/2 translate-x-1/2 p-0.5 rounded-full text-gray-400 bg-gray-800 border border-gray-600 hover:text-white hover:bg-gray-600" title="Save this finding"><svg className="w-2 h-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z"></path></svg></button>
                 )}
                 <FormattedMessage text={message.text} onScrollToLog={onScrollToLog} />
                 
                 {message.action && (
                    <div className="mt-2 pt-2 border-t border-gray-600/50">
                        <button 
                            onClick={() => onUpdateFilters(message.action!.payload, true)}
                            className="flex items-center space-x-1 text-xs bg-blue-600 hover:bg-blue-500 text-white px-2 py-1 rounded transition-colors w-full justify-center"
                        >
                            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z"></path></svg>
                            <span>{message.action.label}</span>
                        </button>
                    </div>
                 )}
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="flex justify-start">
                <div className="max-w-[85%] p-2 rounded-lg bg-gray-700 text-white min-w-[60px]">
                    <div className="flex items-center space-x-2 text-xs mb-1">
                        <div className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-pulse" style={{ animationDelay: '0s' }}></div>
                        <div className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-pulse" style={{ animationDelay: '0.2s' }}></div>
                        <div className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-pulse" style={{ animationDelay: '0.4s' }}></div>
                    </div>
                    {downloadStatus && (
                         <div className="w-full">
                             <div className="text-[9px] text-gray-400 mb-1 truncate">{downloadStatus.text}</div>
                             <div className="w-full bg-gray-600 rounded-full h-1">
                                <div 
                                    className="bg-blue-500 h-1 rounded-full transition-all duration-300 ease-out" 
                                    style={{ width: `${Math.round(downloadStatus.progress * 100)}%` }}
                                ></div>
                             </div>
                         </div>
                    )}
                </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
        
        <div className="flex-shrink-0 p-2 border-t border-gray-700 bg-gray-800">
            <form onSubmit={handleSubmit} className="flex items-center space-x-2">
              <textarea
                ref={textareaRef}
                value={input}
                onChange={handleInputChange}
                onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSubmit(e); }}}
                placeholder="Ask about your logs..."
                disabled={isLoading}
                rows={1}
                className="flex-grow bg-gray-700 border border-gray-600 text-white text-xs rounded-md shadow-sm p-2 focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-800 resize-none overflow-hidden"
              />
              <button
                type="submit"
                disabled={isLoading || !input.trim()}
                className="bg-blue-600 text-white p-2 rounded-md hover:bg-blue-700 transition-colors disabled:bg-gray-600 disabled:cursor-not-allowed"
                aria-label="Send message"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 10l7-7m0 0l7 7m-7-7v18"></path></svg>
              </button>
            </form>
        </div>

        {isSettingsOpen && (
            <div className="absolute inset-0 bg-black/60 z-10 flex items-center justify-center p-4">
                <div className="bg-gray-900 rounded-lg shadow-xl p-4 border border-gray-700 w-full max-w-sm space-y-4">
                    <div>
                        <h3 className="font-semibold text-gray-200 mb-2">API Key Settings</h3>
                        <label htmlFor="api-key-input" className="text-xs text-gray-400 block mb-1">Google AI API Key (Optional)</label>
                        <input id="api-key-input" type="password" value={tempApiKey} onChange={(e) => setTempApiKey(e.target.value)} placeholder="Enter key to override system default" className="w-full bg-gray-700 text-white rounded py-1 px-2 border border-gray-600 focus:ring-1 focus:ring-blue-500 focus:outline-none text-xs"/>
                        <p className="text-[10px] text-gray-500 mt-1">Get a key from <a href="https://aistudio.google.com/api-keys" target="_blank" rel="noopener noreferrer" className="text-blue-400 underline">Google AI Studio</a>. Your key is stored in your browser's local storage.</p>
                    </div>

                    <div className="pt-2 border-t border-gray-700">
                        <h3 className="font-semibold text-gray-200 mb-2">Performance Settings</h3>
                        <label className="flex items-start space-x-2 cursor-pointer">
                            <input 
                                type="checkbox" 
                                checked={tempDisableLocalSearch} 
                                onChange={(e) => setTempDisableLocalSearch(e.target.checked)} 
                                className="mt-0.5 rounded bg-gray-700 border-gray-600 text-blue-600 focus:ring-blue-500"
                            />
                            <div>
                                <span className="text-xs text-gray-300 font-medium">Disable Local Search Pre-processing</span>
                                <p className="text-[10px] text-gray-500 mt-0.5 leading-tight">
                                    Skip the pre-processing step that scans logs locally. The AI will receive your raw prompt and use tools to search if needed. Useful for benchmarking or reducing local CPU usage.
                                </p>
                            </div>
                        </label>
                    </div>

                    <div className="flex justify-end space-x-2">
                        <button onClick={() => setIsSettingsOpen(false)} className="bg-gray-600 text-white px-3 py-1 rounded text-xs hover:bg-gray-700">Cancel</button>
                        <button onClick={handleSaveSettings} className="bg-blue-600 text-white px-3 py-1 rounded text-xs hover:bg-blue-700">Save</button>
                    </div>
                </div>
            </div>
        )}

        {showWebLlmConsent && (
            <div className="absolute inset-0 bg-black/80 z-20 flex items-center justify-center p-4">
                <div className="bg-gray-800 rounded-lg p-6 max-w-md w-full border border-gray-700 text-center">
                    <h3 className="text-xl font-bold text-white mb-2">Download Local AI Model?</h3>
                    <p className="text-sm text-gray-300 mb-4">
                        To run the AI locally on your device, we need to download the Llama 3 model weights (~2.5GB). 
                        This happens only once and is stored in your browser cache.
                    </p>
                    <div className="flex justify-center space-x-4">
                        <button 
                            onClick={() => handleConsent(false)}
                            className="px-4 py-2 rounded bg-gray-600 hover:bg-gray-500 text-white text-sm"
                        >
                            Cancel
                        </button>
                        <button 
                            onClick={() => handleConsent(true)}
                            className="px-4 py-2 rounded bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium"
                        >
                            Download & Run
                        </button>
                    </div>
                </div>
            </div>
        )}
    </div>
  );
};
