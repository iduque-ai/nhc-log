import React, { useState, useRef, useEffect, useCallback } from 'react';
import { GoogleGenAI, Type, FunctionDeclaration, Content, Part } from "@google/genai";
import { CreateMLCEngine } from "@mlc-ai/web-llm";
import { LogEntry, FilterState } from '../types.ts';

interface AIAssistantProps {
  isOpen: boolean;
  onClose: () => void;
  visibleLogs: LogEntry[];
  allLogs: LogEntry[];
  allDaemons: string[];
  onUpdateFilters: (filters: Partial<FilterState>, reset?: boolean) => void;
  onScrollToLog: (logId: number) => void;
}

interface Message {
  id: string;
  role: 'user' | 'model';
  text: string;
  isError?: boolean;
  isWarning?: boolean;
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

// --- Tool Definitions ---

const updateFiltersTool: FunctionDeclaration = {
  name: 'update_filters',
  description: 'Creates a NEW TAB with specific filters to isolate logs. Use this when the user asks to "show errors", "filter by daemon", or "isolate logs". This does NOT affect the current view, it opens a new one.',
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
      }
    },
  },
};

const scrollToLogTool: FunctionDeclaration = {
  name: 'scroll_to_log',
  description: 'Scroll the viewer to a specific log entry. Use this when you find a specific log ID from the search_logs tool and want to show it to the user.',
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
  description: 'Search ALL logs for specific information, including timestamps. You can provide multiple synonyms or related terms to broaden the search.',
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
        description: 'Maximum number of logs to return (default 50).',
      },
    },
    required: ['keywords'],
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
  description: 'Traces events leading up to a specific log entry to help find the root cause. It looks backwards in time from the given log ID.',
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
  description: 'Provides potential solutions or debugging steps for a given error message. This tool is for getting advice, not for searching logs.',
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

// --- Formatted Message Component ---

// Helper to recursively render inline markdown
const renderInlineMarkdown = (text: string, onScrollToLog: (id: number) => void): React.ReactNode => {
    // Split by [Log ID: 123] and markdown links [text](url)
    const parts = text.split(/(\[Log ID: \d+\]|\[.*?\]\(.*?\))/g);

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

        const linkMatch = part.match(/^\[(.*?)\]\((.*?)\)$/);
        if (linkMatch) {
            const [, text, url] = linkMatch;
            return (
                <a href={url} target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:text-blue-200 underline" key={i}>
                    {text}
                </a>
            );
        }

        // Split by **Bold**
        const boldParts = part.split(/\*\*(.*?)\*\*/g);
        return (
            <span key={i}>
                {boldParts.map((boldPart, j) => {
                    if (j % 2 === 1) { // Bold content
                        return <strong key={j} className="font-bold text-white">{boldPart}</strong>;
                    }
                    
                    // Split by `Code`
                    const codeParts = boldPart.split(/`(.*?)`/g);
                    return (
                        <span key={j}>
                            {codeParts.map((codePart, k) => {
                                if (k % 2 === 1) { // Code content
                                    return <code key={k} className="bg-gray-800 text-blue-200 px-1 py-0.5 rounded font-mono text-[11px] border border-gray-700/50">{codePart}</code>;
                                }
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
    // 1. Split by Code Blocks (```language ... ```)
    const parts = text.split(/(```[\s\S]*?```)/g);

    return (
        <div className="text-xs space-y-2">
            {parts.map((part, index) => {
                // Code Blocks
                if (part.startsWith('```')) {
                    const content = part.replace(/^```\w*\n?/, '').replace(/```$/, '');
                    return (
                        <div key={index} className="bg-gray-950 rounded p-2 overflow-x-auto border border-gray-700">
                             <pre className="font-mono text-[10px] text-gray-300 whitespace-pre-wrap">{content}</pre>
                        </div>
                    );
                }

                // Text Blocks - Split by newlines to handle lists and spacing
                const lines = part.split('\n');
                
                return (
                    <div key={index}>
                        {lines.map((line, lineIdx) => {
                             // Detect bullet points: "- ", "* ", "1. "
                             const listMatch = line.match(/^(\s*)([-*]|\d+\.)\s+(.*)/);
                             
                             if (listMatch) {
                                 const [, indent, marker, content] = listMatch;
                                 // Calculate indent size (approximate)
                                 const paddingLeft = indent.length > 0 ? `${(indent.length / 2) + 0.25}rem` : '0';
                                 
                                 return (
                                     <div key={lineIdx} className="flex items-start ml-1 mt-1" style={{ paddingLeft }}>
                                         <span className="mr-2 text-gray-500 flex-shrink-0 select-none min-w-[1rem] text-right font-mono opacity-80">
                                             {marker.match(/\d/) ? marker : '•'}
                                         </span>
                                         <span className="flex-1 break-words">
                                             {renderInlineMarkdown(content, onScrollToLog)}
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

// WebLLM Model ID
const WEB_LLM_MODEL_ID = "Llama-3.2-3B-Instruct-q4f16_1-MLC";
const WEBLMM_CONSENT_KEY = 'nhc_log_viewer_webllm_consent';

// Helper to parse a tool call from local model output (raw JSON or markdown)
const parseLocalToolCall = (text: string): { tool_name: string, arguments: any } | null => {
    let jsonString = text.trim();
    const jsonRegex = /```(?:json)?\s*([\s\S]*?)\s*```/;
    const match = jsonString.match(jsonRegex);
    if (match && match[1]) {
        jsonString = match[1].trim();
    }

    if (jsonString.startsWith('{') && jsonString.endsWith('}')) {
        try {
            const parsed = JSON.parse(jsonString);
            if (parsed.tool_name && parsed.arguments) {
                return parsed;
            }
        } catch (e) {
            // Not valid JSON
        }
    }
    return null;
};

export const AIAssistant: React.FC<AIAssistantProps> = ({ isOpen, onClose, visibleLogs, allLogs, allDaemons, onUpdateFilters, onScrollToLog }) => {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 'welcome',
      role: 'model',
      text: "Hello! I'm your AI log assistant. How can I help you analyze these logs?"
    }
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState<string>('');
  
  // Default to Flash as it's balanced
  const [modelTier, setModelTier] = useState<string>('gemini-2.5-flash');
  const [showWebLlmConsent, setShowWebLlmConsent] = useState(false);
  const [pendingPrompt, setPendingPrompt] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const cloudPrivacyWarningShown = useRef(false);

  // API Key Management
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [userApiKey, setUserApiKey] = useState('');
  const [tempApiKey, setTempApiKey] = useState('');

  useEffect(() => {
      const storedKey = localStorage.getItem('nhc_log_viewer_api_key');
      if (storedKey) {
          setUserApiKey(storedKey);
          setTempApiKey(storedKey);
      }
  }, []);

  const handleSaveSettings = () => {
      localStorage.setItem('nhc_log_viewer_api_key', tempApiKey.trim());
      setUserApiKey(tempApiKey.trim());
      setIsSettingsOpen(false);

      // After saving a key, check if the last action failed due to a missing key and retry it.
      const lastMessage = messages[messages.length - 1];
      if (lastMessage?.isError && lastMessage.text.includes('API Key is missing')) {
          const lastUserMessage = [...messages].reverse().find(m => m.role === 'user');
          if (lastUserMessage) {
              // Remove the error message from the UI before retrying
              setMessages(prev => prev.filter(m => m.id !== lastMessage.id));
              // Re-process the user's message with the new key
              processUserMessage(lastUserMessage.text);
          }
      }
  };

  // Ref to hold the WebLLM engine instance to avoid re-init
  const webLlmEngineRef = useRef<any>(null);

  const getCloudSystemInstruction = () => ({
        role: 'user',
        parts: [{ text: `
System Instruction: 
You are an expert log analyst and debugging assistant. Your goal is to help the user understand and solve issues in their application logs.

BEHAVIOR RULES:
1. **Analyze First**: Always use the \`search_logs\`, \`find_log_patterns\`, or \`trace_error_origin\` tools to find relevant information before answering. Do not guess.
2. **Expand Search**: When searching, generate SYNONYMS and RELATED TERMS. For example, if asked about charging, search for "charger", "battery", "voltage", etc.
3. **Be Proactive**: If you find an error, use \`suggest_solution\` to offer debugging steps.
4. **Strict Formatting**: When referring to a specific log line, YOU MUST use the format: [Log ID: <number>]. This creates a clickable link.
5. **ALWAYS RESPOND**: After executing a tool (like \`scroll_to_log\`), you MUST provide a text response explaining what you did. Never return an empty response.
6. **New Tabs**: If the user asks to filter the view (e.g., "show me only errors"), use \`update_filters\`. This will create a NEW tab.

RESPONSE STYLE:
- **Interpret, Don't Just List**: Provide a narrative summary of events. Explain the context behind the logs.
- **Be Conversational**: Write naturally. Use paragraphs and bullet points for readability.
- **Cite Evidence**: Weave [Log ID: <id>] references into your sentences as proof for your analysis.

Available Tools:
- \`search_logs\`: Search the ENTIRE log file. Use 'OR' mode for synonyms.
- \`scroll_to_log\`: Jump the user's view to a specific line.
- \`update_filters\`: Create a NEW tab with specific filters.
- \`find_log_patterns\`: Find repeating errors or unusual log frequency spikes.
- \`trace_error_origin\`: Trace events backwards from an error to find its root cause.
- \`suggest_solution\`: Get debugging advice for a specific error message.
` }]
  });

  const getLocalSystemInstruction = () => `
System Instruction:
You are an expert log analyst. Your goal is to help the user understand their application logs by using the tools available to you.

TOOL USAGE RULES:
1.  **Analyze First**: Always use a tool like \`search_logs\` to find information before answering.
2.  **How to Use Tools**: To use a tool, you MUST respond with ONLY a valid JSON object in the following format. Do not add any other text before or after the JSON.
    \`\`\`json
    {
      "tool_name": "name_of_the_tool",
      "arguments": { "arg1": "value1" }
    }
    \`\`\`
3.  **Strict Formatting**: When referring to a specific log line in your final text answer, YOU MUST use the format: [Log ID: <number>].
4.  **Final Answer**: After you have gathered enough information from the tools, provide a final, conversational answer in plain text. Do NOT use the JSON format for your final answer.

Available Tools:
-   \`search_logs\`: Search the ENTIRE log file. Arguments: \`{"keywords": ["term1"], "match_mode": "OR"|"AND"}\`
-   \`scroll_to_log\`: Jump the user's view to a specific line. Arguments: \`{"log_id": 123}\`
-   \`update_filters\`: Create a NEW tab with specific filters. Arguments: \`{"log_levels": ["ERROR"]}\`
-   \`find_log_patterns\`: Find repeating errors or log spikes. Arguments: \`{"pattern_type": "repeating_error"|"frequency_spike"}\`
-   \`trace_error_origin\`: Find the root cause of an error. Arguments: \`{"error_log_id": 123}\`
-   \`suggest_solution\`: Get debugging advice. Arguments: \`{"error_message": "the error text"}\`
`;

  const chatHistoryRef = useRef<Content[]>([
    getCloudSystemInstruction() as Content,
    {
        role: 'model',
        parts: [{ text: "Understood. I will act as an expert log analyst, using all available tools to proactively find, analyze, and suggest solutions for issues, always citing log IDs and communicating in a clear, conversational manner." }]
    }
  ]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isOpen, downloadProgress]);

  const handleResetChat = () => {
      setMessages([{ 
          id: 'welcome', 
          role: 'model', 
          text: 'Chat history cleared. How can I help you analyze the logs?' 
      }]);
      chatHistoryRef.current = [
        getCloudSystemInstruction() as Content,
        {
            role: 'model',
            parts: [{ text: "Understood. I will act as an expert log analyst, using all available tools to proactively find, analyze, and suggest solutions for issues, always citing log IDs and communicating in a clear, conversational manner." }]
        }
      ];
  };

  const handleWebLlmConsentAccept = () => {
      localStorage.setItem(WEBLMM_CONSENT_KEY, 'true');
      setShowWebLlmConsent(false);
      if (pendingPrompt) {
          setIsLoading(true);
          processWebLLM(pendingPrompt);
          setPendingPrompt(null);
      }
  };

  const handleWebLlmConsentCancel = () => {
      setShowWebLlmConsent(false);
      setPendingPrompt(null);
      // If the user cancels, remove their last message from the UI to avoid confusion.
      const lastMessage = messages[messages.length - 1];
      if(lastMessage && lastMessage.role === 'user') {
        setMessages(prev => prev.slice(0, -1));
      }
      // Revert model selection to avoid getting stuck in a consent loop
      setModelTier('gemini-2.5-flash'); 
      setIsLoading(false);
  };

  // Helper to execute client-side tools
  const executeTool = useCallback(async (name: string, args: any): Promise<any> => {
    console.log(`[AI] Executing tool: ${name}`, args);

    if (name === 'update_filters') {
        const newFilters: Partial<FilterState> = {};
        if (args.log_levels) newFilters.selectedLevels = args.log_levels.map((l: string) => l.toUpperCase());
        if (args.daemons) newFilters.selectedDaemons = args.daemons;
        
        if (args.search_keywords) {
            newFilters.keywordQueries = args.search_keywords.map((k: string) => {
                const clean = k.trim();
                if (!clean.startsWith('"') || !clean.endsWith('"')) {
                    return `"${clean.replace(/"/g, '\\"')}"`;
                }
                return clean;
            });
        }
        
        if (args.keyword_match_mode) {
             newFilters.keywordMatchMode = args.keyword_match_mode;
        } else if (args.search_keywords && args.search_keywords.length > 0) {
             newFilters.keywordMatchMode = 'OR';
        }

        onUpdateFilters(newFilters, args.reset_before_applying);
        return { result: "A new tab has been created with the requested filters. The user is now looking at the new tab." };
    } 
    else if (name === 'scroll_to_log') {
        const logId = Number(args.log_id);
        if (!isNaN(logId)) {
            onScrollToLog(logId);
            return { result: `Success. User view has been scrolled to [Log ID: ${logId}].` };
        }
        return { error: "Invalid log ID provided." };
    } 
    else if (name === 'search_logs') {
        // --- Intelligent Weighted Search ---
        let keywords: string[] = [];
        if (args.keywords && Array.isArray(args.keywords)) {
            keywords = args.keywords.map((k: string) => k.toLowerCase().trim());
        } else if (args.query) {
            keywords = args.query.split(/\s+/).map((k: string) => k.toLowerCase().trim());
        }

        const matchMode = args.match_mode || 'OR';
        const limit = args.limit || 50;
        
        if (keywords.length === 0) return { result: "Empty query." };

        const keywordCounts: Record<string, number> = {};
        keywords.forEach(k => keywordCounts[k] = 0);

        for (const log of allLogs) {
             const text = (log.message + " " + log.daemon + " " + log.level + " " + log.timestamp.toISOString()).toLowerCase();
             for (const k of keywords) {
                 if (text.includes(k)) {
                     keywordCounts[k]++;
                 }
             }
        }

        const keywordWeights: Record<string, number> = {};
        keywords.forEach(k => {
            keywordWeights[k] = 1000 / (keywordCounts[k] + 1);
        });

        interface ScoredLog { log: LogEntry; score: number; }
        const scoredMatches: ScoredLog[] = [];

        for (const log of allLogs) {
            const text = (log.message + " " + log.daemon + " " + log.level + " " + log.timestamp.toISOString()).toLowerCase();
            let logScore = 0;
            let matchedCount = 0;

            for (const k of keywords) {
                if (text.includes(k)) {
                    logScore += keywordWeights[k];
                    matchedCount++;
                }
            }
            const isMatch = matchMode === 'AND' ? matchedCount === keywords.length : matchedCount > 0;
            if (isMatch) scoredMatches.push({ log, score: logScore });
        }

        scoredMatches.sort((a, b) => b.score - a.score);
        const topMatches = scoredMatches.slice(0, limit).map(m => m.log);

        if (topMatches.length === 0) return { result: `No logs found matching terms: [${keywords.join(', ')}] with mode ${matchMode}.` };
        return {
            logs: topMatches.map(l => ({ id: l.id, timestamp: l.timestamp.toISOString(), level: l.level, daemon: l.daemon, message: l.message }))
        };
    }
    else if (name === 'find_log_patterns') {
        const { pattern_type, time_window_minutes } = args;
        const now = allLogs.length > 0 ? allLogs[allLogs.length - 1].timestamp.getTime() : Date.now();
        const startTime = time_window_minutes ? now - time_window_minutes * 60 * 1000 : 0;
        
        const logsInWindow = allLogs.filter(log => log.timestamp.getTime() >= startTime);

        if (pattern_type === 'repeating_error') {
            const errorCounts = new Map<string, { count: number; log: LogEntry }>();
            logsInWindow
                .filter(log => log.level === 'ERROR' || log.level === 'CRITICAL')
                .forEach(log => {
                    const existing = errorCounts.get(log.message);
                    if (existing) {
                        existing.count++;
                    } else {
                        errorCounts.set(log.message, { count: 1, log });
                    }
                });

            const sortedErrors = Array.from(errorCounts.values()).sort((a, b) => b.count - a.count).slice(0, 5);
            if (sortedErrors.length === 0) return { result: "No repeating errors found in the specified time window." };
            return {
                result: `Found ${sortedErrors.length} unique repeating errors.`,
                top_repeating_errors: sortedErrors.map(e => ({ message: e.log.message, count: e.count, example_log_id: e.log.id })),
            };
        }
        
        if (pattern_type === 'frequency_spike') {
            const buckets = new Map<number, number>();
            logsInWindow.forEach(log => {
                const bucketTimestamp = Math.floor(log.timestamp.getTime() / 1000); // per second
                buckets.set(bucketTimestamp, (buckets.get(bucketTimestamp) || 0) + 1);
            });

            if (buckets.size === 0) return { result: "No logs to analyze for spikes." };

            const counts = Array.from(buckets.values());
            const avg = counts.reduce((sum, count) => sum + count, 0) / counts.length;
            const stdDev = Math.sqrt(counts.map(x => Math.pow(x - avg, 2)).reduce((a, b) => a + b) / counts.length);
            const spikeThreshold = Math.max(5, avg + 2 * stdDev);

            const spikes = Array.from(buckets.entries())
                .filter(([, count]) => count > spikeThreshold)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 5)
                .map(([timestamp, count]) => ({
                    timestamp: new Date(timestamp * 1000).toISOString(),
                    log_count: count,
                    average_logs_per_second: avg.toFixed(2)
                }));
            
            if (spikes.length === 0) return { result: `No significant log spikes found. Average was ${avg.toFixed(2)} logs/sec.` };
            return { result: `Found ${spikes.length} significant log spikes.`, spikes };
        }
        return { error: `Unknown pattern_type: ${pattern_type}` };
    }
    else if (name === 'trace_error_origin') {
        const { error_log_id, trace_window_seconds = 60 } = args;
        const errorLog = allLogs.find(log => log.id === error_log_id);
        if (!errorLog) return { error: `Log with ID ${error_log_id} not found.` };

        const endTime = errorLog.timestamp.getTime();
        const startTime = endTime - trace_window_seconds * 1000;
        
        const traceLogs = allLogs.filter(log => {
            const logTime = log.timestamp.getTime();
            return logTime >= startTime && logTime <= endTime;
        });
        return {
            result: `Found ${traceLogs.length} logs in the ${trace_window_seconds} seconds leading up to log ${error_log_id}.`,
            logs: traceLogs.map(l => ({ id: l.id, timestamp: l.timestamp.toISOString(), level: l.level, daemon: l.daemon, message: l.message }))
        };
    }
    else if (name === 'suggest_solution') {
        const { error_message } = args;
        if (!error_message) return { error: "No error message provided." };
        
        const effectiveApiKey = import.meta.env.VITE_API_KEY || userApiKey;
        const isLocal = modelTier.startsWith('local') || modelTier.startsWith('webllm');
        
        if (!isLocal && !effectiveApiKey) return { error: "Cannot suggest solution without API key." };

        try {
            const solutionPrompt = `You are a senior software engineer providing a brief, helpful, and actionable solution for a specific error message. Do not reference the user or the chat history. Focus only on the error. Use markdown for code snippets or commands if applicable. The error is: "${error_message}"`;
            let solutionText = "";

            if (isLocal) {
                if (modelTier === 'local' && window.ai?.languageModel) {
                    const session = await window.ai.languageModel.create();
                    solutionText = await session.prompt(solutionPrompt);
                    session.destroy();
                } else if (modelTier === 'webllm' && webLlmEngineRef.current) {
                    const reply = await webLlmEngineRef.current.chat.completions.create({ messages: [{ role: "user", content: solutionPrompt }] });
                    solutionText = reply.choices[0].message.content || "Could not generate a solution.";
                } else {
                    solutionText = "Local AI is not available to suggest a solution.";
                }
            } else {
                const ai = new GoogleGenAI({ apiKey: effectiveApiKey });
                const result = await ai.models.generateContent({ model: 'gemini-2.5-flash', contents: [{ role: 'user', parts: [{ text: solutionPrompt }] }] });
                solutionText = result.text || "Could not generate a solution.";
            }
            return { result: "Here is a potential solution:", solution: solutionText };
        } catch (e) {
            console.error("Error in suggest_solution tool:", e);
            return { error: "An error occurred while trying to generate a solution." };
        }
    }
    return { error: `Unknown tool: ${name}` };
  }, [allLogs, onUpdateFilters, onScrollToLog, modelTier, userApiKey]);

  const processLocalAI = async (userText: string) => {
      setIsLoading(true);
      const localHistory: { role: 'user' | 'model' | 'tool'; content: string }[] = [];
      localHistory.push({ role: 'user', content: userText });
      
      let turnCount = 0;
      const MAX_TURNS = 10;

      try {
          if (!window.ai?.languageModel) {
            throw new Error("Local AI (Gemini Nano) is not available in this browser. This feature requires the latest Chrome with the 'Prompt API for Gemini Nano' flag enabled.");
          }
          
          setDownloadProgress("Checking local model availability...");
          const capabilities = await window.ai.languageModel.capabilities();
          if (capabilities.available === 'no') throw new Error("Local AI (Gemini Nano) is not supported by your device.");
          if (capabilities.available === 'after-download') setDownloadProgress("Downloading Gemini Nano model...");

          const session = await window.ai.languageModel.create({ outputLanguage: 'en' });
          setDownloadProgress("");

          while (turnCount < MAX_TURNS) {
              turnCount++;

              const fullPrompt = getLocalSystemInstruction() + '\n\n--- CHAT HISTORY ---\n\n' +
                                 localHistory.map(msg => `${msg.role.toUpperCase()}:\n${msg.content}`).join('\n\n');

              const replyText = await session.prompt(fullPrompt);
              const toolCall = parseLocalToolCall(replyText);
              
              if (toolCall) {
                  localHistory.push({ role: 'model', content: replyText });
                  const toolResult = await executeTool(toolCall.tool_name, toolCall.arguments);
                  localHistory.push({ role: 'tool', content: JSON.stringify(toolResult, null, 2) });
                  continue;
              }

              setMessages(prev => [...prev, { id: Date.now().toString(), role: 'model', text: replyText + "\n\n*(Generated locally on-device)*" }]);
              session.destroy();
              return;
          }

          throw new Error("The local model could not complete the request in the allotted time.");

      } catch (error: any) {
          console.error("Local AI Error:", error);
           setMessages(prev => [...prev, { 
               id: Date.now().toString(), 
               role: 'model', 
               text: "Error running local model: " + (error.message || "Unknown error"), 
               isError: true 
           }]);
      } finally {
          setIsLoading(false);
          setDownloadProgress("");
      }
  };

  const processWebLLM = async (userText: string) => {
      setIsLoading(true);
      const webLlmHistory: any[] = [
           { role: "system", content: getLocalSystemInstruction() },
           { role: 'user', content: userText }
      ];

      let turnCount = 0;
      const MAX_TURNS = 10;

      try {
          if (!webLlmEngineRef.current) {
               setDownloadProgress("Initializing engine...");
               webLlmEngineRef.current = await CreateMLCEngine(
                   WEB_LLM_MODEL_ID,
                   { initProgressCallback: (report) => setDownloadProgress(report.text) }
               );
          }
          setDownloadProgress("");

          while (turnCount < MAX_TURNS) {
               turnCount++;

               const reply = await webLlmEngineRef.current.chat.completions.create({
                   messages: webLlmHistory,
                   temperature: 0.5,
                   max_tokens: 1024,
               });

               const replyText = reply.choices[0].message.content || "";
               const toolCall = parseLocalToolCall(replyText);
               
               if (toolCall) {
                  webLlmHistory.push({ role: 'assistant', content: replyText });
                  const toolResult = await executeTool(toolCall.tool_name, toolCall.arguments);
                  webLlmHistory.push({ role: 'tool', content: JSON.stringify(toolResult, null, 2) });
                  continue;
               }
               
               setMessages(prev => [...prev, { id: Date.now().toString(), role: 'model', text: replyText + "\n\n*(Generated locally via WebLLM)*" }]);
               return;
          }

          throw new Error("The local model could not complete the request in the allotted time.");

      } catch (error: any) {
          console.error("WebLLM Error:", error);
          let errorMsg = "Error running WebLLM: " + (error.message || "Unknown error");
          if (error.message?.includes("WebGPU")) {
              errorMsg = "WebGPU is not supported or enabled in this browser. Please use Chrome/Edge and ensure hardware acceleration is on.";
          } else if (error.message?.includes("Cache")) {
              errorMsg = "Failed to download model. Please check your internet connection or firewall. (Cache Error)";
          }
          setMessages(prev => [...prev, { 
              id: Date.now().toString(), 
              role: 'model', 
              text: errorMsg, 
              isError: true 
          }]);
      } finally {
          setIsLoading(false);
          setDownloadProgress("");
      }
  };

  const processUserMessage = async (userText: string) => {
    // Handle Local Modes
    if (modelTier === 'local') {
        setIsLoading(true);
        processLocalAI(userText);
        return;
    }
    
    if (modelTier === 'webllm') {
        const hasConsented = localStorage.getItem(WEBLMM_CONSENT_KEY) === 'true';

        if (!webLlmEngineRef.current && !hasConsented) {
            setPendingPrompt(userText);
            setShowWebLlmConsent(true);
            return;
        }
        
        setIsLoading(true);
        processWebLLM(userText);
        return;
    }

    // Show privacy warning on first cloud use
    if (!cloudPrivacyWarningShown.current) {
        setMessages(prev => [...prev, {
            id: `privacy-warning-${Date.now()}`,
            role: 'model',
            text: '**Privacy Notice:** You are using a cloud-based AI model. To answer your questions, a summary of relevant logs will be sent to Google for analysis. For 100% on-device processing, please select a "Local" model from the dropdown menu.',
            isWarning: true,
        }]);
        cloudPrivacyWarningShown.current = true;
    }

    const effectiveApiKey = import.meta.env.VITE_API_KEY || userApiKey;

    if (!effectiveApiKey) {
      setMessages(prev => [...prev, { 
          id: Date.now().toString(), 
          role: 'model', 
          text: "Error: API Key is missing. Please click the Settings icon (⚙️) to enter your key. You can get one from [Google AI Studio](https://aistudio.google.com/api-keys).", 
          isError: true 
      }]);
      setIsSettingsOpen(true);
      return;
    }

    setIsLoading(true);
    const historyStartIndex = chatHistoryRef.current.length;

    try {
      const ai = new GoogleGenAI({ apiKey: effectiveApiKey });
      const modelName = modelTier;
      const model = ai.models;
      
      const userContent: Content = { role: 'user', parts: [{ text: userText }] };
      chatHistoryRef.current = [...chatHistoryRef.current, userContent];

      let finalResponseText = '';
      let turnCount = 0;
      const MAX_TURNS = 10;

      while (turnCount < MAX_TURNS) {
        turnCount++;
        console.log(`[AI] Turn ${turnCount}/${MAX_TURNS} using ${modelName}`);
        
        const result = await model.generateContent({
            model: modelName,
            contents: chatHistoryRef.current,
            config: {
                tools: [{ functionDeclarations: [updateFiltersTool, scrollToLogTool, searchLogsTool, findLogPatternsTool, traceErrorOriginTool, suggestSolutionTool] }],
            },
        });

        const responseContent = result.candidates?.[0]?.content;
        if (!responseContent) throw new Error("No content in response");

        chatHistoryRef.current.push(responseContent);

        const functionCalls = result.functionCalls;
        
        if (functionCalls && functionCalls.length > 0) {
             const toolResponses: Part[] = [];

             for (const call of functionCalls) {
                 const toolResult = await executeTool(call.name, call.args);
                 toolResponses.push({
                     functionResponse: {
                         name: call.name,
                         response: { result: toolResult }
                     }
                 });
             }
             chatHistoryRef.current.push({ role: 'user', parts: toolResponses });
        } else {
            if (responseContent.parts) {
                for (const part of responseContent.parts) {
                    if (part.text) finalResponseText += part.text;
                }
            }
            break;
        }
      }

      if (!finalResponseText && turnCount >= MAX_TURNS) {
          console.warn("[AI] Max turns reached. Forcing summary.");
          const summaryPrompt: Content = { 
              role: 'user', 
              parts: [{ text: "You have reached the maximum number of tool calls. Please stop searching and answer the user's question based on the information you have found so far. Summarize what you know." }] 
          };
          chatHistoryRef.current.push(summaryPrompt);
          
          const finalResult = await model.generateContent({
            model: modelName,
            contents: chatHistoryRef.current,
          });
          
          finalResponseText = finalResult.candidates?.[0]?.content?.parts?.[0]?.text || "I was unable to complete the analysis in the allotted time.";
          chatHistoryRef.current.push(finalResult.candidates?.[0]?.content as Content);
      }

      if (finalResponseText) {
        setMessages(prev => [...prev, { id: Date.now().toString(), role: 'model', text: finalResponseText }]);
      } else {
        setMessages(prev => [...prev, { id: Date.now().toString(), role: 'model', text: "I've completed the requested actions." }]);
      }

    } catch (error: any) {
      console.error("AI Error:", error);
      
      chatHistoryRef.current = chatHistoryRef.current.slice(0, historyStartIndex);

      let errorMessage = "I'm having trouble connecting to the AI right now.";
      
      if (typeof error === 'object' && error !== null) {
          const msg = error.message || JSON.stringify(error);
          if (msg.includes('429') || msg.includes('RESOURCE_EXHAUSTED')) {
              errorMessage = "Rate limit exceeded. Try switching to a faster model (Fast/Balanced) or wait a moment.";
              if (modelTier === 'gemini-3-pro-preview') {
                  errorMessage += " Switching to Balanced automatically.";
                  setModelTier('gemini-2.5-flash');
              }
          }
          if (msg.includes('API_KEY')) {
              errorMessage = "Invalid API Key. Please check settings.";
          }
      }
      
      setMessages(prev => [...prev, { id: Date.now().toString(), role: 'model', text: errorMessage, isError: true }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!input.trim() || isLoading) return;

    const text = input;
    setMessages(prev => [...prev, { id: Date.now().toString(), role: 'user', text }]);
    setInput('');
    processUserMessage(text);
  };

  const handleRetry = (errorMsgId: string) => {
      if (isLoading) return;
      
      const lastUserMsg = [...messages].reverse().find(m => m.role === 'user');
      
      if (lastUserMsg) {
          setMessages(prev => prev.filter(m => m.id !== errorMsgId));
          processUserMessage(lastUserMsg.text);
      }
  };

  const handleQuickAction = (action: 'summarize' | 'errors' | 'solution' | 'capabilities') => {
    if (isLoading) return;
    
    let prompt = "";
    if (action === 'summarize') {
      prompt = "Summarize the key events by searching the entire log file.";
    } else if (action === 'errors') {
      prompt = "Find critical failures in the entire log file and explain the root cause.";
    } else if (action === 'solution') {
      prompt = "Find the most critical error in the entire log file and then provide a detailed solution for it.";
    } else if (action === 'capabilities') {
      prompt = "Explain your capabilities as an expert log analyst assistant. Describe the tools you have available and provide a clear, user-friendly example for each one in a markdown list format. Frame it as if you are introducing yourself.";
    }

    setMessages(prev => [...prev, { id: Date.now().toString(), role: 'user', text: prompt }]);
    processUserMessage(prompt);
  };

  if (!isOpen) return null;

  return (
    <>
    <div className="fixed inset-y-0 right-0 z-40 w-full md:w-96 bg-gray-800 shadow-2xl border-l border-gray-700 flex flex-col transform transition-transform duration-300 ease-in-out">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-700 bg-gray-900">
        <div className="flex items-center space-x-2">
            <div className="bg-gradient-to-tr from-blue-500 to-purple-500 p-1.5 rounded-lg">
                <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
            </div>
            <div>
                <h2 className="text-sm font-bold text-white leading-none">AI Assistant</h2>
                <div className="flex items-center space-x-1 mt-1">
                    <span className="text-[9px] text-gray-400">Model:</span>
                    <select 
                        value={modelTier} 
                        onChange={(e) => setModelTier(e.target.value)}
                        className="bg-gray-800 border border-gray-600 text-xs rounded px-1 py-0.5 text-blue-300 focus:outline-none focus:border-blue-500 cursor-pointer max-w-[140px]"
                        disabled={isLoading}
                    >
                        <option value="gemini-flash-lite-latest">Fast</option>
                        <option value="gemini-2.5-flash">Balanced</option>
                        <option value="gemini-3-pro-preview">Reasoning</option>
                        <option value="local">Local (Chrome Nano)</option>
                        <option value="webllm">Local (WebLLM - Llama 3)</option>
                    </select>
                </div>
            </div>
        </div>
        <div className="flex items-center space-x-1">
            <button 
                onClick={() => setIsSettingsOpen(true)}
                className="text-gray-400 hover:text-white p-1 rounded-md hover:bg-gray-700"
                title="Settings / API Key"
            >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"></path><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path></svg>
            </button>
             <button 
                onClick={handleResetChat} 
                className="text-gray-400 hover:text-red-400 p-1 rounded-md hover:bg-gray-700"
                title="Reset Chat"
            >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path></svg>
            </button>
            <button onClick={onClose} className="text-gray-400 hover:text-white p-1 rounded-md hover:bg-gray-700">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path></svg>
            </button>
        </div>
      </div>
      
      {/* Quick Actions */}
      <div className="p-3 grid grid-cols-2 gap-2 border-b border-gray-700 bg-gray-800/50">
        <button 
            onClick={() => handleQuickAction('summarize')}
            disabled={isLoading}
            className="flex items-center justify-center space-x-1 bg-gray-700 hover:bg-gray-600 text-gray-200 py-2 px-3 rounded text-xs font-medium transition-colors disabled:opacity-50"
            title="Ask AI to search all logs and provide a summary of key events."
        >
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 011.414.586l5.414 5.414a2 2 0 01.586 1.414V19a2 2 0 01-2 2z"></path></svg>
            <span>Summarize</span>
        </button>
        <button 
            onClick={() => handleQuickAction('errors')}
            disabled={isLoading}
            className="flex items-center justify-center space-x-1 bg-gray-700 hover:bg-gray-600 text-gray-200 py-2 px-3 rounded text-xs font-medium transition-colors disabled:opacity-50"
            title="Ask AI to search all logs for errors and analyze the root cause."
        >
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path></svg>
            <span>Analyze Errors</span>
        </button>
         <button 
            onClick={() => handleQuickAction('solution')}
            disabled={isLoading}
            className="flex items-center justify-center space-x-1 bg-gray-700 hover:bg-gray-600 text-gray-200 py-2 px-3 rounded text-xs font-medium transition-colors disabled:opacity-50"
            title="Ask AI to find the most critical error and suggest a solution for it."
        >
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" /></svg>
            <span>Suggest Solution</span>
        </button>
        <button 
            onClick={() => handleQuickAction('capabilities')}
            disabled={isLoading}
            className="flex items-center justify-center space-x-1 bg-gray-700 hover:bg-gray-600 text-gray-200 py-2 px-3 rounded text-xs font-medium transition-colors disabled:opacity-50"
            title="Learn what the AI assistant can do."
        >
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
            <span>Capabilities</span>
        </button>
      </div>

      {/* Messages */}
      <div className="flex-grow overflow-y-auto p-4 space-y-4 bg-gray-900/50">
        {messages.map((msg) => (
          <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[90%] rounded-lg p-3 text-xs leading-relaxed shadow-sm ${
                msg.role === 'user' 
                  ? 'bg-blue-600 text-white' 
                  : msg.isError
                    ? 'bg-red-900/50 border border-red-700 text-red-200'
                    : msg.isWarning
                      ? 'bg-yellow-900/50 border border-yellow-700 text-yellow-200'
                      : 'bg-gray-700 text-gray-200 border border-gray-600'
            }`}>
              <div className="flex items-start">
                  {msg.isWarning && (
                    <div className="mr-2 flex-shrink-0 text-yellow-400 pt-0.5">
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path></svg>
                    </div>
                  )}
                  <div className="flex-1">
                     <FormattedMessage text={msg.text} onScrollToLog={onScrollToLog} />
                  </div>
                  {msg.isError && (
                      <button 
                          onClick={() => handleRetry(msg.id)}
                          className="ml-2 p-1 text-red-400 hover:text-red-300 hover:bg-red-900/50 rounded transition-colors flex-shrink-0"
                          title="Retry"
                      >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path></svg>
                      </button>
                  )}
              </div>
            </div>
          </div>
        ))}
        {isLoading && (
            <div className="flex justify-start">
                <div className="bg-gray-700 rounded-lg p-3 border border-gray-600 flex flex-col items-start space-y-2 max-w-[90%]">
                    <div className="flex items-center space-x-2">
                        <div className="flex space-x-1">
                            <div className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce"></div>
                            <div className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                            <div className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
                        </div>
                        <span className="text-[10px] text-gray-400">
                            {modelTier === 'gemini-3-pro-preview' ? 'Reasoning...' : 
                             modelTier === 'local' ? (downloadProgress ? 'Initializing...' : 'Processing locally (Nano)...') : 
                             modelTier === 'webllm' ? (downloadProgress ? 'Initializing...' : 'Thinking (Llama 3)...') :
                             'Analyzing logs...'}
                        </span>
                    </div>
                    {downloadProgress && (
                        <div className="w-full">
                            <div className="text-[9px] text-gray-500 font-mono mb-1 truncate">{downloadProgress}</div>
                            <div className="w-full bg-gray-900 rounded-full h-1">
                                <div className="bg-blue-500 h-1 rounded-full transition-all duration-300" style={{ width: '100%' }}></div>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} className="p-4 border-t border-gray-700 bg-gray-800">
        <div className="flex space-x-2">
            <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Message AI..."
                disabled={isLoading}
                className="flex-grow bg-gray-900 border border-gray-600 text-white rounded-md px-3 py-2 text-xs focus:outline-none focus:ring-1 focus:ring-blue-500 disabled:opacity-50"
            />
            <button
                type="submit"
                disabled={!input.trim() || isLoading}
                className="bg-blue-600 text-white p-2 rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:bg-gray-600 transition-colors"
            >
                <svg className="w-4 h-4 transform rotate-90" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"></path></svg>
            </button>
        </div>
      </form>
    </div>

    {/* Settings Modal */}
    {isSettingsOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
            <div className="bg-gray-800 rounded-lg shadow-xl border border-gray-700 w-full max-w-md p-6">
                <div className="flex justify-between items-center mb-4">
                    <h3 className="text-lg font-bold text-white">Settings</h3>
                    <button onClick={() => setIsSettingsOpen(false)} className="text-gray-400 hover:text-white">
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path></svg>
                    </button>
                </div>
                
                <div className="space-y-4">
                    <div>
                        <label className="block text-xs font-medium text-gray-300 mb-1">Google Gemini API Key</label>
                        <input 
                            type="password" 
                            value={tempApiKey}
                            onChange={(e) => setTempApiKey(e.target.value)}
                            placeholder="AIza..."
                            className="w-full bg-gray-900 border border-gray-600 text-white rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
                        />
                        <p className="text-[10px] text-gray-500 mt-1">
                            Your key is stored locally in your browser and used only for AI requests. 
                            If you have set a VITE_API_KEY environment variable, it will be prioritized over this one.
                        </p>
                         <p className="text-[10px] text-gray-500 mt-2">
                          Don't have a key? Get one from{' '}
                          <a href="https://aistudio.google.com/api-keys" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">
                            Google AI Studio
                          </a>.
                        </p>
                    </div>
                </div>

                <div className="mt-6 flex justify-end space-x-2">
                    <button 
                        onClick={() => setIsSettingsOpen(false)}
                        className="px-4 py-2 bg-gray-700 text-gray-200 text-sm rounded-md hover:bg-gray-600 transition-colors"
                    >
                        Cancel
                    </button>
                    <button 
                        onClick={handleSaveSettings}
                        className="px-4 py-2 bg-blue-600 text-white text-sm rounded-md hover:bg-blue-700 transition-colors"
                    >
                        Save Key
                    </button>
                </div>
            </div>
        </div>
    )}

    {/* WebLLM Consent Modal */}
    {showWebLlmConsent && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
            <div className="bg-gray-800 rounded-lg shadow-xl border border-gray-700 w-full max-w-md">
                <div className="p-6">
                    <div className="flex items-start space-x-4">
                        <div className="flex-shrink-0 h-10 w-10 flex items-center justify-center rounded-full bg-gray-700">
                            <svg className="h-6 w-6 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"></path></svg>
                        </div>
                        <div>
                            <h3 className="text-lg font-bold text-white">Enable On-Device AI</h3>
                            <p className="text-xs text-gray-400 mt-1 leading-relaxed">
                                To use the on-device AI, a large model file needs to be downloaded and cached in your browser.
                            </p>
                        </div>
                    </div>

                    <div className="mt-4 bg-gray-900/50 border border-gray-700 rounded-md p-3 text-xs space-y-2">
                        <div className="flex justify-between items-center">
                            <span className="font-medium text-gray-400">Model Size</span>
                            <span className="font-mono text-gray-200">~2 GB</span>
                        </div>
                        <div className="flex justify-between items-center">
                            <span className="font-medium text-gray-400">Download</span>
                            <span className="font-mono text-gray-200">One-time only</span>
                        </div>
                        <div className="flex justify-between items-center">
                            <span className="font-medium text-gray-400">Privacy</span>
                            <span className="font-mono text-green-400">100% Local</span>
                        </div>
                    </div>
                </div>
                
                <div className="bg-gray-700/50 px-6 py-3 flex justify-end space-x-2 rounded-b-lg">
                    <button 
                        onClick={handleWebLlmConsentCancel}
                        className="px-4 py-2 bg-gray-600 text-gray-200 text-sm rounded-md hover:bg-gray-500 transition-colors"
                    >
                        Cancel
                    </button>
                    <button 
                        onClick={handleWebLlmConsentAccept}
                        className="px-4 py-2 bg-blue-600 text-white text-sm rounded-md hover:bg-blue-700 transition-colors"
                    >
                        Accept & Download
                    </button>
                </div>
            </div>
        </div>
    )}
    </>
  );
};