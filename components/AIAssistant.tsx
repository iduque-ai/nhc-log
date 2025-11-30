// FIX: Imported `useMemo` from React to resolve reference error.
import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { GoogleGenAI, Type, FunctionDeclaration, Content, Part } from "@google/genai";
import { CreateMLCEngine } from "@mlc-ai/web-llm";
import { LogEntry, FilterState } from '../types.ts';

// FIX: Local definition to solve import issue with GenerateContentResponse
interface GenerateContentResponse {
  text: string | undefined;
  functionCalls?: { name: string; args: any; }[];
  candidates?: { content?: Content }[];
}

interface AIAssistantProps {
  isOpen: boolean;
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
  description: 'Search ALL logs for specific information, including timestamps. You can provide multiple synonyms or related terms to broaden the search. Returns a summary of findings.',
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
        description: 'Maximum number of logs to return (default 100).',
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
            // After finding a specific issue, the AI should focus on digging deeper or solving it.
            return [allTools.traceErrorOriginTool, allTools.suggestSolutionTool, allTools.scrollToLogTool, allTools.searchLogsTool];
        case 'IDLE':
        default:
            // Initially, the AI should focus on broad exploration.
            return [allTools.searchLogsTool, allTools.findLogPatternsTool, allTools.updateFiltersTool, allTools.scrollToLogTool];
    }
};

// --- Formatted Message Component ---

// Helper to recursively render inline markdown
const renderInlineMarkdown = (text: string, onScrollToLog: (id: number) => void): React.ReactNode => {
    // Split by [Log ID: 123], markdown links [text](url), and raw http(s) links
    const parts = text.split(/(\[Log ID: \d+\]|\[.*?\]\(.*?\)|https?:\/\/[^\s\)]+)/g);

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

        if (part.startsWith('http')) {
            return (
                <a href={part} target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:text-blue-200 underline" key={i}>
                    {part}
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

export const AIAssistant: React.FC<AIAssistantProps> = ({ isOpen, onClose, visibleLogs, allLogs, allDaemons, onUpdateFilters, onScrollToLog, savedFindings, onSaveFinding }) => {
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

  // Conversation State Management
  const conversationStateRef = useRef<ConversationState>('IDLE');
  
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
  };
  
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const addMessage = useCallback((role: 'user' | 'model', text: string, isError: boolean = false) => {
    setMessages(prev => [...prev, { id: Date.now().toString(), role, text, isError }]);
  }, []);
  
  const handleToolCall = useCallback((toolName: string, args: any) => {
    switch (toolName) {
      case 'update_filters':
        onUpdateFilters({
          selectedLevels: args.log_levels,
          selectedDaemons: args.daemons,
          keywordQueries: args.search_keywords,
          keywordMatchMode: args.keyword_match_mode || 'OR',
        }, args.reset_before_applying ?? true);
        return { success: true, message: `Created a new tab with the specified filters.` };
      
      case 'scroll_to_log':
        onScrollToLog(args.log_id);
        return { success: true, message: `Scrolled to log ID ${args.log_id}.` };

      case 'search_logs': {
        const { keywords, match_mode = 'OR', limit = 100 } = args;
        if (!keywords || keywords.length === 0) {
          return { logs: [], message: 'No keywords provided for search.' };
        }
        
        const lowerCaseKeywords = keywords.map((k: string) => k.toLowerCase());
        
        const results = allLogs
          .filter(log => {
            const lowerCaseMessage = log.message.toLowerCase();
            if (match_mode === 'AND') {
              return lowerCaseKeywords.every((kw: string) => lowerCaseMessage.includes(kw));
            } else { // OR
              return lowerCaseKeywords.some((kw: string) => lowerCaseMessage.includes(kw));
            }
          })
          .slice(0, limit)
          .map(log => ({ id: log.id, timestamp: log.timestamp, level: log.level, message: log.message }));

        return {
          logs: results,
          message: `Found ${results.length} logs matching [${keywords.join(', ')}].`
        };
      }
      
      case 'find_log_patterns': {
        const { pattern_type, time_window_minutes } = args;
        const targetLogs = time_window_minutes
            ? allLogs.filter(log => {
                const logTime = log.timestamp.getTime();
                const endTime = allLogs[allLogs.length - 1].timestamp.getTime();
                const startTime = endTime - time_window_minutes * 60 * 1000;
                return logTime >= startTime && logTime <= endTime;
              })
            : allLogs;
        
        if (pattern_type === 'repeating_error') {
            const errorCounts: Record<string, { count: number, firstId: number }> = {};
            targetLogs.forEach(log => {
                if (log.level === 'ERROR' || log.level === 'CRITICAL') {
                    const genericMessage = log.message.replace(/\d+/g, 'N'); // Normalize numbers
                    if (!errorCounts[genericMessage]) {
                        errorCounts[genericMessage] = { count: 0, firstId: log.id };
                    }
                    errorCounts[genericMessage].count++;
                }
            });
            const topErrors = Object.entries(errorCounts)
                .sort((a, b) => b[1].count - a[1].count)
                .slice(0, 5)
                .map(([msg, data]) => ({ message: msg, count: data.count, example_log_id: data.firstId }));
            return { patterns: topErrors, message: `Found ${topErrors.length} repeating error patterns.` };
        }
        // ... other pattern types
        return { patterns: [], message: 'Pattern type not implemented yet.' };
      }
        
      case 'trace_error_origin': {
          const { error_log_id, trace_window_seconds = 60 } = args;
          const errorLog = allLogs.find(l => l.id === error_log_id);
          if (!errorLog) return { trace: [], message: `Log ID ${error_log_id} not found.` };
          
          const endTime = errorLog.timestamp.getTime();
          const startTime = endTime - trace_window_seconds * 1000;
          
          const traceLogs = allLogs.filter(log => {
             const logTime = log.timestamp.getTime();
             return logTime >= startTime && logTime <= endTime;
          }).map(log => ({ id: log.id, timestamp: log.timestamp, level: log.level, message: log.message }));
          
          return { trace: traceLogs, message: `Found ${traceLogs.length} logs in the ${trace_window_seconds}s before log ${error_log_id}.` };
      }
        
      case 'suggest_solution':
        // This is a placeholder. In a real scenario, this might call another API
        // or have a predefined set of solutions. The model itself will generate the text.
        return { success: true, message: `Providing solution for: "${args.error_message}"` };

      default:
        return { error: `Tool "${toolName}" not found.` };
    }
  }, [allLogs, onUpdateFilters, onScrollToLog]);

  const runCloudAI = useCallback(async (prompt: string) => {
    const apiKey = userApiKey || import.meta.env.VITE_API_KEY;
    if (!apiKey) {
      addMessage('model', "API key is not configured. Please set one in the settings (⚙️).", true);
      setIsLoading(false);
      return;
    }
    
    if (!cloudPrivacyWarningShown.current) {
        addMessage('model', "You are using a cloud-based AI model. A summary of your log data will be sent to Google for analysis. For fully private, on-device analysis, you can switch to a local model in settings.", false);
        cloudPrivacyWarningShown.current = true;
    }
    
    const ai = new GoogleGenAI({ apiKey });

    const logSamples = visibleLogs.length > 5 ?
        [...visibleLogs.slice(0, 3), ...visibleLogs.slice(-2)] :
        visibleLogs;

    const sampleLogText = logSamples.map(l => `ID ${l.id}: ${l.timestamp.toISOString()} ${l.daemon} [${l.level}] ${l.message}`).join('\n');
    
    const systemPrompt = `You are an expert AI assistant embedded in a log analysis tool. Your primary goal is to help users understand their logs and identify problems.

# CONTEXT
- You are analyzing a set of logs from a system.
- Total logs in the current filtered view: ${visibleLogs.length.toLocaleString()}
- Total logs across all files: ${allLogs.length.toLocaleString()}
- Available Daemons: ${allDaemons.join(', ') || 'N/A'}
- Time range of visible logs: ${visibleLogs.length > 0 ? new Date(visibleLogs[0].timestamp).toISOString() : 'N/A'} to ${visibleLogs.length > 0 ? new Date(visibleLogs[visibleLogs.length - 1].timestamp).toISOString() : 'N/A'}

# LOG SAMPLES (from current view)
This is a sample of the logs you are analyzing to understand their structure. Do not limit your analysis to only these lines; use the 'search_logs' tool to get more data.
\`\`\`
${sampleLogText || 'No logs in current view.'}
\`\`\`

# PREVIOUSLY IDENTIFIED FINDINGS
The user has saved these findings from previous sessions. These might be relevant.
${savedFindings.length > 0 ? savedFindings.map(f => `- ${f}`).join('\n') : 'None'}

# AVAILABLE TOOLS
You have a set of tools to interact with the log viewer.
- Use 'search_logs' to find specific log entries across ALL logs. This is powerful for finding related events outside the current view.
- Use 'find_log_patterns' to identify trends like repeating errors or spikes in activity.
- Use 'trace_error_origin' AFTER you have found a specific error log ID to see what happened before it.
- Use 'update_filters' to create a new, focused view for the user. This helps them drill down into issues.
- Use 'scroll_to_log' to highlight a specific log entry for the user.
- Use 'suggest_solution' ONLY when asked for help fixing an error.

# RESPONSE GUIDELINES
- Be concise and clear.
- When you find a specific log, ALWAYS mention its ID using the format [Log ID: 123] so the user can click it.
- Think step-by-step. First, understand the user's request. Second, decide which tool(s) to use. Third, analyze the tool output. Finally, formulate a user-facing response.
- If a user's request is ambiguous, ask clarifying questions before using tools.
- Do not invent information. If the logs don't contain the answer, say so.
- When presenting findings, summarize them first, then provide supporting log IDs.
- You can use markdown for formatting (bold, lists, code blocks).
`;

    const history: Content[] = messages.slice(1) // Exclude welcome message
      .flatMap(m => {
          // Heuristic: Don't include error/warning messages in history to prevent confusion
          if (m.isError || m.isWarning) return [];
          
          // Crude attempt to find tool calls/responses in past messages to reconstruct history.
          // This is a simplified approach. A more robust solution would store the structured
          // tool calls and responses separately in the message history state.
          if (m.role === 'model' && m.text.startsWith('Tool Call:')) {
              try {
                  const callText = m.text.substring('Tool Call:'.length).trim();
                  const call = JSON.parse(callText);
                  return [{ role: 'model' as const, parts: [{ functionCall: call }] }];
              } catch {
                  return [{ role: 'model' as const, parts: [{ text: m.text }] }];
              }
          }
          if (m.role === 'model' && m.text.startsWith('Tool Response:')) {
              try {
                  const responseText = m.text.substring('Tool Response:'.length).trim();
                  const response = JSON.parse(responseText);
                  return [{ role: 'tool' as const, parts: [{ functionResponse: response }] }];
              } catch {
                  return [{ role: 'model' as const, parts: [{ text: m.text }] }];
              }
          }

          return [{
              role: m.role,
              parts: [{ text: m.text }],
          }];
      });

    history.push({ role: 'user', parts: [{ text: prompt }] });
    
    const MAX_TURNS = 10;
    for (let turn = 1; turn <= MAX_TURNS; turn++) {
        const modelToUse = modelTier === 'gemini-3-pro-preview' ? 'gemini-3-pro-preview' : 'gemini-2.5-flash';
        
        console.log(`[AI] Turn ${turn}/${MAX_TURNS} using ${modelToUse} in state: ${conversationStateRef.current}`);

        let response: GenerateContentResponse;
        try {
            response = await ai.models.generateContent({
                model: modelToUse,
                contents: history,
                tools: [{ functionDeclarations: getAvailableTools(conversationStateRef.current) }],
                systemInstruction: { role: 'system', parts: [{ text: systemPrompt }] },
            });
        } catch (e: any) {
            console.error("Gemini API Error:", e);
            if (e.message && (e.message.toLowerCase().includes('rate limit') || e.message.toLowerCase().includes('quota'))) {
                 addMessage('model', `I've hit a rate limit while processing your request. This usually means too many requests were sent in a short period.\n\nPlease try again in a few moments. If the problem persists, you may need to check the usage limits on your Google AI API key.`, true);
            } else {
                 addMessage('model', `An error occurred while communicating with the AI model: ${e.message || 'Unknown error'}`, true);
            }
            setIsLoading(false);
            return;
        }

        const functionCalls = response.functionCalls;

        if (functionCalls && functionCalls.length > 0) {
            history.push({ role: 'model', parts: [{ functionCall: functionCalls[0] }] });
            const toolCall = functionCalls[0];
            
            console.log('[AI] Executing tool:', toolCall.name, toolCall.args);

            const toolResponseParts: Part[] = [];

            switch (toolCall.name) {
                case 'update_filters': {
                    const result = handleToolCall(toolCall.name, toolCall.args);
                    toolResponseParts.push({ functionResponse: { name: 'update_filters', response: { result: JSON.stringify(result) } } });
                    break;
                }
                case 'scroll_to_log': {
                    const result = handleToolCall(toolCall.name, toolCall.args);
                    toolResponseParts.push({ functionResponse: { name: 'scroll_to_log', response: { result: JSON.stringify(result) } } });
                    break;
                }
                case 'search_logs': {
                    const result = handleToolCall(toolCall.name, toolCall.args);
                    if (result.logs.length > 0) {
                        conversationStateRef.current = 'ANALYZING';
                        console.log('[AI State] Transitioning to ANALYZING after finding data.');
                    }
                    toolResponseParts.push({ functionResponse: { name: 'search_logs', response: { result: JSON.stringify(result) } } });
                    break;
                }
                case 'find_log_patterns': {
                    const result = handleToolCall(toolCall.name, toolCall.args);
                    toolResponseParts.push({ functionResponse: { name: 'find_log_patterns', response: { result: JSON.stringify(result) } } });
                    break;
                }
                case 'trace_error_origin': {
                    const result = handleToolCall(toolCall.name, toolCall.args);
                    toolResponseParts.push({ functionResponse: { name: 'trace_error_origin', response: { result: JSON.stringify(result) } } });
                    break;
                }
                 case 'suggest_solution': {
                    const result = handleToolCall(toolCall.name, toolCall.args);
                    toolResponseParts.push({ functionResponse: { name: 'suggest_solution', response: { result: JSON.stringify(result) } } });
                    break;
                }
                default:
                    toolResponseParts.push({ functionResponse: { name: toolCall.name, response: { error: `Tool "${toolCall.name}" is not supported.` } } });
                    break;
            }
            history.push({ role: 'tool', parts: toolResponseParts });

        } else if (response.text) {
            const finalText = response.text;
            addMessage('model', finalText);
            conversationStateRef.current = 'IDLE';
            console.log('[AI State] Resetting to IDLE after final answer.');
            break; // Exit the tool-use loop
        } else {
            addMessage('model', "I received an unexpected response from the AI. Please try again.", true);
            break;
        }
    }
    setIsLoading(false);
  }, [userApiKey, visibleLogs, allLogs, allDaemons, messages, modelTier, addMessage, handleToolCall, savedFindings]);
  
  const mlcEngine = useRef<any>(null);

  const runLocalAI = useCallback(async (prompt: string) => {
    if (!mlcEngine.current) {
        addMessage('model', 'Local AI model is not loaded yet. Please wait.', true);
        setIsLoading(false);
        return;
    }

    const localSystemPrompt = `You are a helpful log analysis assistant.
You can use tools by responding with a JSON object with 'tool_name' and 'arguments'.
Available tools:
- search_logs({keywords: string[], match_mode: 'OR'|'AND', limit: number}): Searches all logs.
- find_log_patterns({pattern_type: 'repeating_error'|'frequency_spike', time_window_minutes?: number}): Finds common patterns.
- update_filters({log_levels?: string[], daemons?: string[], search_keywords?: string[]}): Creates a new tab with filters.
Example:
To search for "error" and "database", you would respond with:
\`\`\`json
{"tool_name": "search_logs", "arguments": {"keywords": ["error", "database"], "match_mode": "AND"}}
\`\`\`
Analyze the user's request and respond with ONLY the JSON tool call. Do not add any other text.
`;
    
    let fullResponse = '';
    const MAX_TURNS = 5;
    let currentPrompt = prompt;

    for (let turn = 1; turn <= MAX_TURNS; turn++) {
        const completion = await mlcEngine.current.chat.completions.create({
            messages: [
                { role: "system", content: localSystemPrompt },
                { role: "user", content: currentPrompt },
            ],
            max_tokens: 512,
            temperature: 0.1,
        });
        
        const responseText = completion.choices[0].message.content || '';
        const toolCall = parseLocalToolCall(responseText);
        
        if (toolCall) {
            const result = handleToolCall(toolCall.tool_name, toolCall.arguments);
            const resultString = JSON.stringify(result, null, 2);
            currentPrompt = `I used the tool '${toolCall.tool_name}' with arguments ${JSON.stringify(toolCall.arguments)}. The result was: ${resultString}. Now, based on this result, what is the final answer to the user's original question: "${prompt}"? Provide a concise, user-facing summary. Do not call any more tools.`;
            // In the last turn, we just append the result and let it fall through
            if (turn === MAX_TURNS) {
                fullResponse += `\nAfter analyzing the tool output, here is a summary:\n${resultString}`;
            }
        } else {
            fullResponse = responseText;
            break; // No tool call, so this is the final answer
        }
    }

    addMessage('model', fullResponse || "I couldn't generate a response. Please try rephrasing.");
    setIsLoading(false);

  }, [addMessage, handleToolCall]);

  const loadWebLlm = useCallback(async () => {
    setIsLoading(true);
    try {
        const engine = await CreateMLCEngine(WEB_LLM_MODEL_ID, {
            initProgressCallback: (progress) => {
                setDownloadProgress(`Loading model: ${progress.text}`);
            }
        });
        mlcEngine.current = engine;
        setDownloadProgress('');
        addMessage('model', 'Local AI model (Llama 3) loaded successfully! You can now use it for analysis.');
        // If there was a prompt waiting, execute it now.
        if (pendingPrompt) {
            runLocalAI(pendingPrompt);
            setPendingPrompt(null);
        } else {
            setIsLoading(false);
        }
    } catch (e: any) {
        console.error("WebLLM Error:", e);
        addMessage('model', `Failed to load the local AI model. Your device might not have enough memory or GPU support. Error: ${e.message}`, true);
        setModelTier('gemini-2.5-flash'); // Fallback to cloud
        setIsLoading(false);
        setDownloadProgress('');
    }
  }, [addMessage, runLocalAI, pendingPrompt]);

  const handleConsent = (consented: boolean) => {
      setShowWebLlmConsent(false);
      if (consented) {
          localStorage.setItem(WEBLMM_CONSENT_KEY, 'true');
          loadWebLlm();
      } else {
          setModelTier('gemini-2.5-flash'); // Revert if they decline
          setIsLoading(false);
          if (pendingPrompt) {
              addMessage('model', 'Switched back to cloud model. Please send your request again.', false);
              setPendingPrompt(null);
          }
      }
  };

  const handleSubmit = (e?: React.FormEvent) => {
    e?.preventDefault();
    const trimmedInput = input.trim();
    if (!trimmedInput || isLoading) return;

    addMessage('user', trimmedInput);
    setIsLoading(true);

    if (modelTier === 'web-llm') {
      if (mlcEngine.current) {
        runLocalAI(trimmedInput);
      } else {
        setPendingPrompt(trimmedInput); // Save the prompt
        const hasConsented = localStorage.getItem(WEBLMM_CONSENT_KEY) === 'true';
        if (!hasConsented) {
            setShowWebLlmConsent(true);
        } else {
            loadWebLlm();
        }
      }
    } else {
      runCloudAI(trimmedInput);
    }
    setInput('');
  };
  
  const quickPrompts = [
      "Summarize the most common errors.",
      "Are there any unusual activity spikes?",
      "Show me all logs related to 'database connection'.",
      "What happened around log ID 1234?",
  ];

  return (
    <div className={`absolute inset-0 bg-gray-900/50 backdrop-blur-sm z-40 transition-opacity ${isOpen ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}>
      <div className={`absolute top-0 right-0 h-full w-full max-w-lg bg-gray-800 border-l border-gray-700 shadow-2xl flex flex-col transition-transform duration-300 ease-in-out ${isOpen ? 'translate-x-0' : 'translate-x-full'}`}>
        <div className="flex-shrink-0 flex items-center justify-between p-2 border-b border-gray-700">
          <div className="flex items-center space-x-2">
            <svg className="w-4 h-4 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
            <h2 className="font-bold text-sm text-white">AI Assistant</h2>
          </div>
          <div className="flex items-center space-x-1">
            <button
               onClick={() => setIsSettingsOpen(!isSettingsOpen)}
               className="p-1 text-gray-400 hover:text-white rounded-md hover:bg-gray-700"
               title="Settings"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"></path><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path></svg>
            </button>
            <button onClick={onClose} className="p-1 text-gray-400 hover:text-white rounded-md hover:bg-gray-700" aria-label="Close AI Assistant">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path></svg>
            </button>
          </div>
        </div>

        {isSettingsOpen && (
            <div className="p-3 bg-gray-900/50 border-b border-gray-700 space-y-3 text-xs">
                <div>
                    <label className="font-semibold text-gray-300 block mb-1">AI Model</label>
                    <select value={modelTier} onChange={e => setModelTier(e.target.value)} className="w-full bg-gray-700 text-white rounded py-1 px-2 border border-gray-600 focus:ring-1 focus:ring-blue-500 focus:outline-none">
                        <option value="gemini-2.5-flash">Gemini 2.5 Flash (Cloud)</option>
                        <option value="gemini-3-pro-preview">Gemini 3 Pro (Cloud)</option>
                        <option value="web-llm">Llama 3 (Local)</option>
                    </select>
                    <p className="text-[10px] text-gray-500 mt-1">Local model runs entirely in your browser for maximum privacy. Cloud models are more powerful but send data to Google.</p>
                </div>
                 <div>
                    <label htmlFor="api-key-input" className="font-semibold text-gray-300 block mb-1">Google AI API Key (Optional)</label>
                    <input 
                       id="api-key-input"
                       type="password"
                       value={tempApiKey}
                       onChange={(e) => setTempApiKey(e.target.value)}
                       placeholder="Enter your key to override default"
                       className="w-full bg-gray-700 text-white rounded py-1 px-2 border border-gray-600 focus:ring-1 focus:ring-blue-500 focus:outline-none"
                    />
                     <p className="text-[10px] text-gray-500 mt-1">If you provide your own key, it will be used for all cloud model requests. Stored in local storage.</p>
                </div>
                <div className="flex justify-end">
                    <button onClick={handleSaveSettings} className="bg-blue-600 text-white px-3 py-1 rounded text-xs hover:bg-blue-700">Save</button>
                </div>
            </div>
        )}
        
        {showWebLlmConsent && (
            <div className="p-3 bg-yellow-900/50 border-b border-yellow-700 space-y-2 text-xs">
                <p className="font-semibold text-yellow-200">Local Model Download</p>
                <p className="text-yellow-300 text-[10px]">The Llama 3 model is ~2GB and needs to be downloaded and cached by your browser. This is a one-time download per device. Do you want to proceed?</p>
                <div className="flex justify-end space-x-2">
                    <button onClick={() => handleConsent(false)} className="bg-gray-600 text-white px-2 py-0.5 rounded text-xs hover:bg-gray-500">Cancel</button>
                    <button onClick={() => handleConsent(true)} className="bg-yellow-600 text-white px-2 py-0.5 rounded text-xs hover:bg-yellow-700">Download</button>
                </div>
            </div>
        )}

        <div className="flex-grow p-3 overflow-y-auto space-y-4">
          {messages.map((message, index) => (
            <div key={message.id} className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`relative max-w-sm md:max-w-md p-2 rounded-lg text-white ${message.role === 'user' ? 'bg-blue-600' : (message.isError ? 'bg-red-800' : 'bg-gray-700')}`}>
                 {message.role === 'model' && index > 0 && !message.isError && (
                    <div className="absolute top-0 right-0 flex -translate-y-1/2 translate-x-1/2 space-x-0.5">
                       {savedFindings.includes(message.text) ? (
                            <div className="p-0.5 rounded-full bg-green-600 text-white" title="Finding Saved">
                                <svg className="w-2 h-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="3" d="M5 13l4 4L19 7"></path></svg>
                            </div>
                       ) : (
                            <button
                                onClick={() => onSaveFinding(message.text)}
                                className="p-0.5 rounded-full text-gray-400 bg-gray-800 border border-gray-600 hover:text-white hover:bg-gray-600"
                                title="Save this finding"
                            >
                                <svg className="w-2 h-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z"></path></svg>
                            </button>
                       )}
                    </div>
                 )}
                 <FormattedMessage text={message.text} onScrollToLog={onScrollToLog} />
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="flex justify-start">
              <div className="max-w-sm md:max-w-md p-2 rounded-lg bg-gray-700 text-white">
                <div className="flex items-center space-x-2 text-xs">
                    <div className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-pulse" style={{ animationDelay: '0s' }}></div>
                    <div className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-pulse" style={{ animationDelay: '0.2s' }}></div>
                    <div className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-pulse" style={{ animationDelay: '0.4s' }}></div>
                    {downloadProgress && <span className="text-gray-400 text-[10px]">{downloadProgress}</span>}
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
        
        <div className="flex-shrink-0 p-2 border-t border-gray-700 bg-gray-800 space-y-1">
            <div className="flex flex-wrap gap-1">
                {quickPrompts.map(p => (
                    <button 
                       key={p}
                       onClick={() => { setInput(p); setTimeout(() => handleSubmit(), 50); }}
                       disabled={isLoading}
                       className="px-1.5 py-0.5 bg-gray-700 hover:bg-gray-600 text-gray-300 rounded-full text-[10px] transition-colors disabled:opacity-50"
                    >
                        {p}
                    </button>
                ))}
            </div>
            <form onSubmit={handleSubmit} className="flex items-center space-x-2">
              <textarea
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={e => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSubmit();
                  }
                }}
                placeholder="Ask about your logs..."
                disabled={isLoading}
                rows={1}
                className="flex-grow bg-gray-700 border border-gray-600 text-white text-xs rounded-md shadow-sm p-2 focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-800 resize-none"
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
      </div>
    </div>
  );
};
