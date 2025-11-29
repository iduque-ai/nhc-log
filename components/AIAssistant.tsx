import React, { useState, useRef, useEffect } from 'react';
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
}

// Extend Window interface for Chrome's Built-in AI
declare global {
  interface Window {
    ai?: {
      languageModel: {
        capabilities: () => Promise<{ available: 'readily' | 'after-download' | 'no' }>;
        create: (options?: { systemPrompt?: string }) => Promise<{
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

// --- Formatted Message Component ---

// Helper to recursively render inline markdown
const renderInlineMarkdown = (text: string, onScrollToLog: (id: number) => void): React.ReactNode => {
    // Split by [Log ID: 123]
    const parts = text.split(/(\[Log ID: \d+\])/g);

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

export const AIAssistant: React.FC<AIAssistantProps> = ({ isOpen, onClose, visibleLogs, allLogs, allDaemons, onUpdateFilters, onScrollToLog }) => {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([
    { 
      id: 'welcome', 
      role: 'model', 
      text: 'Hello! I can help you analyze these logs. Ask me questions like "Why did the charger fail?", "Search for wifi errors", or "Summarize the errors from yesterday".' 
    }
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState<string>('');
  
  // Default to Flash as it's balanced
  const [modelTier, setModelTier] = useState<string>('gemini-2.5-flash');
  const messagesEndRef = useRef<HTMLDivElement>(null);

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
  };

  // Ref to hold the WebLLM engine instance to avoid re-init
  const webLlmEngineRef = useRef<any>(null);

  const getSystemInstruction = () => ({
        role: 'user',
        parts: [{ text: `
System Instruction: 
You are an expert log analyst. Your goal is to help the user understand their application logs.

BEHAVIOR RULES:
1. **Analyze First**: Always use the \`search_logs\` tool to find relevant information before answering. Do not guess.
2. **Expand Search**: When searching, generate SYNONYMS and RELATED TERMS.
   - User: "Why is the charger failing?"
   - You: Search for ["charger", "charging", "voltage", "current", "battery", "power"] with mode "OR".
3. **Strict Formatting**: 
   - When referring to a specific log line, YOU MUST use the format: [Log ID: <number>]. This creates a clickable link.
   - Example: "The error started at [Log ID: 450] and cascaded to [Log ID: 455]."
4. **ALWAYS RESPOND**: After executing a tool (like \`scroll_to_log\` or \`update_filters\`), you MUST provide a text response explaining what you did. Never return an empty response.
5. **New Tabs**: If the user asks to filter the view (e.g., "show me only errors"), use \`update_filters\`. This will create a NEW tab. Tell the user you have opened a new tab with their requested data.

RESPONSE STYLE:
- **Interpret, Don't Just List**: Provide a narrative summary of events. Explain the context behind the logs rather than just reading them out.
- **Be Conversational**: Write naturally. Use paragraphs and bullet points for readability. Avoid being overly brief or robotic.
- **Cite Evidence**: Weave [Log ID: <id>] references into your sentences as proof for your analysis.

Available Tools:
- \`search_logs\`: Search the ENTIRE log file. Use 'OR' mode for synonyms.
- \`scroll_to_log\`: Jump the user's view to a specific line.
- \`update_filters\`: Create a NEW tab with specific filters.

When you find a "smoking gun" or root cause log, use \`scroll_to_log\` to show it to the user immediately.
` }]
  });

  const chatHistoryRef = useRef<Content[]>([
    getSystemInstruction() as Content,
    {
        role: 'model',
        parts: [{ text: "Understood. I will analyze logs using tools, strict [Log ID: <id>] formatting, and synonym-based search. I will provide conversational, interpretive responses while citing evidence." }]
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
        getSystemInstruction() as Content,
        {
            role: 'model',
            parts: [{ text: "Understood. I will analyze logs using tools, strict [Log ID: <id>] formatting, and synonym-based search. I will provide conversational, interpretive responses while citing evidence." }]
        }
      ];
  };

  // Helper to execute client-side tools
  const executeTool = (name: string, args: any): any => {
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

        // Step 1: Frequency Analysis
        const keywordCounts: Record<string, number> = {};
        keywords.forEach(k => keywordCounts[k] = 0);

        // Pre-scan to build counts
        for (const log of allLogs) {
             const text = (log.message + " " + log.daemon + " " + log.level + " " + log.timestamp.toISOString()).toLowerCase();
             for (const k of keywords) {
                 if (text.includes(k)) {
                     keywordCounts[k]++;
                 }
             }
        }

        // Step 2: Calculate Relevance Weights
        const keywordWeights: Record<string, number> = {};
        keywords.forEach(k => {
            keywordWeights[k] = 1000 / (keywordCounts[k] + 1); // Normalized scale
        });

        // Step 3: Score Logs
        interface ScoredLog {
            log: LogEntry;
            score: number;
        }
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
            
            if (isMatch) {
                scoredMatches.push({ log, score: logScore });
            }
        }

        // Step 4: Sort by Score (Descending) and Limit
        scoredMatches.sort((a, b) => b.score - a.score);
        const topMatches = scoredMatches.slice(0, limit).map(m => m.log);

        console.log(`[AI] Weighted Search [${keywords.join(', ')}]. Found ${scoredMatches.length}. Showing top ${topMatches.length}.`);

        if (topMatches.length === 0) {
            return { result: `No logs found matching terms: [${keywords.join(', ')}] with mode ${matchMode}.` };
        }

        return {
            count: topMatches.length,
            total_logs_searched: allLogs.length,
            total_matches_found: scoredMatches.length,
            note: `Showing top ${limit} most relevant matches. Results prioritized by keyword rarity (rare terms rank higher).`,
            logs: topMatches.map(l => ({
                id: l.id,
                timestamp: l.timestamp.toISOString(),
                level: l.level,
                daemon: l.daemon,
                message: l.message
            }))
        };
    }

    return { error: `Unknown tool: ${name}` };
  };

  const processLocalAI = async (userText: string) => {
      try {
          if (!window.ai?.languageModel) {
            throw new Error("Local AI (Gemini Nano) is not available in this browser. This feature requires the latest Chrome (Canary/Dev) with the 'Prompt API for Gemini Nano' flag enabled.");
          }
          
          const session = await window.ai.languageModel.create({
              systemPrompt: "You are a log analysis assistant. Answer briefly based on the logs provided. Tools are NOT available in this mode. Do not ask to filter or scroll, just analyze."
          });
          
          // Local AI context window is small. We can only pass a summary of visible logs.
          const visibleLogSummary = visibleLogs.slice(0, 30).map(l => 
              `${l.timestamp.toISOString()} [${l.level}] ${l.daemon}: ${l.message}`
          ).join('\n');

          const prompt = `
Context (First 30 visible logs):
${visibleLogSummary}

User Question: ${userText}
          `;

          const result = await session.prompt(prompt);
          
          // Cleanup
          session.destroy();
          
          setMessages(prev => [...prev, { id: Date.now().toString(), role: 'model', text: result + "\n\n*(Generated locally on-device)*" }]);

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
      }
  };

  const processWebLLM = async (userText: string) => {
    try {
        if (!webLlmEngineRef.current) {
             setDownloadProgress("Initializing engine...");
             const engine = await CreateMLCEngine(
                 WEB_LLM_MODEL_ID,
                 {
                     initProgressCallback: (report) => {
                         setDownloadProgress(report.text);
                     }
                 }
             );
             webLlmEngineRef.current = engine;
        }

        setDownloadProgress(""); // Clear progress after init

        // WebLLM Context Preparation
        const visibleLogSummary = visibleLogs.slice(0, 50).map(l => 
            `${l.timestamp.toISOString()} [${l.level}] ${l.daemon}: ${l.message}`
        ).join('\n');

        const systemPrompt = `You are a log analysis assistant. Answer based on the logs provided. Be concise.`;
        const contextPrompt = `Here are the first 50 visible logs for context:\n${visibleLogSummary}`;

        const messagesForWebLLM: any[] = [
            { role: "system", content: systemPrompt },
            // Add previous messages (simplified)
            ...messages.filter(m => !m.isError && m.id !== 'welcome').map(m => ({ 
                role: m.role === 'model' ? 'assistant' : 'user', 
                content: m.text 
            })),
            { role: "user", content: `Context:\n${contextPrompt}\n\nQuestion: ${userText}` }
        ];
        
        // Keep history manageable
        if (messagesForWebLLM.length > 10) {
             messagesForWebLLM.splice(1, messagesForWebLLM.length - 10);
        }

        const reply = await webLlmEngineRef.current.chat.completions.create({
            messages: messagesForWebLLM,
            temperature: 0.5,
            max_tokens: 1024,
        });

        const replyText = reply.choices[0].message.content || "No response generated.";
        setMessages(prev => [...prev, { id: Date.now().toString(), role: 'model', text: replyText + "\n\n*(Generated locally via WebLLM)*" }]);

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
        setIsLoading(true);
        processWebLLM(userText);
        return;
    }

    // Determine API Key: Prioritize VITE_API_KEY (set at build), fallback to user localStorage
    const effectiveApiKey = import.meta.env.VITE_API_KEY || userApiKey;

    if (!effectiveApiKey) {
      setMessages(prev => [...prev, { 
          id: Date.now().toString(), 
          role: 'model', 
          text: 'Error: API Key is missing. Please click the Settings icon (⚙️) above to enter your Google Gemini API Key.', 
          isError: true 
      }]);
      setIsSettingsOpen(true); // Open settings to prompt user
      return;
    }

    setIsLoading(true);
    // Capture the history state before adding the new message.
    // If an error occurs, we rollback to this index to clean the context.
    const historyStartIndex = chatHistoryRef.current.length;

    try {
      const ai = new GoogleGenAI({ apiKey: effectiveApiKey });
      // Use the selected model tier directly as the model name
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
                tools: [{ functionDeclarations: [updateFiltersTool, scrollToLogTool, searchLogsTool] }],
            },
        });

        const responseContent = result.candidates?.[0]?.content;
        if (!responseContent) throw new Error("No content in response");

        chatHistoryRef.current.push(responseContent);

        const functionCalls = result.functionCalls;
        
        if (functionCalls && functionCalls.length > 0) {
             const toolResponses: Part[] = [];

             for (const call of functionCalls) {
                 const toolResult = executeTool(call.name, call.args);
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
      
      // Rollback history to clean up the failed interaction
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
      
      // Find the last user message to retry
      const lastUserMsg = [...messages].reverse().find(m => m.role === 'user');
      
      if (lastUserMsg) {
          // Remove the error message from the UI
          setMessages(prev => prev.filter(m => m.id !== errorMsgId));
          // Re-process the last user message
          processUserMessage(lastUserMsg.text);
      }
  };

  const handleQuickAction = (action: 'summarize' | 'errors') => {
    if (isLoading) return;
    
    let prompt = "";
    if (action === 'summarize') prompt = "Summarize the key events in the current visible logs.";
    if (action === 'errors') prompt = "Find critical failures in the entire log file and explain the root cause.";

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
            disabled={isLoading || visibleLogs.length === 0 || (modelTier !== 'gemini-2.5-flash' && modelTier !== 'gemini-3-pro-preview' && modelTier !== 'gemini-flash-lite-latest')}
            className="flex items-center justify-center space-x-1 bg-gray-700 hover:bg-gray-600 text-gray-200 py-2 px-3 rounded text-xs font-medium transition-colors disabled:opacity-50"
            title={modelTier === 'webllm' || modelTier === 'local' ? 'Tools unavailable in local mode' : ''}
        >
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 011.414.586l5.414 5.414a2 2 0 01.586 1.414V19a2 2 0 01-2 2z"></path></svg>
            <span>Summarize View</span>
        </button>
        <button 
            onClick={() => handleQuickAction('errors')}
            disabled={isLoading || (modelTier !== 'gemini-2.5-flash' && modelTier !== 'gemini-3-pro-preview' && modelTier !== 'gemini-flash-lite-latest')}
            className="flex items-center justify-center space-x-1 bg-gray-700 hover:bg-gray-600 text-gray-200 py-2 px-3 rounded text-xs font-medium transition-colors disabled:opacity-50"
            title={modelTier === 'webllm' || modelTier === 'local' ? 'Tools unavailable in local mode' : ''}
        >
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path></svg>
            <span>Analyze Errors</span>
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
                    : 'bg-gray-700 text-gray-200 border border-gray-600'
            }`}>
              <div className="flex items-start">
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
                             modelTier === 'local' ? 'Processing locally (Nano)...' : 
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
                placeholder={(modelTier === 'local' || modelTier === 'webllm') ? "Ask (Tools unavailable in Local mode)..." : "Message AI..."}
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
    </>
  );
};