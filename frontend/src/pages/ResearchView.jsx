import React, { useState, useRef, useEffect } from "react";
import client from "../api/client";
import { motion, AnimatePresence } from "framer-motion";
import { Send, Loader2, FileText, Zap, User, Hexagon, ExternalLink } from "lucide-react";
import ReactMarkdown from "react-markdown";

function CitationCard({ citation, index }) {
  const isUrl = !!citation.url;
  const linkText = isUrl ? (citation.title || citation.source || "Web Page") : `Page ${citation.page_num || "Web Page"}`;
  const icon = isUrl ? <ExternalLink className="w-4 h-4 text-zinc-400" /> : <FileText className="w-4 h-4 text-zinc-400" />;

  return (
    <motion.div
      whileHover={{ y: -2, rotateX: 2, scale: 1.02 }}
      className="bg-zinc-900/40 border border-white/5 p-3 rounded-xl cursor-default transition-all shadow-sm hover:shadow-lg hover:border-white/10 group flex flex-col gap-1 w-[220px] flex-shrink-0"
    >
      <div className="flex items-center gap-2 mb-1">
        <span className="inline-flex items-center justify-center w-5 h-5 rounded-full bg-zinc-800 text-[10px] text-zinc-400 font-medium">
          {index + 1}
        </span>
        <div className="flex-1 truncate text-xs font-medium text-zinc-300">
          {citation.source ? citation.source.split("/").pop() : "Unknown Source"}
        </div>
      </div>
      <div className="flex items-center gap-1.5 mt-auto">
        {icon}
        {isUrl ? (
          <a
            href={citation.url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-[11px] text-zinc-400 hover:text-white truncate"
          >
            {linkText}
          </a>
        ) : (
          <span className="text-[11px] text-zinc-500">{linkText}</span>
        )}
      </div>
    </motion.div>
  );
}

export default function ResearchView() {
  const [messages, setMessages] = useState(() => {
    const saved = sessionStorage.getItem("hermes_chat_history");
    return saved ? JSON.parse(saved) : [];
  });
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => {
    sessionStorage.setItem("hermes_chat_history", JSON.stringify(messages));
  }, [messages]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const query = input.trim();
    setInput("");
    
    const userMsg = { role: "user", content: query };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    try {
      const res = await client.post("/research", { query });
      const aiMsg = { 
        role: "assistant", 
        content: res.data.answer, 
        citations: res.data.citations,
        cache_hit: res.data.cache_hit,
        model: res.data.model_used
      };
      setMessages((prev) => [...prev, aiMsg]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", error: true, content: "Sorry, I encountered an error during research. Please try again." }
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen max-h-screen relative">
      {/* Header */}
      <div className="flex-none p-6 pb-2 pl-8 lg:pl-12">
        <h1 className="text-3xl font-semibold tracking-tight text-white">Research Agent</h1>
        <p className="text-zinc-400 mt-1">Ask questions spanning your ingested knowledge base.</p>
      </div>

      {/* Chat Area */}
      <div className="flex-1 overflow-y-auto px-4 lg:px-12 py-6 flex flex-col gap-8 pb-32">
        {messages.length === 0 && !loading && (
          <div className="flex-1 flex flex-col items-center justify-center text-center opacity-70 mt-20">
            <Hexagon className="w-12 h-12 text-zinc-700 mb-4" />
            <h3 className="text-lg font-medium text-zinc-300">Ready for queries</h3>
            <p className="text-sm text-zinc-500 max-w-sm mt-2">
              The agent uses RAG to fetch context from your uploaded docs and uses reasoning to formulate an answer.
            </p>
          </div>
        )}

        {messages.map((msg, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className={`flex w-full gap-4 ${msg.role === "user" ? "flex-row-reverse" : "flex-row"}`}
          >
            {/* Avatar */}
            <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 mt-1 ${
              msg.role === "user" ? "bg-zinc-800" : "bg-white text-zinc-950"
            }`}>
              {msg.role === "user" ? <User className="w-4 h-4 text-zinc-300" /> : <Hexagon className="w-5 h-5 fill-zinc-950" />}
            </div>

            {/* Bubble Layout */}
            <div className={`flex flex-col max-w-[85%] ${msg.role === "user" ? "items-end" : "items-start"}`}>
              {msg.role === "assistant" && (msg.cache_hit || msg.model) && (
                <div className="flex items-center gap-2 mb-2 px-1">
                  {msg.cache_hit && (
                    <span className="flex items-center gap-1 text-[10px] font-semibold text-amber-400 bg-amber-400/10 px-2 py-0.5 rounded-full border border-amber-400/20 tracking-wider">
                      <Zap className="w-3 h-3" /> CACHE HIT
                    </span>
                  )}
                  {msg.model && (
                    <span className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider">
                      {msg.model}
                    </span>
                  )}
                </div>
              )}
              
              <div
                className={`px-5 py-4 rounded-3xl ${
                  msg.role === "user"
                    ? "bg-zinc-800/80 text-white rounded-tr-sm"
                    : msg.error
                    ? "bg-red-500/10 border border-red-500/20 text-red-200"
                    : "bg-transparent text-zinc-200"
                }`}
              >
                {msg.role === "assistant" && !msg.error ? (
                  <div className="prose prose-invert prose-zinc max-w-none text-[15px] leading-relaxed prose-p:my-2 prose-pre:bg-zinc-900/50 prose-pre:border prose-pre:border-white/5">
                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                  </div>
                ) : (
                  <div className="text-[15px] leading-relaxed whitespace-pre-wrap">{msg.content}</div>
                )}
              </div>

              {/* Citations Row */}
              {msg.role === "assistant" && msg.citations && msg.citations.length > 0 && (
                <div className="mt-4 w-full">
                  <div className="text-xs font-semibold text-zinc-500 uppercase tracking-wider mb-2 px-2">Sources</div>
                  <div className="flex gap-3 overflow-x-auto pb-4 px-1 snap-x scrollbar-hide">
                    {msg.citations.map((cit, idx) => (
                      <div key={idx} className="snap-start pt-1">
                        <CitationCard citation={cit} index={idx} />
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        ))}

        {loading && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex gap-4">
            <div className="w-8 h-8 rounded-full bg-white text-zinc-950 flex items-center justify-center flex-shrink-0 mt-1">
               <Hexagon className="w-5 h-5 fill-zinc-950 animate-pulse" />
            </div>
            <div className="px-5 py-5 flex items-center gap-2">
               <div className="w-1.5 h-1.5 rounded-full bg-zinc-500 animate-bounce [animation-delay:-0.3s]"></div>
               <div className="w-1.5 h-1.5 rounded-full bg-zinc-500 animate-bounce [animation-delay:-0.15s]"></div>
               <div className="w-1.5 h-1.5 rounded-full bg-zinc-500 animate-bounce"></div>
            </div>
          </motion.div>
        )}
        <div ref={bottomRef} className="h-4 w-full" />
      </div>

      {/* Floating Input area */}
      <div className="absolute bottom-0 left-0 right-0 p-4 lg:p-8 bg-gradient-to-t from-zinc-950 via-zinc-950/80 to-transparent pointer-events-none flex justify-center">
        <div className="w-full max-w-3xl pointer-events-auto">
          <form
            onSubmit={handleSubmit}
            className="flex items-center gap-2 glass p-2 rounded-2xl shadow-2xl relative transition-all focus-within:ring-1 focus-within:ring-white/20"
          >
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask anything..."
              className="flex-1 bg-transparent border-none text-white px-4 py-3 text-[15px] focus:outline-none focus:ring-0 placeholder:text-zinc-500 disabled:opacity-50"
              disabled={loading}
              autoFocus
            />
            <button
              type="submit"
              disabled={!input.trim() || loading}
              className="w-12 h-12 rounded-xl bg-white text-zinc-950 flex items-center justify-center hover:bg-zinc-200 transition-colors disabled:opacity-50 disabled:bg-zinc-800 disabled:text-zinc-500 shadow-sm"
            >
              {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Send className="w-5 h-5 ml-0.5" />}
            </button>
          </form>
          <div className="text-center mt-3 text-xs text-zinc-600 font-medium tracking-wide">
            Hermes can make mistakes. Consider verifying important information.
          </div>
        </div>
      </div>
    </div>
  );
}
