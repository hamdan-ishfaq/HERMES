import React, { useState } from "react";
import { Link, Globe, FileText, UploadCloud, CheckCircle2, Loader2 } from "lucide-react";
import client from "../api/client";
import { motion, AnimatePresence } from "framer-motion";

function IngestCard({ title, icon: Icon, children }) {
  return (
    <div className="glass-card p-6 relative overflow-hidden group">
      <div className="absolute inset-0 bg-white/[0.02] opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none" />
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2.5 bg-white/10 rounded-xl border border-white/5">
          <Icon className="w-5 h-5 text-zinc-300" />
        </div>
        <h3 className="font-medium text-lg text-white tracking-tight">{title}</h3>
      </div>
      {children}
    </div>
  );
}

export default function KnowledgeBaseView() {
  const [url, setUrl] = useState("");
  const [pdfFile, setPdfFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  
  const [loadingType, setLoadingType] = useState(null); // 'url' | 'pdf'
  const [successMsg, setSuccessMsg] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);

  const showNotification = (msg) => {
    setSuccessMsg(msg);
    setTimeout(() => setSuccessMsg(null), 5000);
  };

  const handleUrlSubmit = async (e) => {
    e.preventDefault();
    if (!url) return;
    setLoadingType("url");
    setErrorMsg(null);
    try {
      const endpoint = url.includes("youtube.com") || url.includes("youtu.be") 
        ? "/ingest/youtube" 
        : "/ingest/url";
      const res = await client.post(endpoint, { url });
      showNotification(`Successfully ingested URL. ${res.data.chunks_stored} chunks stored.`);
      setUrl("");
    } catch (err) {
      setErrorMsg(err.response?.data?.detail || "Failed to ingest URL");
    } finally {
      setLoadingType(null);
    }
  };

  const handlePdfSubmit = async (e) => {
    e.preventDefault();
    if (!pdfFile) return;
    setLoadingType("pdf");
    setErrorMsg(null);
    
    const formData = new FormData();
    formData.append("file", pdfFile);

    try {
      const res = await client.post("/ingest/pdf", formData, {
        headers: { "Content-Type": "multipart/form-data" }
      });
      showNotification(`Successfully ingested PDF. ${res.data.chunks_stored} chunks stored.`);
      setPdfFile(null);
    } catch (err) {
      setErrorMsg(err.response?.data?.detail || "Failed to ingest PDF");
    } finally {
      setLoadingType(null);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    
    const file = e.dataTransfer.files[0];
    if (file && file.type === "application/pdf") {
      setPdfFile(file);
    } else if (file) {
      showNotification("Only PDF files are supported.");
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-8 lg:p-12 w-full">
      <div className="mb-10">
        <h1 className="text-3xl font-semibold tracking-tight text-white mb-2">Knowledge Base</h1>
        <p className="text-zinc-400">Add documents and URLs to expand the agent's context.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* URL Ingestion */}
        <IngestCard title="Web & YouTube URL" icon={Globe}>
          <form onSubmit={handleUrlSubmit} className="space-y-4">
            <input
              type="url"
              placeholder="https://example.com/article"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              className="w-full input-base"
              required
            />
            <button
              type="submit"
              disabled={loadingType !== null}
              className="w-full btn-secondary flex justify-center items-center gap-2"
            >
              {loadingType === "url" ? (
                <Loader2 className="w-5 h-5 animate-spin text-zinc-400" />
              ) : (
                "Ingest URL"
              )}
            </button>
          </form>
        </IngestCard>

        {/* PDF Ingestion */}
        <IngestCard title="Local Document (PDF)" icon={FileText}>
          <form onSubmit={handlePdfSubmit} className="space-y-4">
            <label 
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              className={`flex flex-col items-center justify-center w-full h-32 border-2 border-dashed rounded-xl cursor-pointer transition-colors relative overflow-hidden group ${isDragging ? 'border-emerald-500/50 bg-emerald-500/5' : 'border-white/10 hover:border-white/20 hover:bg-white/[0.02]'}`}
            >
              <div className="flex flex-col items-center justify-center pt-5 pb-6 text-center px-4">
                <UploadCloud className={`w-6 h-6 mb-2 transition-colors ${isDragging ? 'text-emerald-400' : 'text-zinc-400 group-hover:text-zinc-200'}`} />
                <p className={`text-sm font-medium ${isDragging ? 'text-emerald-300' : 'text-zinc-400'}`}>
                  {pdfFile ? pdfFile.name : (isDragging ? "Drop PDF here" : "Click to upload or drag & drop")}
                </p>
                {!pdfFile && <p className="text-xs text-zinc-500 mt-1">PDF max 50MB</p>}
              </div>
              <input
                type="file"
                accept=".pdf"
                className="hidden"
                onChange={(e) => setPdfFile(e.target.files[0])}
              />
            </label>
            <button
              type="submit"
              disabled={loadingType !== null || !pdfFile}
              className="w-full btn-secondary flex justify-center items-center gap-2"
            >
              {loadingType === "pdf" ? (
                <Loader2 className="w-5 h-5 animate-spin text-zinc-400" />
              ) : (
                "Ingest PDF"
              )}
            </button>
          </form>
        </IngestCard>
      </div>

      <AnimatePresence>
        {successMsg && (
          <motion.div
            initial={{ opacity: 0, y: 50, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 20, scale: 0.95 }}
            className="fixed bottom-10 right-10 glass px-6 py-4 rounded-2xl flex items-center gap-3 shadow-[0_20px_40px_-15px_rgba(0,0,0,0.5)] z-50 pointer-events-none"
          >
            <CheckCircle2 className="w-5 h-5 text-emerald-400" />
            <span className="text-sm font-medium text-white">{successMsg}</span>
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {errorMsg && (
          <motion.div
            initial={{ opacity: 0, y: 50, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 20, scale: 0.95 }}
            className="fixed bottom-10 right-10 bg-red-500/10 border border-red-500/20 px-6 py-4 rounded-2xl flex items-center gap-3 backdrop-blur-md z-50 pointer-events-none"
          >
            <span className="text-sm font-medium text-red-400">{errorMsg}</span>
          </motion.div>
        )}
      </AnimatePresence>

    </div>
  );
}
