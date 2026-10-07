import React, { useState } from 'react';
import { useAuth } from '../context/authContext';

export default function E2EView() {
  const { token } = useAuth();
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const runE2E = async (skipQuality = false) => {
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const res = await fetch(`/api/e2e/run?skip_quality=${skipQuality}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'E2E run failed');
      }
      setResult(data);
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full p-6 max-w-6xl mx-auto">
      <div className="mb-6">
        <h1 className="text-2xl font-semibold text-white mb-2">End-to-End Tests</h1>
        <p className="text-zinc-400">Run comprehensive E2E test suite (35+ checks). Admin access required.</p>
      </div>
      <div className="flex gap-3 mb-6">
        <button
          onClick={() => runE2E(false)}
          disabled={loading}
          className="px-4 py-2 bg-white text-black rounded-lg font-medium hover:bg-gray-100 disabled:opacity-50"
        >
          {loading ? 'Running E2E...' : 'Run Full E2E Suite'}
        </button>
        <button
          onClick={() => runE2E(true)}
          disabled={loading}
          className="px-4 py-2 bg-zinc-800 text-white rounded-lg font-medium hover:bg-zinc-700 disabled:opacity-50 border border-zinc-700"
        >
          {loading ? 'Running...' : 'Run E2E (Skip Quality)'}
        </button>
      </div>
      {error && (
        <div className="bg-red-950/30 border border-red-900/50 rounded-xl p-4 mb-6">
          <p className="text-red-400 font-medium">{error}</p>
        </div>
      )}
      {result && (
        <div className="bg-zinc-900/50 border border-white/5 rounded-xl p-6 overflow-auto">
          <div className="mb-4">
            <h2 className="text-lg font-semibold text-white mb-2">Results</h2>
            <p className="text-zinc-400">Return code: {result.return_code}</p>
          </div>
          <pre className="text-xs text-zinc-300 whitespace-pre-wrap font-mono max-h-[70vh] overflow-auto">
            {result.output}
          </pre>
        </div>
      )}
    </div>
  );
}
