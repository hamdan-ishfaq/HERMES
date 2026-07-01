/**
 * Public authentication page — toggles between Sign In and Sign Up forms.
 *
 * Role in the UI:
 *   - Rendered at `/auth` for unauthenticated users (see App.jsx).
 *   - Collects email + password and delegates to AuthContext `login` / `register`.
 *   - Displays API validation errors returned in `response.data.detail`.
 *
 * API endpoints (via AuthContext → api/client.js):
 *   - POST /auth/login
 *   - POST /auth/register
 */

import React, { useState } from "react";
import { useAuth } from "../../context/AuthContext";
import { motion, AnimatePresence } from "framer-motion";
import { Hexagon, Loader2, ArrowRight } from "lucide-react";

/**
 * Dual-mode auth form with animated card layout.
 * Successful auth redirects automatically because App.jsx watches `isAuthenticated`.
 */
export default function LoginRegister() {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const { login, register } = useAuth();

  /**
   * Submit handler — routes to login or register based on `isLogin` toggle.
   * Surfaces FastAPI `detail` strings on 4xx responses.
   */
  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      if (isLogin) {
        await login(email, password);
      } else {
        await register(email, password);
      }
    } catch (err) {
      if (err.response && err.response.data && err.response.data.detail) {
        setError(err.response.data.detail);
      } else {
        setError("An unexpected error occurred.");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-zinc-950 flex items-center justify-center p-4 relative overflow-hidden">
      {/* Decorative background glow */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-white/[0.02] rounded-full blur-3xl pointer-events-none" />

      {/* Auth card — logo, heading, form */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
        className="w-full max-w-sm glass p-8 rounded-[2rem] relative z-10"
      >
        {/* Logo */}
        <div className="flex justify-center mb-8 text-white">
          <div className="p-3 bg-white/10 rounded-2xl border border-white/10 shadow-2xl">
            <Hexagon className="w-8 h-8 text-zinc-100" />
          </div>
        </div>

        {/* Heading — copy changes with login vs. register mode */}
        <div className="text-center mb-8">
          <h2 className="text-2xl font-semibold text-white tracking-tight mb-2">
            {isLogin ? "Welcome back" : "Create an account"}
          </h2>
          <p className="text-zinc-400 text-sm">
            {isLogin
              ? "Enter your credentials to access your research."
              : "Sign up to start querying the knowledge base."}
          </p>
        </div>

        {/* Credentials form */}
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-zinc-400 ml-1">Email</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              className="w-full input-base"
              placeholder="name@example.com"
            />
          </div>

          <div className="space-y-1.5">
            <label className="text-xs font-medium text-zinc-400 ml-1">Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              className="w-full input-base"
              placeholder="••••••••"
            />
          </div>

          {/* Inline API error banner */}
          {error && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              className="text-red-400 text-sm text-center bg-red-500/10 py-2 rounded-lg border border-red-500/20"
            >
              {error}
            </motion.div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full btn-primary flex items-center justify-center gap-2 mt-4"
          >
            {loading ? (
              <Loader2 className="w-5 h-5 animate-spin text-zinc-500" />
            ) : (
              <>
                {isLogin ? "Sign In" : "Sign Up"}
                <ArrowRight className="w-4 h-4" />
              </>
            )}
          </button>
        </form>

        {/* Mode toggle — switch between login and register without changing route */}
        <div className="mt-8 text-center text-sm text-zinc-500">
          {isLogin ? "Don't have an account?" : "Already have an account?"}{" "}
          <button
            onClick={() => {
              setIsLogin(!isLogin);
              setError("");
            }}
            className="text-zinc-300 hover:text-white transition-colors font-medium border-b border-transparent hover:border-white/50 pb-0.5"
          >
            {isLogin ? "Sign up" : "Sign in"}
          </button>
        </div>
      </motion.div>
    </div>
  );
}
