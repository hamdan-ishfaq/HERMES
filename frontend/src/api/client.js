/**
 * Shared Axios HTTP client for all backend API calls.
 *
 * Role in the UI:
 *   - Single configured instance used by AuthContext, ResearchView, KnowledgeBaseView,
 *     and AnalyticsView so auth headers and base URL are applied consistently.
 *
 * API base: `http://localhost:8000/api`
 *
 * Endpoints consumed (by callers, not this file):
 *   - POST /auth/login, /auth/register
 *   - POST /research
 *   - POST /ingest/url, /ingest/youtube, /ingest/pdf
 *   - GET  /eval/dashboard
 */

import axios from "axios";

/** Pre-configured Axios instance pointing at the FastAPI backend. */
const client = axios.create({
  baseURL: "http://localhost:8000/api",
});

/**
 * Request interceptor — attaches the JWT from localStorage to every outgoing call.
 * The backend expects `Authorization: Bearer <token>` on protected routes.
 */
client.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem("hermes_access_token");
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

export default client;
