/**
 * Root application shell — routing, authentication gating, and page layout.
 *
 * Role in the UI:
 *   - Wraps the entire app in `AuthProvider` so every route can read login state.
 *   - Defines public (`/auth`) vs. protected routes (`/`, `/knowledge-base`, `/analytics`).
 *   - Renders the persistent `Sidebar` + scrollable main content area for logged-in users.
 *   - Lazy-loads page views to keep the initial bundle small.
 *
 * API endpoints: none directly — pages call the API via `api/client.js`.
 */

import React from "react";
import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider, useAuth } from "./context/AuthContext";
import Sidebar from "./components/Sidebar";
import LoginRegister from "./components/Auth/LoginRegister";

// Lazy loading views for better structure
const KnowledgeBaseView = React.lazy(() => import("./pages/KnowledgeBaseView"));
const ResearchView = React.lazy(() => import("./pages/ResearchView"));
const AnalyticsView = React.lazy(() => import("./pages/AnalyticsView"));

/**
 * Route guard — renders `children` only when the user holds a valid JWT.
 * Unauthenticated visitors are redirected to `/auth`.
 *
 * @param {{ children: React.ReactNode }} props
 */
function PrivateRoute({ children }) {
  const { isAuthenticated } = useAuth();
  return isAuthenticated ? children : <Navigate to="/auth" />;
}

/**
 * Authenticated shell layout — fixed sidebar + scrollable main pane.
 *
 * @param {{ children: React.ReactNode }} props
 */
function MainLayout({ children }) {
  return (
    <div className="flex min-h-screen bg-zinc-950 text-zinc-100">
      <Sidebar />
      <main className="flex-1 overflow-y-auto relative">
        {children}
      </main>
    </div>
  );
}

/**
 * Declares all application routes and wires auth redirects.
 * Uses `React.Suspense` around lazy pages to show a loading fallback.
 */
function AppRoutes() {
  const { isAuthenticated } = useAuth();

  return (
    <Routes>
      {/* Public auth page — bounce to home if already logged in */}
      <Route
        path="/auth"
        element={!isAuthenticated ? <LoginRegister /> : <Navigate to="/" />}
      />

      {/* Research chat — default landing page */}
      <Route
        path="/"
        element={
          <PrivateRoute>
            <MainLayout>
              <React.Suspense fallback={<div className="flex items-center justify-center h-full"><span className="text-zinc-500">Loading...</span></div>}>
                <ResearchView />
              </React.Suspense>
            </MainLayout>
          </PrivateRoute>
        }
      />

      {/* Document / URL ingestion */}
      <Route
        path="/knowledge-base"
        element={
          <PrivateRoute>
            <MainLayout>
              <React.Suspense fallback={<div className="flex items-center justify-center h-full"><span className="text-zinc-500">Loading...</span></div>}>
                <KnowledgeBaseView />
              </React.Suspense>
            </MainLayout>
          </PrivateRoute>
        }
      />

      {/* RAG evaluation dashboard */}
      <Route
        path="/analytics"
        element={
          <PrivateRoute>
            <MainLayout>
              <React.Suspense fallback={<div className="flex items-center justify-center h-full"><span className="text-zinc-500">Loading...</span></div>}>
                <AnalyticsView />
              </React.Suspense>
            </MainLayout>
          </PrivateRoute>
        }
      />

      {/* Catch-all — send unknown paths back to Research */}
      <Route path="*" element={<Navigate to="/" />} />
    </Routes>
  );
}

/**
 * Top-level component — composes AuthProvider → Router → AppRoutes.
 */
export default function App() {
  return (
    <AuthProvider>
      <Router>
        <AppRoutes />
      </Router>
    </AuthProvider>
  );
}
