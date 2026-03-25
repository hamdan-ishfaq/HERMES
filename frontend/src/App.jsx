import React from "react";
import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider, useAuth } from "./context/AuthContext";
import Sidebar from "./components/Sidebar";
import LoginRegister from "./components/Auth/LoginRegister";

// Lazy loading views for better structure
const KnowledgeBaseView = React.lazy(() => import("./pages/KnowledgeBaseView"));
const ResearchView = React.lazy(() => import("./pages/ResearchView"));
const AnalyticsView = React.lazy(() => import("./pages/AnalyticsView"));

function PrivateRoute({ children }) {
  const { isAuthenticated } = useAuth();
  return isAuthenticated ? children : <Navigate to="/auth" />;
}

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

function AppRoutes() {
  const { isAuthenticated } = useAuth();

  return (
    <Routes>
      <Route
        path="/auth"
        element={!isAuthenticated ? <LoginRegister /> : <Navigate to="/" />}
      />

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

      <Route path="*" element={<Navigate to="/" />} />
    </Routes>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <Router>
        <AppRoutes />
      </Router>
    </AuthProvider>
  );
}
