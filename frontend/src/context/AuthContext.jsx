/**
 * Global authentication state — login, register, logout, and session persistence.
 *
 * Role in the UI:
 *   - Provides `isAuthenticated`, `login`, `register`, and `logout` to any descendant
 *     via React Context (consumed through `useAuth()`).
 *   - Persists the JWT in `localStorage` under `hermes_access_token` so page
 *     refreshes keep the user logged in.
 *   - Gates rendering until the initial token check completes (`loading` flag).
 *
 * API endpoints:
 *   - POST /auth/login   — exchange email + password for access_token
 *   - POST /auth/register — create account and receive access_token
 */

import React, { createContext, useContext, useState, useEffect } from "react";
import client from "../api/client";

const AuthContext = createContext();

/**
 * Hook to read auth state and actions from the nearest AuthProvider.
 * Must be called inside a component tree wrapped by `<AuthProvider>`.
 *
 * @returns {{ isAuthenticated: boolean, login: Function, register: Function, logout: Function }}
 */
export const useAuth = () => useContext(AuthContext);

/**
 * Wraps the app and owns JWT lifecycle (read on mount, write on login/register, clear on logout).
 *
 * @param {{ children: React.ReactNode }} props
 */
export const AuthProvider = ({ children }) => {
  const [token, setToken] = useState(localStorage.getItem("hermes_access_token"));
  const [isAuthenticated, setIsAuthenticated] = useState(!!token);
  const [loading, setLoading] = useState(true);

  /** Re-sync auth flag whenever the in-memory token changes. */
  useEffect(() => {
    // Optionally we could fetch user profile here if the backend had an endpoint
    setIsAuthenticated(!!token);
    setLoading(false);
  }, [token]);

  /**
   * Authenticate an existing user and store the returned JWT.
   * @param {string} email
   * @param {string} password
   */
  const login = async (email, password) => {
    const res = await client.post("/auth/login", { email, password });
    const accessToken = res.data.access_token;
    localStorage.setItem("hermes_access_token", accessToken);
    setToken(accessToken);
    setIsAuthenticated(true);
  };

  /**
   * Create a new account and immediately log the user in with the returned JWT.
   * @param {string} email
   * @param {string} password
   */
  const register = async (email, password) => {
    const res = await client.post("/auth/register", { email, password });
    const accessToken = res.data.access_token;
    localStorage.setItem("hermes_access_token", accessToken);
    setToken(accessToken);
    setIsAuthenticated(true);
  };

  /** Clear stored credentials and mark the session as unauthenticated. */
  const logout = () => {
    localStorage.removeItem("hermes_access_token");
    setToken(null);
    setIsAuthenticated(false);
  };

  if (loading) return null;

  return (
    <AuthContext.Provider value={{ isAuthenticated, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
};
