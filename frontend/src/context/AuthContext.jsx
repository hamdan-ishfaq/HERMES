/**
 * Global authentication state — login, register, logout, and session persistence.
 *
 * Role in the UI:
 *   - Provides `isAuthenticated`, `login`, `register`, and `logout` to any descendant
 *     via React Context (consumed through `useAuth()`).
 *   - Persists the JWT in `localStorage` under `hermes_access_token` so page
 *     refreshes keep the user logged in.
 *
 * The context object and `useAuth` live in ./authContext so that this module
 * exports only a component and stays fast-refresh friendly.
 *
 * API endpoints:
 *   - POST /auth/login   — exchange email + password for access_token
 *   - POST /auth/register — create account and receive access_token
 */

import React, { useState } from "react";
import client from "../api/client";
import { AuthContext } from "./authContext";

const TOKEN_KEY = "hermes_access_token";

/**
 * Wraps the app and owns JWT lifecycle (read on mount, write on login/register, clear on logout).
 *
 * `isAuthenticated` is derived from `token` rather than stored alongside it:
 * the two could only ever disagree, and keeping them in step needed an effect
 * that re-rendered the subtree on every token change.
 *
 * @param {{ children: React.ReactNode }} props
 */
export const AuthProvider = ({ children }) => {
  const [token, setToken] = useState(() => localStorage.getItem(TOKEN_KEY));
  const isAuthenticated = !!token;

  /** Single write path, keeping localStorage and in-memory state in step. */
  const persist = (next) => {
    if (next) localStorage.setItem(TOKEN_KEY, next);
    else localStorage.removeItem(TOKEN_KEY);
    setToken(next);
  };

  /**
   * Authenticate an existing user and store the returned JWT.
   * @param {string} email
   * @param {string} password
   */
  const login = async (email, password) => {
    const res = await client.post("/auth/login", { email, password });
    persist(res.data.access_token);
  };

  /**
   * Create a new account and immediately log the user in with the returned JWT.
   * @param {string} email
   * @param {string} password
   */
  const register = async (email, password) => {
    const res = await client.post("/auth/register", { email, password });
    persist(res.data.access_token);
  };

  /** Clear stored credentials and mark the session as unauthenticated. */
  const logout = () => persist(null);

  return (
    <AuthContext.Provider value={{ isAuthenticated, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
};