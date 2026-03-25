import React, { createContext, useContext, useState, useEffect } from "react";
import client from "../api/client";

const AuthContext = createContext();

export const useAuth = () => useContext(AuthContext);

export const AuthProvider = ({ children }) => {
  const [token, setToken] = useState(localStorage.getItem("hermes_access_token"));
  const [isAuthenticated, setIsAuthenticated] = useState(!!token);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Optionally we could fetch user profile here if the backend had an endpoint
    setIsAuthenticated(!!token);
    setLoading(false);
  }, [token]);

  const login = async (email, password) => {
    const res = await client.post("/auth/login", { email, password });
    const accessToken = res.data.access_token;
    localStorage.setItem("hermes_access_token", accessToken);
    setToken(accessToken);
    setIsAuthenticated(true);
  };

  const register = async (email, password) => {
    const res = await client.post("/auth/register", { email, password });
    const accessToken = res.data.access_token;
    localStorage.setItem("hermes_access_token", accessToken);
    setToken(accessToken);
    setIsAuthenticated(true);
  };

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
