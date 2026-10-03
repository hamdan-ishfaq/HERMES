/**
 * The auth context object and its consumer hook.
 *
 * Kept apart from AuthContext.jsx so that file exports only a component:
 * react-refresh cannot hot-reload a module that mixes component and
 * non-component exports.
 */

import { createContext, useContext } from "react";

export const AuthContext = createContext();

/**
 * Hook to read auth state and actions from the nearest AuthProvider.
 * Must be called inside a component tree wrapped by `<AuthProvider>`.
 *
 * @returns {{ isAuthenticated: boolean, login: Function, register: Function, logout: Function }}
 */
export const useAuth = () => useContext(AuthContext);