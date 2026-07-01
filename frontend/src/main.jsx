/**
 * Application entry point — bootstraps the React SPA into the DOM.
 *
 * This file is the first JavaScript module executed by Vite. It mounts the root
 * `<App />` component inside React's `StrictMode`, which double-invokes certain
 * lifecycle hooks in development to surface side-effect bugs early.
 *
 * API endpoints: none directly — routing and API calls live in child components.
 */

import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

/**
 * Attach the React tree to the `#root` element in index.html.
 * `createRoot` is the React 18+ concurrent rendering API (replaces `ReactDOM.render`).
 */
createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
