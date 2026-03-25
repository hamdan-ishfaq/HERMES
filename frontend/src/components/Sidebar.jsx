import React from "react";
import { NavLink } from "react-router-dom";
import { Search, Database, BarChart3, LogOut, Hexagon } from "lucide-react";
import { useAuth } from "../context/AuthContext";
import clsx from "clsx";

const navItems = [
  { name: "Research", path: "/", icon: Search },
  { name: "Knowledge Base", path: "/knowledge-base", icon: Database },
  { name: "Analytics", path: "/analytics", icon: BarChart3 },
];

export default function Sidebar() {
  const { logout } = useAuth();

  return (
    <div className="w-64 h-screen border-r border-white/5 bg-zinc-950/50 flex flex-col p-6 backdrop-blur-md sticky top-0">
      <div className="flex items-center gap-3 mb-10 text-white">
        <div className="p-2 bg-white/10 rounded-xl border border-white/10">
          <Hexagon className="w-5 h-5 text-zinc-300" />
        </div>
        <span className="font-semibold text-lg tracking-tight">Hermes</span>
      </div>

      <nav className="flex-1 space-y-2">
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) =>
              clsx(
                "flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all duration-200 text-sm font-medium",
                isActive
                  ? "bg-white/10 text-white shadow-sm ring-1 ring-white/10"
                  : "text-zinc-400 hover:text-zinc-200 hover:bg-white/5"
              )
            }
          >
            <item.icon className="w-4 h-4" />
            {item.name}
          </NavLink>
        ))}
      </nav>

      <div className="mt-auto">
        <button
          onClick={logout}
          className="flex items-center gap-3 px-3 py-2.5 rounded-xl w-full text-zinc-400 hover:text-zinc-200 hover:bg-white/5 transition-colors text-sm font-medium"
        >
          <LogOut className="w-4 h-4" />
          Logout
        </button>
      </div>
    </div>
  );
}
