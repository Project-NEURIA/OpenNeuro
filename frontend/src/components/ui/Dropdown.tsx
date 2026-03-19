import { useState, useRef, useEffect } from "react";
import { cn } from "@/lib/utils";
import { ChevronDown } from "lucide-react";

interface DropdownOption {
  value: string;
  label: string;
}

interface DropdownProps {
  value: string;
  options: DropdownOption[];
  onChange: (value: string) => void;
  placeholder?: string;
  className?: string;
}

export function Dropdown({ value, options, onChange, placeholder = "Select...", className }: DropdownProps) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  const selected = options.find((o) => o.value === value);

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    if (open) {
      document.addEventListener("mousedown", handleClickOutside);
      return () => document.removeEventListener("mousedown", handleClickOutside);
    }
  }, [open]);

  return (
    <div ref={ref} className={cn("relative", className)}>
      <button
        type="button"
        onClick={() => setOpen(!open)}
        onPointerDown={(e) => e.stopPropagation()}
        className={cn(
          "w-full flex items-center justify-between gap-2",
          "rounded-lg px-3 py-1.5 text-[13px] font-mono text-left",
          "bg-accent border border-white/[0.1] text-foreground",
          "focus:outline-none focus:border-conduit/60",
          "transition-colors hover:bg-glass-hover",
        )}
      >
        <span className={cn("truncate", !selected && "text-muted-foreground")}>
          {selected?.label ?? placeholder}
        </span>
        <ChevronDown size={14} className={cn("shrink-0 text-muted-foreground transition-transform", open && "rotate-180")} />
      </button>

      {open && (
        <div
          className={cn(
            "absolute z-50 mt-1 w-full",
            "bg-secondary border border-white/[0.1] rounded-lg",
            "shadow-xl shadow-black/40",
            "max-h-48 overflow-y-auto overscroll-none",
            "py-1",
          )}
        >
          {options.map((opt) => (
            <button
              key={opt.value}
              type="button"
              onClick={() => {
                onChange(opt.value);
                setOpen(false);
              }}
              onPointerDown={(e) => e.stopPropagation()}
              className={cn(
                "w-full px-3 py-1.5 text-left text-[13px] font-mono",
                "transition-colors",
                opt.value === value
                  ? "bg-conduit/20 text-conduit"
                  : "text-foreground/80 hover:bg-glass-hover",
              )}
            >
              {opt.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
