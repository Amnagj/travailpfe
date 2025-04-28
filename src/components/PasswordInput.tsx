import { useState, forwardRef } from "react";
import { Eye, EyeOff } from "lucide-react";
import { Input } from "./ui/input";
import { FormControl, FormItem, FormLabel, FormMessage, useFormField } from "./ui/form";
import { Control } from "react-hook-form";
import { useTheme } from "@/hooks/useTheme";

interface PasswordInputProps {
  control?: Control<any>;
  name: string;
  label: string;
  className?: string;
  labelClass?: string;
  inputClass?: string;
}

export const PasswordInput = forwardRef<HTMLInputElement, PasswordInputProps>(
  ({ control, name, label, labelClass, inputClass, ...props }, ref) => {
    const [showPassword, setShowPassword] = useState(false);
    const { error } = useFormField();
    const { theme } = useTheme();
    const isDark = theme === "dark";

    return (
      <FormItem>
        <FormLabel className={labelClass}>{label}</FormLabel>
        <div className="relative">
          <FormControl>
            <Input
              ref={ref}
              type={showPassword ? "text" : "password"}
              className={`pr-10 ${inputClass || ""}`}
              {...props}
            />
          </FormControl>
          <button
            type="button"
            onClick={() => setShowPassword(!showPassword)}
            className={isDark
              ? "absolute right-3 top-1/2 -translate-y-1/2 text-white/70 hover:text-white"
              : "absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-800"}
            tabIndex={-1}
          >
            {showPassword ? (
              <EyeOff className="h-4 w-4" />
            ) : (
              <Eye className="h-4 w-4" />
            )}
            <span className="sr-only">
              {showPassword ? "Masquer le mot de passe" : "Afficher le mot de passe"}
            </span>
          </button>
        </div>
        <FormMessage />
      </FormItem>
    );
  }
);

// Ajoutez un displayName pour faciliter le débogage
PasswordInput.displayName = "PasswordInput";

