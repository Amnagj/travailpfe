
import { FileSpreadsheet, X, Upload } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useTheme } from "@/hooks/useTheme";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";
interface FilePreviewProps {
  file: File;
  onRemove: () => void;
  onUpload: () => void;
  uploading: boolean;
}

export const FilePreview = ({ file, onRemove, onUpload, uploading }: FilePreviewProps) => {
  const { theme } = useTheme();
  const isDark = theme === "dark";

  return (
    <motion.div 
      className={cn(
        "border rounded-xl",
        isDark ? "bg-card/30 border-gray-800" : "bg-white border-gray-200"
      )}
      layout
    >
      <div className="p-3 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className={cn(
            "p-1.5 rounded-lg",
            isDark ? "bg-blue-900/50" : "bg-blue-100"
          )}>
            <FileSpreadsheet size={18} className={isDark ? "text-blue-400" : "text-blue-600"} />
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium truncate" title={file.name}>
              {file.name}
            </p>
            <p className="text-xs text-muted-foreground">
              {(file.size / 1024).toFixed(2)} KB
            </p>
          </div>
        </div>
        {!uploading && (
          <Button 
            variant="ghost" 
            size="sm" 
            onClick={onRemove}
            className="text-muted-foreground hover:text-foreground"
          >
            <X size={16} />
          </Button>
        )}
      </div>
      
      {!uploading && (
        <div className="p-4 pt-0 border-t border-gray-700">
          <Button 
            onClick={onUpload} 
            className={cn(
              "w-full mt-2",
              isDark ? "bg-blue-600 hover:bg-blue-700" : "bg-blue-500 hover:bg-blue-600"
            )}
          >
            <Upload size={18} className="mr-2" />
            Analyser le ticket
          </Button>
        </div>
      )}
    </motion.div>
  );
};
