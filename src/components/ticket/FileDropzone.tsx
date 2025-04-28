
import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Upload } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { useTheme } from "@/hooks/useTheme";
import { cn } from "@/lib/utils";

interface FileDropzoneProps {
  onFileAccepted: (file: File) => void;
}

export const FileDropzone = ({ onFileAccepted }: FileDropzoneProps) => {
  const { toast } = useToast();
  const { theme } = useTheme();
  const isDark = theme === "dark";

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const selectedFile = acceptedFiles[0];
    
    if (selectedFile) {
      const fileName = selectedFile.name.toLowerCase();
      
      if (
        fileName.endsWith('.xlsx') 

      ) {
        onFileAccepted(selectedFile);
        toast({
          title: "Fichier accepté",
          description: `Le fichier ${selectedFile.name} a été chargé avec succès`,
        });
      } else {
        toast({
          title: "Format non supporté",
          description: "Veuillez télécharger un fichier Excel (.xlsx, .xls) ou CSV",
          variant: "destructive",
        });
      }
    }
  }, [toast, onFileAccepted]);
  
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/vnd.ms-excel': ['.xls'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'text/csv': ['.csv']
    },
    maxFiles: 1,
    multiple: false,
  });

  return (
    <div
      {...getRootProps()}
      className={cn(
        "border-2 border-dashed rounded-xl p-8 text-center transition-colors",
        isDragActive 
          ? isDark
            ? "border-blue-500 bg-blue-900/20" 
            : "border-blue-500 bg-blue-50"
          : isDark
            ? "border-gray-700 hover:border-blue-500 hover:bg-blue-900/10"
            : "border-gray-300 hover:border-blue-400 hover:bg-blue-50/30"
      )}
    >
      <input {...getInputProps()} />
      <div className="flex flex-col items-center justify-center space-y-2">
        <div className={cn(
          "p-2 rounded-full",
          isDragActive 
            ? isDark
              ? "bg-blue-900/50"
              : "bg-blue-100" 
            : isDark
              ? "bg-gray-800"
              : "bg-gray-100"
        )}>
          <Upload 
            size={24} 
            className={isDragActive ? 'text-blue-600' : 'text-gray-400'} 
          />
        </div>
        <h3 className="text-base font-medium text-foreground">
          {isDragActive ? "Déposez votre fichier ici" : "Importer un ticket"}
        </h3>
        <p className={cn(
          "text-xs",
          isDark ? "text-gray-400" : "text-gray-500"
        )}>
          Formats supportés: XLSX, XLS, CSV
        </p>
        <Button 
          type="button" 
          variant="outline"
          size="sm"
          className={cn(
            isDark 
              ? "border-blue-700 text-blue-400 hover:bg-blue-900/30"
              : "border-blue-200 text-blue-600"
          )}
        >
          Parcourir
        </Button>
      </div>
    </div>
  );
};
