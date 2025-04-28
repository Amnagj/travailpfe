import { useState } from "react";
import { useTheme } from "@/hooks/useTheme";
import { cn } from "@/lib/utils";
import { FileDropzone } from "./ticket/FileDropzone";
import { FilePreview } from "./ticket/FilePreview";
import { UploadProgress } from "./ticket/UploadProgress";
import { TicketInstructions } from "./TicketInstructions";
import { useSearchHistory } from "@/hooks/useSearchHistory";
import { useToast } from "@/hooks/use-toast";
import { validateExcelFormat, uploadExcelFile } from '../api/fastApiService';

export const TicketUpload = ({ onFileUploaded }: { onFileUploaded: (text: string, ticketIds?: string[]) => void }) => {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [isMinimized, setIsMinimized] = useState(false);
  const { toast } = useToast();
  const { theme } = useTheme();
  const { addToHistory } = useSearchHistory();
  const isDark = theme === "dark";

  // Validation du fichier Excel
  const validateAndUpload = async () => {
    if (!file) return;
    try {
      // Valider le format du fichier Excel
      const validation = await validateExcelFormat(file);
      if (validation.isValid) {
        handleUpload();
      } else {
        toast({
          title: "Format incorrect",
          description: validation.message,
          variant: "destructive",
        });
      }
    } catch (error) {
      console.error("Error validating file:", error);
      toast({
        title: "Erreur de validation",
        description: "Impossible de valider le format du fichier.",
        variant: "destructive",
      });
    }
  };

  // Upload et traitement du fichier
  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setProgress(0);
    
    let progressInterval: NodeJS.Timeout;
    
    try {
      progressInterval = setInterval(() => {
        setProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 500);
      
      // Envoyer le fichier au backend via l'API
      const response = await uploadExcelFile(file);
      
      if (response.status === "success") {
        clearInterval(progressInterval);
        setProgress(100);
        
        // Traiter la réponse pour obtenir les tickets similaires
        if (response.tickets && response.tickets.length > 0) {
          const ticketIds = response.tickets.map(t => t.ticket_id);
          const bestMatch = response.tickets[0];
          const responseMessage = `
J'ai trouvé une solution pour votre ticket!
**Problème identifié:** ${bestMatch.problem}
**Solution:** ${bestMatch.solution}
*Temps de recherche: ${response.temps_recherche?.toFixed(2)}s*
`;
          toast({
            title: "Solution trouvée",
            description: `Une solution a été trouvée avec un score de similarité de ${(bestMatch.similarity_score * 100).toFixed(1)}%`,
          });
          
          // Ajouter à l'historique si l'utilisateur est connecté
          if (localStorage.getItem('token')) {
            addToHistory({
              queryText: file.name,
              result: responseMessage,
              ticketIds: ticketIds
            });
          }
          onFileUploaded(responseMessage, ticketIds);
        } else {
          toast({
            title: "Aucun résultat",
            description: "Aucun ticket similaire n'a été trouvé.",
          });
          onFileUploaded("Aucun ticket similaire n'a été trouvé pour votre demande.");
        }
        setIsMinimized(true);
        setTimeout(() => setProgress(0), 1000);
        setUploading(false);
      } else {
        throw new Error(response.message || "Erreur lors du traitement du fichier");
      }
    } catch (error) {
      console.error("Error processing file:", error);
      toast({
        title: "Erreur de traitement",
        description: "Une erreur est survenue lors du traitement du fichier.",
        variant: "destructive",
      });
      onFileUploaded("Une erreur est survenue lors de l'analyse de votre ticket. Veuillez réessayer.");
      clearInterval(progressInterval);
      setUploading(false);
      setProgress(0);
    }
  };

  const readFileAsText = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (event) => {
        if (event.target?.result) {
          resolve(event.target.result.toString());
        } else {
          reject(new Error("Failed to read file"));
        }
      };
      reader.onerror = () => reject(reader.error);
      reader.readAsText(file);
    });
  };

  return (
    <div className="flex flex-col md:flex-row gap-6">
      <div className={cn(
        "transition-all duration-300",
        isMinimized ? "w-full md:w-1/3" : "w-full md:w-2/3"
      )}>
        {!file ? (
          <FileDropzone onFileAccepted={setFile} />
        ) : (
          <FilePreview
            file={file}
            onRemove={() => setFile(null)}
            onUpload={validateAndUpload} // Utilisez validateAndUpload ici au lieu de handleUpload directement
            uploading={uploading}
          />
        )}
        {uploading && <UploadProgress progress={progress} />}
      </div>
      <div className={cn(
        "transition-all duration-300",
        isMinimized ? "hidden" : "w-full md:w-1/3"
      )}>
        <TicketInstructions />
      </div>
    </div>
  );
};