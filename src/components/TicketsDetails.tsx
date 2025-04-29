import { useState, useEffect } from "react";
import { Loader2, Clock, AlertCircle, CheckCircle2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { useTheme } from "@/hooks/useTheme";

interface TicketDetailsProps {
  ticketData: Record<string, any> | null;
  loading: boolean;
}

export const TicketDetails = ({ ticketData, loading }: TicketDetailsProps) => {
  const { theme } = useTheme();
  const isDark = theme === "dark";
  const [visibleFields, setVisibleFields] = useState<string[]>([]);
  const [processingTime, setProcessingTime] = useState<number>(0);

  // Simuler le temps de traitement qui s'écoule
  useEffect(() => {
    let timer: NodeJS.Timeout;
    if (loading && ticketData) {
      timer = setInterval(() => {
        setProcessingTime(prev => prev + 0.1);
      }, 100);
    } else {
      setProcessingTime(0);
    }
    
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [loading, ticketData]);

  // Choisir les champs à afficher lorsque les données changent
  useEffect(() => {
    if (ticketData) {
      // Sélectionner les champs les plus pertinents
      const allFields = Object.keys(ticketData);
      const priorityFields = [
        "Description", "Summary", "Title", "Issue", "Problem", "Category", 
        "Priority", "Status", "Component", "Assigned", "ID", "Ticket_ID", 
        "Severity", "Type", "Module", "Customer"
      ];
      
      // Filtrer les champs par priorité et disponibilité
      const fieldsToShow = priorityFields
        .filter(field => allFields.some(f => 
          f.toLowerCase().includes(field.toLowerCase())))
        .slice(0, 5); // Limiter à 5 champs maximum
      
      // Ajouter d'autres champs si nécessaire pour arriver à 5 champs
      if (fieldsToShow.length < 5) {
        const remainingFields = allFields
          .filter(field => !fieldsToShow.some(f => 
            field.toLowerCase().includes(f.toLowerCase())))
          .filter(field => {
            const value = ticketData[field];
            return value && typeof value === 'string' && value.length < 100;
          })
          .slice(0, 5 - fieldsToShow.length);
        
        setVisibleFields([...fieldsToShow, ...remainingFields]);
      } else {
        setVisibleFields(fieldsToShow);
      }
    } else {
      setVisibleFields([]);
    }
  }, [ticketData]);

  if (!loading && !ticketData) return null;

  return (
    <div className={cn(
      "p-4 rounded-xl border mb-4 transition-all",
      isDark
        ? "bg-slate-800/95 border-slate-700 text-blue-100"
        : "bg-white/95 border-gray-200 text-slate-900"
    )}>
      <div className="flex items-center justify-between mb-3">
        <h3 className={cn(
          "text-sm font-medium flex items-center gap-2",
          isDark ? "text-blue-300" : "text-blue-700"
        )}>
          {loading ? (
            <AlertCircle size={16} className="text-amber-500" />
          ) : (
            <CheckCircle2 size={16} className="text-green-500" />
          )}
          Détails du ticket en cours d'analyse
        </h3>
        
        {loading && (
          <div className="flex items-center gap-2">
            <Loader2 size={14} className="animate-spin text-blue-500" />
            <span className="text-xs opacity-75">Recherche en cours...</span>
          </div>
        )}
      </div>

      {loading && !ticketData && (
        <div className="flex justify-center py-6">
          <Loader2 size={24} className="animate-spin text-blue-500" />
        </div>
      )}

      {ticketData && (
        <>
          <div className="grid gap-2 mb-3">
            {visibleFields.map((field) => (
              <div key={field} className="grid grid-cols-3 text-sm">
                <div className={cn(
                  "font-medium",
                  isDark ? "text-blue-200" : "text-slate-700"
                )}>
                  {field}:
                </div>
                <div className="col-span-2 break-words">
                  {String(ticketData[field])}
                </div>
              </div>
            ))}
          </div>

          {loading && (
            <div className={cn(
              "text-xs flex items-center gap-1 justify-end",
              isDark ? "text-blue-200/70" : "text-slate-500"
            )}>
              <Clock size={12} />
              <span>Temps d'analyse: {processingTime.toFixed(1)}s</span>
            </div>
          )}
        </>
      )}
    </div>
  );
};