
import { useState } from "react";
import { useSearchHistory } from "@/hooks/useSearchHistory";
import { Clock, X, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";
import { useTheme } from "@/hooks/useTheme";

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";

export function SearchHistory() {
  const { history, deleteFromHistory, clearHistory, loading } = useSearchHistory();
  const { theme } = useTheme();
  const isDark = theme === "dark";
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false);
  const [isClearDialogOpen, setIsClearDialogOpen] = useState(false);

  const handleDeleteClick = (id: string) => {
    setSelectedId(id);
    setIsDeleteDialogOpen(true);
  };

  const confirmDelete = () => {
    if (selectedId) {
      deleteFromHistory(selectedId);
    }
    setIsDeleteDialogOpen(false);
  };

  const confirmClearAll = () => {
    clearHistory();
    setIsClearDialogOpen(false);
  };

  // Formatage de la date
  const formatDate = (timestamp: number) => {
    const date = new Date(timestamp);
    return date.toLocaleDateString('fr-FR', {
      day: '2-digit',
      month: '2-digit',
      year: '2-digit',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  return (
    <div className={cn(
      "border rounded-xl h-full flex flex-col",
      isDark ? "bg-card/30 border-gray-800" : "bg-white border-gray-200"
    )}>
      <div className={cn(
        "p-4 border-b flex items-center justify-between",
        isDark ? "border-gray-800" : "border-gray-200"
      )}>
        <h3 className="font-medium flex items-center gap-2">
          <Clock size={18} className={isDark ? "text-blue-400" : "text-blue-600"} />
          Historique de recherche
        </h3>
        
        {history.length > 0 && (
          <Button
            variant="ghost"
            size="sm"
            className="text-red-500 hover:text-red-700 hover:bg-red-100/10 -mr-2"
            onClick={() => setIsClearDialogOpen(true)}
          >
            <Trash2 size={16} />
          </Button>
        )}
      </div>
      
      <ScrollArea className="flex-1">
        {history.length === 0 ? (
          <div className="p-4 text-center text-muted-foreground">
            Aucune recherche récente
          </div>
        ) : (
          <div className="py-2">
            {history.map((item) => (
              <div key={item.id} className="group">
                <div className={cn(
                  "px-4 py-2 hover:bg-muted/40 flex justify-between items-start",
                  isDark ? "hover:bg-gray-800/50" : "hover:bg-gray-50"
                )}>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium truncate" title={item.queryText}>
                      {item.queryText}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {formatDate(item.timestamp)}
                    </p>
                    {item.ticketIds && item.ticketIds.length > 0 && (
                      <p className="text-xs mt-1 text-blue-500">
                        {item.ticketIds.length} ticket{item.ticketIds.length > 1 ? 's' : ''} trouvé{item.ticketIds.length > 1 ? 's' : ''}
                      </p>
                    )}
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="opacity-0 group-hover:opacity-100 h-6 w-6"
                    onClick={() => handleDeleteClick(item.id)}
                  >
                    <X size={14} />
                  </Button>
                </div>
                <Separator />
              </div>
            ))}
          </div>
        )}
      </ScrollArea>
      
      {/* Dialog de confirmation de suppression */}
      <AlertDialog open={isDeleteDialogOpen} onOpenChange={setIsDeleteDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Supprimer cet élément ?</AlertDialogTitle>
            <AlertDialogDescription>
              Cette action ne peut pas être annulée. Cet élément sera définitivement supprimé de votre historique.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Annuler</AlertDialogCancel>
            <AlertDialogAction onClick={confirmDelete} className="bg-red-600 hover:bg-red-700">
              Supprimer
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
      
      {/* Dialog de confirmation pour tout effacer */}
      <AlertDialog open={isClearDialogOpen} onOpenChange={setIsClearDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Effacer tout l'historique ?</AlertDialogTitle>
            <AlertDialogDescription>
              Cette action ne peut pas être annulée. Tout votre historique de recherche sera définitivement supprimé.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Annuler</AlertDialogCancel>
            <AlertDialogAction onClick={confirmClearAll} className="bg-red-600 hover:bg-red-700">
              Tout effacer
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
