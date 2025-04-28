// Améliorations du hook useSearchHistory.tsx
import React, { createContext, useContext, useState, useEffect, ReactNode } from "react";
import { useAuth } from "@/hooks/useAuth";
import { useToast } from "@/hooks/use-toast";
import { getSearchHistory, hideHistoryItem, clearSearchHistory, addSearchToHistory } from "../api/fastApiService";

interface SearchHistoryItem {
  id: string;
  timestamp: number;
  queryText: string;
  result: string;
  ticketIds?: string[];
  visible: boolean;
}

interface SearchHistoryContextType {
  history: SearchHistoryItem[];
  loading: boolean;
  refreshHistory: () => Promise<void>;
  deleteFromHistory: (id: string) => Promise<void>;
  clearHistory: () => Promise<void>;
  addToHistory: (item: { queryText: string; result: string; ticketIds?: string[] }) => Promise<void>;
}

const SearchHistoryContext = createContext<SearchHistoryContextType | undefined>(undefined);

export function SearchHistoryProvider({ children }: { children: ReactNode }) {
  const [history, setHistory] = useState<SearchHistoryItem[]>([]);
  const [loading, setLoading] = useState(false);
  const { user, isAuthenticated } = useAuth();
  const { toast } = useToast();

  // Charger l'historique depuis l'API quand l'utilisateur est authentifié
  useEffect(() => {
    if (isAuthenticated && user?.id) {
      refreshHistory();
    } else {
      // Réinitialiser l'historique si l'utilisateur n'est pas connecté
      setHistory([]);
    }
  }, [isAuthenticated, user?.id]);

  // Récupérer l'historique depuis l'API
  const refreshHistory = async () => {
    if (!isAuthenticated || !user?.id) {
      console.log("Impossible de rafraîchir l'historique: utilisateur non connecté");
      return;
    }
    
    setLoading(true);
    try {
      console.log("Récupération de l'historique...");
      const response = await getSearchHistory();
      console.log("Réponse de l'API historique:", response);
      
      if (response.status === 'success' && Array.isArray(response.history)) {
        setHistory(response.history);
      } else {
        toast({
          title: "Avertissement",
          description: response.message || "Aucun historique disponible",
          variant: "default"
        });
        // S'assurer que l'historique est vide si aucune donnée n'est disponible
        setHistory([]);
      }
    } catch (error: any) {
      console.error("Erreur lors du chargement de l'historique:", error);
      toast({
        title: "Erreur",
        description: error.message || "Impossible de charger l'historique",
        variant: "destructive"
      });
    } finally {
      setLoading(false);
    }
  };

  // Masquer un élément de l'historique
  const deleteFromHistory = async (id: string) => {
    if (!isAuthenticated || !user?.id) return;
    
    setLoading(true);
    try {
      const response = await hideHistoryItem(id);
      
      if (response.status === 'success') {
        // Mettre à jour l'état local
        setHistory(prev => prev.filter(item => item.id !== id));
        toast({
          title: "Succès",
          description: "L'élément a été retiré de l'historique.",
        });
      } else {
        toast({
          title: "Erreur",
          description: response.message || "Erreur lors de la suppression de l'élément",
          variant: "destructive"
        });
      }
    } catch (error: any) {
      console.error("Erreur lors de la suppression:", error);
      toast({
        title: "Erreur",
        description: error.message || "Impossible de supprimer l'élément",
        variant: "destructive"
      });
    } finally {
      setLoading(false);
    }
  };

  // Effacer tout l'historique
  const clearHistory = async () => {
    if (!isAuthenticated || !user?.id) return;
    
    setLoading(true);
    try {
      const response = await clearSearchHistory();
      
      if (response.status === 'success') {
        setHistory([]);
        toast({
          title: "Historique effacé",
          description: "Votre historique de recherche a été effacé.",
        });
      } else {
        toast({
          title: "Erreur",
          description: response.message || "Erreur lors de l'effacement de l'historique",
          variant: "destructive"
        });
      }
    } catch (error: any) {
      console.error("Erreur lors de l'effacement:", error);
      toast({
        title: "Erreur",
        description: error.message || "Impossible d'effacer l'historique",
        variant: "destructive"
      });
    } finally {
      setLoading(false);
    }
  };

  // Ajouter un élément à l'historique
  const addToHistory = async (item: { queryText: string; result: string; ticketIds?: string[] }) => {
    if (!isAuthenticated || !user?.id) {
      console.log("Impossible d'ajouter à l'historique: utilisateur non connecté");
      return;
    }
    
    try {
      console.log("Ajout à l'historique:", item.queryText);
      // Appeler l'API pour ajouter à l'historique dans la base de données
      const response = await addSearchToHistory(item.queryText, item.result);
      
      if (response.status === 'success' && response.item) {
        // Créer le nouvel élément avec l'ID retourné par l'API
        const newItem: SearchHistoryItem = {
          id: response.item.id,
          timestamp: response.item.timestamp || Date.now(),
          queryText: item.queryText,
          result: item.result,
          ticketIds: item.ticketIds,
          visible: true
        };
        
        // Mettre à jour l'état local pour afficher immédiatement
        setHistory(prev => [newItem, ...prev]);
        console.log("Élément ajouté à l'historique avec succès");
      } else {
        console.error("Erreur lors de l'ajout à l'historique:", response.message);
      }
    } catch (error: any) {
      console.error("Erreur lors de l'ajout à l'historique:", error);
    }
  };

  return (
    <SearchHistoryContext.Provider value={{
      history,
      loading,
      refreshHistory,
      deleteFromHistory,
      clearHistory,
      addToHistory
    }}>
      {children}
    </SearchHistoryContext.Provider>
  );
}

export function useSearchHistory() {
  const context = useContext(SearchHistoryContext);
  if (context === undefined) {
    throw new Error("useSearchHistory doit être utilisé à l'intérieur d'un SearchHistoryProvider");
  }
  return context;
}

