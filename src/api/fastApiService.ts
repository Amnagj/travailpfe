import axios from 'axios';

// Base URL for our FastAPI backend
const API_BASE_URL = 'http://localhost:8000';

// Interface for the ticket search response
interface TicketSearchResponse {
  status: 'success' | 'not_found' | 'error';
  message: string;
  tickets?: {
    ticket_id: string;
    problem: string;
    solution: string;
    keywords: string;
    similarity_score: number;
  }[];
  temps_recherche?: number;
  query?: string;
}

// Interface for the file upload response
interface FileUploadResponse {
  status: 'success' | 'error';
  message: string;
  processed_tickets?: number;
  timestamp?: string;
}

// Interface pour la création d'utilisateur
interface UserCreationResponse {
  status: 'success' | 'error';
  message: string;
  user?: {
    id: string;
    username: string;
    email: string;
    isAdmin: boolean;
  };
  password?: string; // Mot de passe généré
}

interface SearchHistoryItem {
  id: string;
  userId: string;
  queryText: string;
  result: string;
  ticketIds?: string[];
  timestamp: number;
  visible: boolean;
}

// Interface pour la réponse d'historique
interface SearchHistoryResponse {
  status: 'success' | 'error';
  history?: SearchHistoryItem[];
  message?: string;
}

/**
 * Valide si le fichier Excel est correctement formaté pour la recherche
 */
export async function validateExcelFormat(file: File): Promise<{ isValid: boolean; message: string }> {
  try {
    const formData = new FormData();
    formData.append('file', file);

    const response = await axios.post(`${API_BASE_URL}/validate-excel`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });

    return response.data;
  } catch (error) {
    console.error('Erreur lors de la validation du fichier:', error);
    return {
      isValid: false,
      message: "Erreur lors de la validation du format du fichier Excel."
    };
  }
}

/**
 * Upload an Excel file to be processed by the backend
 */
export async function uploadExcelFile(file: File): Promise<TicketSearchResponse> {
  try {
    const formData = new FormData();
    formData.append('file', file);

    const token = localStorage.getItem('token');

    const headers: Record<string, string> = {
      'Content-Type': 'multipart/form-data'
    };

    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }

    const response = await axios.post(`${API_BASE_URL}/upload-file`, formData, {
      headers,
      onUploadProgress: (progressEvent) => {
        const percentCompleted = Math.round((progressEvent.loaded * 100) / (progressEvent.total || 100));
        console.log(`Upload progress: ${percentCompleted}%`);
      }
    });

    if (token && response.data.status === 'success') {
      try {
        await axios.post(`${API_BASE_URL}/messages/record`, {
          message_text: file.name,
          ticket_ids: response.data.tickets?.map(t => t.ticket_id) || []
        }, {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        });
      } catch (error) {
        console.error("Erreur lors de l'enregistrement du message:", error);
      }
    }

    return response.data;
  } catch (error) {
    console.error('Error uploading file:', error);
    return {
      status: 'error',
      message: 'Une erreur est survenue lors du téléchargement du fichier.'
    };
  }
}

/**
 * Get statistics about the tickets in MongoDB
 */
export async function getTicketStats() {
  try {
    const response = await axios.get(`${API_BASE_URL}/ticket-stats`);
    return response.data;
  } catch (error) {
    console.error('Error fetching ticket statistics:', error);
    return {
      status: 'error',
      message: 'Une erreur est survenue lors de la récupération des statistiques.'
    };
  }
}

/**
 * Créer un nouvel utilisateur (admin uniquement)
 */
export async function createUser(username: string, email: string): Promise<UserCreationResponse> {
  try {
    const response = await axios.post(`${API_BASE_URL}/create-user`, {
      username,
      email
    });

    return response.data;
  } catch (error) {
    console.error('Erreur lors de la création de l\'utilisateur:', error);
    return {
      status: 'error',
      message: 'Une erreur est survenue lors de la création de l\'utilisateur.'
    };
  }
}

export async function addSearchToHistory(searchText: string, result: any): Promise<any> {
  try {
    const response = await axios.post(`${API_BASE_URL}/search-history/add`, {
      ticket_text: searchText,
      result: typeof result === 'string' ? result : JSON.stringify(result)
    }, {
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('token')}`
      }
    });
    return response.data;
  } catch (error) {
    console.error('Error adding search to history:', error);
    return {
      status: 'error',
      message: 'Une erreur est survenue lors de l\'ajout à l\'historique.'
    };
  }
}

/**
 * Récupère l'historique des recherches
 */
export async function getSearchHistory(): Promise<SearchHistoryResponse> {
  try {
    const response = await axios.get(`${API_BASE_URL}/search-history`, {
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('token')}`
      }
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching search history:', error);
    return {
      status: 'error',
      message: 'Une erreur est survenue lors de la récupération de l\'historique.'
    };
  }
}

/**
 * Cache un élément de l'historique
 */
export async function hideHistoryItem(itemId: string): Promise<any> {
  try {
    const response = await axios.post(`${API_BASE_URL}/search-history/hide`, {
      item_id: itemId
    }, {
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('token')}`
      }
    });
    return response.data;
  } catch (error) {
    console.error('Error hiding history item:', error);
    return {
      status: 'error',
      message: 'Une erreur est survenue lors du masquage de l\'élément.'
    };
  }
}

/**
 * Efface tout l'historique
 */
export async function clearSearchHistory(): Promise<any> {
  try {
    const response = await axios.post(`${API_BASE_URL}/search-history/clear`, {}, {
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('token')}`
      }
    });
    return response.data;
  } catch (error) {
    console.error('Error clearing history:', error);
    return {
      status: 'error',
      message: 'Une erreur est survenue lors de l\'effacement de l\'historique.'
    };
  }
}

/**
 * Recherche de tickets similaires
 */
export async function searchSimilarTickets(ticketText: string): Promise<TicketSearchResponse> {
  try {
    const response = await axios.post(`${API_BASE_URL}/search-tickets`, {
      ticket_text: ticketText
    });

    const userData = localStorage.getItem('user');
    const user = userData ? JSON.parse(userData) : null;

    const token = localStorage.getItem('token');
    if (token && user) {
      try {
        await addSearchToHistory(ticketText, JSON.stringify(response.data));
      } catch (error) {
        console.error("Erreur lors de l'ajout à l'historique:", error);
      }
    }

    return response.data;
  } catch (error) {
    console.error('Error searching for similar tickets:', error);
    return {
      status: 'error',
      message: 'Une erreur est survenue lors de la recherche de tickets similaires.'
    };
  }
}

// Interface pour la réponse d'importation de tickets
interface TicketImportResponse {
  status: 'success' | 'error';
  message: string;
  processed_tickets?: number;
  fixed_tickets?: number;
  skipped_tickets?: number;
  timestamp?: string;
}

/**
 * Télécharge un fichier Excel vers l'API pour traitement et importation dans MongoDB
 */
export async function telecharger(file: File): Promise<TicketImportResponse> {
  try {
    const formData = new FormData();
    formData.append('file', file);

    const response = await axios.post(`${API_BASE_URL}/telecharger-excel`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      },
      onUploadProgress: (progressEvent) => {
        const percentCompleted = Math.round((progressEvent.loaded * 100) / (progressEvent.total || 100));
        console.log(`Upload progress: ${percentCompleted}%`);
      }
    });

    return response.data;
  } catch (error) {
    console.error('Error uploading file:', error);
    return {
      status: 'error',
      message: 'Une erreur est survenue lors du téléchargement du fichier.'
    };
  }
}
