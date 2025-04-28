import axios from 'axios';

const API_URL = 'http://localhost:8000';

export async function authenticateUser(email: string, password: string) {
  try {
      const formData = new FormData();
      formData.append('username', email); // FastAPI attend 'username' pour OAuth2
      formData.append('password', password);
      
      // Utiliser API_URL au lieu de API_BASE_URL
      const response = await axios.post(`${API_URL}/token`, formData);
      
      // Retourner l'utilisateur ET le token
      return {
          user: response.data.user,
          access_token: response.data.access_token
      };
  } catch (error) {
      console.error('Authentication error:', error);
      throw new Error('Échec de l\'authentification');
  }
}

export const registerUser = async (username: string, email: string, password: string) => {
  try {
    const response = await axios.post(`${API_URL}/register`, {
      username,
      email,
      password
    });
    
    if (response.data && response.data.access_token) {
      localStorage.setItem('token', response.data.access_token);
      return response.data.user;
    } else {
      throw new Error('Invalid response format');
    }
  } catch (error) {
    console.error('Registration error:', error);
    throw new Error(error.response?.data?.detail || 'Registration failed');
  }
};



// Nouvelle fonction pour connecter à MongoDB
export const connectToMongoDB = async () => {
  // Cette fonction est utilisée côté serveur, donc on n'a pas besoin 
  // de l'implémenter ici dans le frontend
  console.warn('connectToMongoDB est appelé dans le frontend, ce qui ne devrait pas arriver');
  throw new Error('Cette fonction ne doit pas être appelée côté client');
};

// Nouvelle fonction pour enregistrer un message
export const recordMessage = async (userId: string, content: string, role: "user" | "assistant") => {
  try {
    const token = localStorage.getItem('token');
    if (!token) {
      throw new Error('User not authenticated');
    }

    const response = await axios.post(
      `${API_URL}/messages/record`, 
      {
        userId,
        content,
        role
      },
      {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      }
    );
    
    return response.data;
  } catch (error) {
    console.error('Error recording message:', error);
    throw error;
  }
};

// Ajoutez d'autres fonctions si nécessaire pour votre application