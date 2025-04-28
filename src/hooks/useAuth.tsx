
import { createContext, useContext, useState, useEffect, ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import { useToast } from "@/hooks/use-toast";
import { authenticateUser, registerUser } from "../api/mongodb";

type User = {
  id: string;
  username: string;
  email: string;
  isAdmin: boolean;
} | null;

interface AuthContextType {
  user: User;
  loading: boolean;
  login: (email: string, password: string) => Promise<void>;
  signup: (username: string, email: string, password: string) => Promise<void>;
  logout: () => void;
  isAuthenticated: boolean;
  isAdmin: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const [user, setUser] = useState<User>(null);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();
  const { toast } = useToast();

  // Check if user is already logged in
  useEffect(() => {
    const checkAuth = async () => {
      try {
        const storedUser = localStorage.getItem("user");
        if (storedUser) {
          setUser(JSON.parse(storedUser));
        }
      } catch (error) {
        console.error("Authentication error:", error);
      } finally {
        setLoading(false);
      }
    };
    
    checkAuth();
  }, []);

  // Dans useAuth.tsx - fonction login modifiée
const login = async (email: string, password: string) => {
  setLoading(true);
  try {
    const response = await authenticateUser(email, password);
    
    // S'assurer que le token est bien stocké
    if (response && response.access_token) {
      localStorage.setItem("user", JSON.stringify(response.user));
      localStorage.setItem("token", response.access_token);
      setUser(response.user);
      
      toast({
        title: "Connexion réussie",
        description: "Bienvenue sur l'IA Ticket Wizard",
      });
      
      navigate(response.user.isAdmin ? "/admin" : "/dashboard");
    } else {
      throw new Error("Réponse d'authentification invalide");
    }
  } catch (error) {
    toast({
      title: "Erreur de connexion",
      description: "Email ou mot de passe incorrect",
      variant: "destructive",
    });
    console.error("Login error:", error);
  } finally {
    setLoading(false);
  }
};


  
  const signup = async (username: string, email: string, password: string) => {
    setLoading(true);
    try {
      const user = await registerUser(username, email, password);
      
      // Ensure the user object has id as a string
      const userWithStringId = {
        ...user,
        id: user.id.toString()
      };
      
      localStorage.setItem("user", JSON.stringify(userWithStringId));
      setUser(userWithStringId);
      
      toast({
        title: "Inscription réussie",
        description: "Votre compte a été créé avec succès",
      });
      
      navigate("/dashboard");
    } catch (error: any) {
      toast({
        title: "Erreur d'inscription",
        description: error.message || "Impossible de créer votre compte",
        variant: "destructive",
      });
      console.error("Signup error:", error);
    } finally {
      setLoading(false);
    }
  };
  
  const logout = () => {
    localStorage.removeItem("user");
    setUser(null);
    navigate("/login");
    toast({
      title: "Déconnexion réussie",
      description: "À bientôt!",
    });
  };
  
  return (
    <AuthContext.Provider
      value={{
        user,
        loading,
        login,
        signup,
        logout,
        isAuthenticated: !!user,
        isAdmin: user?.isAdmin || false,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
};
