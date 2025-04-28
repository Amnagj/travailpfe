import { connectToMongoDB } from "@/api/mongodb";
import * as crypto from 'crypto';
import { MongoClient, Db, ObjectId } from 'mongodb';

// Modifier l'interface pour utiliser ObjectId pour _id
export interface User {
  _id: string | ObjectId;  // Permet les deux types pour la compatibilité
  username: string;
  email: string;
  password: string;
  isAdmin: boolean;
  createdAt: Date;
  updatedAt: Date;
}

export interface NewUserInput {
  username: string;
  email: string;
  password?: string; // Optionnel, sera généré si non fourni
  isAdmin?: boolean; // Optionnel, false par défaut
}

export interface UserOutput {
  id: string;
  username: string;
  email: string;
  isAdmin: boolean;
  createdAt: string;
}

export const usersRepository = {
  // Générer un mot de passe aléatoire
  generatePassword(length = 12): string {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()';
    let password = '';
    for (let i = 0; i < length; i++) {
      password += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return password;
  },

  // Hacher un mot de passe
  hashPassword(password: string): string {
    return crypto.createHash('sha256').update(password).digest('hex');
  },

  async findById(id: string): Promise<User | null> {
    let client: MongoClient | null = null;
    try {
      client = await connectToMongoDB() as MongoClient;
      const db: Db = client.db("Access");
      const collection = db.collection('Users');
      
      // Vérifier si l'id peut être converti en ObjectId (24 caractères hexadécimaux)
      let query = {};
      if (ObjectId.isValid(id) && id.match(/^[0-9a-fA-F]{24}$/)) {
        query = { _id: new ObjectId(id) };
      } else {
        query = { _id: id };
      }
      
      const user = await collection.findOne(query);
      return user ? user as unknown as User : null;
    } finally {
      if (client) await client.close();
    }
  },

  async findAll(): Promise<UserOutput[]> {
    let client: MongoClient | null = null;
    try {
      client = await connectToMongoDB() as MongoClient;
      const db: Db = client.db("Access");
      const collection = db.collection('Users');
      
      const users = await collection.find().toArray();
      return users.map(user => ({
        id: typeof user._id === 'object' && user._id !== null ? user._id.toString() : String(user._id),
        username: user.username,
        email: user.email,
        isAdmin: user.isAdmin,
        createdAt: user.createdAt.toISOString()
      }));
    } finally {
      if (client) await client.close();
    }
  },

  // Créer un nouvel utilisateur
  async create(userData: NewUserInput): Promise<UserOutput> {
    let client: MongoClient | null = null;
    try {
      client = await connectToMongoDB() as MongoClient;
      const db: Db = client.db("Access");
      const collection = db.collection('Users');
      
      // Vérifier si l'utilisateur existe déjà
      const existingUser = await collection.findOne({ email: userData.email });
      if (existingUser) {
        throw new Error(`Un utilisateur avec l'email ${userData.email} existe déjà`);
      }
      
      // Générer un mot de passe si non fourni
      const password = userData.password || this.generatePassword();
      const hashedPassword = this.hashPassword(password);
      const now = new Date();
      
      // Vous pouvez continuer à utiliser des chaînes personnalisées comme ID
      const customId = `user_${Date.now()}`;
      
      const newUser: User = {
        _id: customId,
        username: userData.username,
        email: userData.email,
        password: hashedPassword,
        isAdmin: userData.isAdmin || false,
        createdAt: now,
        updatedAt: now
      };
      
      await collection.insertOne(newUser as any);
      return {
        id: customId,
        username: newUser.username,
        email: newUser.email,
        isAdmin: newUser.isAdmin,
        createdAt: newUser.createdAt.toISOString()
      };
    } finally {
      if (client) await client.close();
    }
  },

  // Supprimer un utilisateur
  async delete(id: string): Promise<boolean> {
    let client: MongoClient | null = null;
    try {
      client = await connectToMongoDB() as MongoClient;
      const db: Db = client.db("Access");
      const collection = db.collection('Users');
      
      // Même logique que pour findById
      let query = {};
      if (ObjectId.isValid(id) && id.match(/^[0-9a-fA-F]{24}$/)) {
        query = { _id: new ObjectId(id) };
      } else {
        query = { _id: id };
      }
      
      const result = await collection.deleteOne(query);
      return result.deletedCount === 1;
    } finally {
      if (client) await client.close();
    }
  },

  // Mettre à jour un utilisateur
  async update(id: string, updates: Partial<Omit<User, '_id' | 'createdAt'>>): Promise<UserOutput | null> {
    let client: MongoClient | null = null;
    try {
      client = await connectToMongoDB() as MongoClient;
      const db: Db = client.db("Access");
      const collection = db.collection('Users');
      
      // Si le mot de passe est fourni, le hacher
      if (updates.password) {
        updates.password = this.hashPassword(updates.password);
      }
      updates.updatedAt = new Date();
      
      // Même logique que précédemment
      let query = {};
      if (ObjectId.isValid(id) && id.match(/^[0-9a-fA-F]{24}$/)) {
        query = { _id: new ObjectId(id) };
      } else {
        query = { _id: id };
      }
      
      const result = await collection.updateOne(
        query,
        { $set: updates }
      );
      
      if (result.matchedCount === 0) {
        return null;
      }
      
      const updatedUser = await collection.findOne(query);
      if (!updatedUser) return null;
      
      return {
        id: typeof updatedUser._id === 'object' && updatedUser._id !== null ? updatedUser._id.toString() : String(updatedUser._id),
        username: updatedUser.username,
        email: updatedUser.email,
        isAdmin: updatedUser.isAdmin,
        createdAt: updatedUser.createdAt.toISOString()
      };
    } finally {
      if (client) await client.close();
    }
  }
};