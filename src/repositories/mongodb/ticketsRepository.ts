import { connectToMongoDB } from "@/api/mongodb";
import { MongoClient, Db } from 'mongodb'; // Ajout des imports nécessaires

export interface Ticket {
  _id: string;
  problem: string;
  solution: string;
  keywords: string[];
  created_at: Date;
  updated_at: Date;
}

// Définir le nom de la base de données
const DB_NAME = "Access";

export const ticketsRepository = {
  async findSimilarTickets(ticketText: string): Promise<Ticket[]> {
    let client: MongoClient | null = null;
    try {
      client = await connectToMongoDB() as MongoClient;
      const db: Db = client.db(DB_NAME);
      const collection = db.collection('tickets');
      return (await collection.find().toArray()).map(doc => ({
        _id: doc._id.toString(),
        problem: doc.problem,
        solution: doc.solution,
        keywords: doc.keywords,
        created_at: doc.created_at,
        updated_at: doc.updated_at
      })) as Ticket[];
    } finally {
      if (client) await client.close();
    }
  },
  
  async saveTicket(ticket: Omit<Ticket, '_id' | 'created_at' | 'updated_at'>): Promise<Ticket> {
    let client: MongoClient | null = null;
    try {
      client = await connectToMongoDB() as MongoClient;
      const db: Db = client.db(DB_NAME);
      const collection = db.collection('tickets');
      const now = new Date();
      const newTicket = {
        ...ticket,
        created_at: now,
        updated_at: now
      };
      const result = await collection.insertOne(newTicket);
      return { 
        ...newTicket, 
        _id: result.insertedId.toString() 
      } as Ticket;
    } finally {
      if (client) await client.close();
    }
  }
};