import { useState } from "react";
import { Navbar } from "@/components/Navbar";
import { TicketUpload } from "@/components/TicketUpload";
import { ChatInterface } from "@/components/ChatInterface";
import { TicketDetails } from "@/components/TicketsDetails";
import StarfieldBackground from "@/components/StarfieldBackground";
import { CosmicElements, GlowingOrb } from "@/components/CosmicElements";
import { motion } from "framer-motion";
import { useTheme } from "@/hooks/useTheme";
import { cn } from "@/lib/utils";
import { SearchHistory } from "@/components/SearchHistory";

const Dashboard = () => {
  const [initialMessage, setInitialMessage] = useState<string | undefined>();
  const [ticketIds, setTicketIds] = useState<string[] | undefined>();
  const [ticketData, setTicketData] = useState<Record<string, any> | null>(null);
  const [loadingAnalysis, setLoadingAnalysis] = useState(false);
  
  const { theme } = useTheme();
  const isDark = theme === "dark";
  
  const handleFileUploaded = (message: string, ids?: string[]) => {
    setInitialMessage(message);
    setTicketIds(ids);
  };
  
  const handleTicketDataExtracted = (data: Record<string, any> | null, loading: boolean) => {
    setTicketData(data);
    setLoadingAnalysis(loading);
    
    // Si nous avons des données de ticket et que l'analyse est en cours,
    // nous pouvons afficher un message explicatif
    if (data && loading) {
      console.log("Données du ticket en cours d'analyse:", data);
    }
  }
  
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
        delayChildren: 0.2
      }
    }
  };
  
  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: { duration: 0.5, ease: "easeOut" }
    }
  };

  return (
    <div className="min-h-screen relative overflow-x-hidden font-sourcesans text-foreground">
      {isDark && <StarfieldBackground />}
      {isDark && <CosmicElements />}
      <Navbar />
      <main className="container mx-auto pt-16 px-4 relative z-10 pb-10">
        <motion.div
          className="mx-auto"
          variants={containerVariants}
          initial="hidden"
          animate="visible"
        >
          <motion.div variants={itemVariants} className="text-center mb-2">
            <h1 className={cn(
              "text-xl md:text-2xl font-bold text-gradient mb-0",
              isDark ? "text-white" : "text-gray-800"
            )}>
              Ticket AI Wizard
            </h1>
          </motion.div>
          
          <div className="grid gap-4 md:grid-cols-7 lg:gap-6">
            {/* Section historique à gauche */}
            <motion.div
              variants={itemVariants}
              className="md:col-span-2"
            >
              <SearchHistory />
            </motion.div>

            {/* Section principale */}
            <motion.div
              variants={itemVariants}
              className="md:col-span-5"
            >
              <motion.div variants={itemVariants}>
                <div className={cn(
                  "mb-4 p-4 rounded-xl border",
                  isDark
                    ? "bg-card/20 border-white/10"
                    : "bg-white border-gray-200"
                )}>
                  <h2 className={cn(
                    "text-lg font-medium mb-3",
                    isDark ? "text-white" : "text-gray-800"
                  )}>
                    Importer votre ticket
                  </h2>
                  <TicketUpload 
                    onFileUploaded={handleFileUploaded} 
                    onTicketDataExtracted={handleTicketDataExtracted} 
                  />
                </div>
              </motion.div>

              {/* Affichage des détails du ticket pendant l'analyse */}
              {(loadingAnalysis || ticketData) && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5 }}
                  className="mb-4"
                >
                  <TicketDetails 
                    ticketData={ticketData} 
                    loading={loadingAnalysis} 
                  />
                </motion.div>
              )}

              {initialMessage && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5 }}
                >
                  <ChatInterface 
                    initialMessage={initialMessage} 
                    ticketIds={ticketIds} 
                  />
                </motion.div>
              )}
            </motion.div>
          </div>
        </motion.div>
      </main>
      
      {isDark && (
        <>
          <GlowingOrb className="fixed top-1/4 left-1/5 -z-10" size={250} color="rgba(79, 70, 229, 0.08)" />
          <GlowingOrb className="fixed bottom-1/4 right-1/5 -z-10" size={300} color="rgba(124, 58, 237, 0.06)" />
        </>
      )}
    </div>
  );
};

export default Dashboard;