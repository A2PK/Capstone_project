export interface ModelAI {
    id: string;       
    created_at: string;        
    updated_at: string;     
    deleted_at: string | null; 
    name: string;     
    version: string;          
    file_path: string;         
    description: string;     
    trained_at: string;       
    station_id: string;         
    availability: boolean;        
    parameter_list: string[];    
  }
  