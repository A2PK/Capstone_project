export interface Article {
    id: string;
    authorId: string;
    title: string;
    content: string;
    badge: string; // ('common', 'good', 'danger')
    pictureUrl: string; 
    fileIds: string[]; 
    createdAt: string; 
    updatedAt: string; 
    deletedAt: string | null; 
  }
  export type CreateArticleData = Omit<Article, 'id' | 'createdAt' | 'updatedAt' | 'deletedAt'> & {};