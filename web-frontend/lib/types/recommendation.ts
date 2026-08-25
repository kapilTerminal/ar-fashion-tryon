/**
 * Recommendation Engine Data Models & Types
 * Stage 5.1
 */

export interface RecommendationRequest {
  person_image: File;
  gender: string;
  occasion: string;
  preferred_color: string;
  preferred_style: string;
  style_preference: string;
  body_type: string;
  skin_tone: string;
}

export interface UserPreferences {
  gender: string;
  occasion: string;
  preferredColor: string;
  preferredStyle: string;
  stylePreference: string;
  bodyType: string;
  skinTone: string;
}

export interface RawGarmentMetadata {
  articleType?: string;
  baseColour?: string;
  usage?: string;
  [key: string]: any;
}

export interface RawGarmentRecommendation {
  id?: string | number;
  garment_id?: string | number;
  productDisplayName?: string;
  name?: string;
  similarity?: number;
  similarity_score?: number;
  final_score?: number;
  metadata?: RawGarmentMetadata;
  image_url?: string;
  garment_url?: string;
  cutout_url?: string;
  [key: string]: any;
}

export interface RawBackendRecommendationResponse {
  success?: boolean;
  request_id?: string;
  user_profile?: any;
  recommendations?: RawGarmentRecommendation[];
  query_parameters?: RecommendationQueryParams;
  message?: string;
  [key: string]: any;
}

export interface GarmentRecommendation {
  rank: number;
  garment_id: string;
  name: string;
  category: string;
  occasion: string;
  color?: string;
  primary_color?: string;
  style?: string;
  similarity_score: number;
  final_score?: number;
  garment_url?: string;
  cutout_url?: string;
  image_url?: string;
  tryon_payload?: {
    cloth_type: string;
    process_garment?: boolean;
  };
}

export interface RecommendationQueryParams {
  gender?: string;
  occasion?: string;
  preferred_color?: string;
  preferred_style?: string;
  style_preference?: string;
  body_type?: string;
  skin_tone?: string;
}

export interface RecommendationResponse {
  success: boolean;
  request_id: string;
  person_image_url?: string;
  query_parameters?: RecommendationQueryParams;
  total_candidates_found?: number;
  recommendations: GarmentRecommendation[];
  message?: string;
}

