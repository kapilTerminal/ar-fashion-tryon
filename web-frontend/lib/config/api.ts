/**
 * Global API Configuration
 * Provides centralized API_BASE_URL configuration for recommendation and virtual try-on services.
 */

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_RECOMMENDATION_API_URL ||
  process.env.NEXT_PUBLIC_VTON_API_URL ||
  process.env.NEXT_PUBLIC_API_URL ||
  'http://localhost:5000';

export const RECOMMEND_ENDPOINT = `${API_BASE_URL}/recommend`;
