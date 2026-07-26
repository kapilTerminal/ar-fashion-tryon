import { RECOMMEND_ENDPOINT } from '@/lib/config/api';
import type {
  RecommendationRequest,
  RecommendationResponse,
  GarmentRecommendation,
  RawBackendRecommendationResponse,
  RawGarmentRecommendation,
} from '@/lib/types/recommendation';

/**
 * Send request to FastAPI /recommend endpoint for AI-powered outfit recommendations.
 * Transforms raw backend response format into frontend model structure.
 *
 * @param request RecommendationRequest object
 * @param timeoutMs Optional timeout in milliseconds (default: 60000ms)
 * @returns Promise<RecommendationResponse>
 */
export async function getFashionRecommendations(
  request: RecommendationRequest,
  timeoutMs: number = 60000
): Promise<RecommendationResponse> {
  const {
    person_image,
    gender,
    occasion,
    preferred_color,
    preferred_style,
    style_preference,
    body_type,
    skin_tone,
  } = request;

  if (!person_image) {
    throw new Error('Please upload a person photo.');
  }

  const formData = new FormData();
  formData.append('person_image', person_image);
  formData.append('gender', (gender || 'unisex').toLowerCase());
  formData.append('occasion', (occasion || 'casual').toLowerCase());
  formData.append('preferred_color', (preferred_color || 'black').toLowerCase());
  formData.append('preferred_style', (preferred_style || 'casual').toLowerCase());
  formData.append('style_preference', (style_preference || 'modern').toLowerCase());
  formData.append('body_type', (body_type || 'rectangle').toLowerCase());
  formData.append('skin_tone', (skin_tone || 'medium').toLowerCase());

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  let response: Response;
  try {
    response = await fetch(RECOMMEND_ENDPOINT, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    });
  } catch (err: unknown) {
    clearTimeout(timeoutId);
    if (err instanceof Error && err.name === 'AbortError') {
      throw new Error('Recommendation request timed out. Please check your backend connection and try again.');
    }
    const error = err as Error;
    throw new Error(
      `Unable to connect to the recommendation service at ${RECOMMEND_ENDPOINT}. Please verify the backend server is running. (${error.message})`
    );
  } finally {
    clearTimeout(timeoutId);
  }

  if (!response.ok) {
    let errorDetail = `Backend server error (${response.status})`;
    try {
      const errJson = await response.json();
      if (errJson.detail) {
        errorDetail = typeof errJson.detail === 'string' ? errJson.detail : JSON.stringify(errJson.detail);
      } else if (errJson.message) {
        errorDetail = errJson.message;
      }
    } catch {
      // Non-JSON error response
    }
    throw new Error(errorDetail);
  }

  try {
    const rawData: RawBackendRecommendationResponse | RawGarmentRecommendation[] = await response.json();
    if (typeof rawData !== 'object' || rawData === null) {
      throw new Error('Received malformed response format from backend server.');
    }

    const rawList: RawGarmentRecommendation[] = Array.isArray(rawData)
      ? rawData
      : Array.isArray(rawData.recommendations)
      ? rawData.recommendations
      : [];

    const mappedRecommendations: GarmentRecommendation[] = rawList.map((item, index) => {
      const garment_id = String(item.id ?? item.garment_id ?? index);
      const name = item.productDisplayName || item.name || 'Garment';
      const category = item.metadata?.articleType || item.category || 'Apparel';
      const primary_color = item.metadata?.baseColour || item.primary_color || item.color || 'Unspecified';
      const occasion = item.metadata?.usage || item.occasion || 'Casual';
      const similarity_score = typeof item.similarity === 'number' ? item.similarity : (item.similarity_score ?? 0);
      const final_score = typeof item.final_score === 'number' ? item.final_score : similarity_score;

      const generatedImageUrl = item.image_url || item.garment_url || `http://127.0.0.1:8000/images/${garment_id}.jpg`;

      return {
        rank: item.rank || index + 1,
        garment_id,
        name,
        category,
        occasion,
        color: primary_color,
        primary_color,
        style: item.metadata?.subCategory || item.style || occasion,
        similarity_score,
        final_score,
        image_url: generatedImageUrl,
        garment_url: item.garment_url || generatedImageUrl,
        cutout_url: item.cutout_url,
        tryon_payload: item.tryon_payload,
      };
    });

    return {
      success: true,
      request_id: !Array.isArray(rawData) && rawData.request_id ? rawData.request_id : 'rec_' + Date.now(),
      query_parameters: !Array.isArray(rawData) ? rawData.query_parameters : undefined,
      total_candidates_found: mappedRecommendations.length,
      recommendations: mappedRecommendations,
    };
  } catch (jsonErr: unknown) {
    if (jsonErr instanceof Error && jsonErr.message.includes('malformed')) {
      throw jsonErr;
    }
    throw new Error('Failed to parse recommendation response from backend server.');
  }
}

