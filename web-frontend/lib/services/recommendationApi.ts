import { RECOMMEND_ENDPOINT } from '@/lib/config/api';
import type { RecommendResponse, RecommendationRequestParams } from '@/lib/types';

/**
 * Send request to FastAPI /recommend endpoint for AI-powered outfit recommendations.
 * 
 * @param params Object containing person_image (File), occasion, preferred_color, and style_preference
 * @returns Promise<RecommendResponse>
 */
export async function getFashionRecommendations(
  params: RecommendationRequestParams
): Promise<RecommendResponse> {
  const { person_image, occasion, preferred_color, style_preference } = params;

  if (!person_image) {
    throw new Error('Please upload a person photo.');
  }

  const formData = new FormData();
  formData.append('person_image', person_image);
  formData.append('occasion', occasion.toLowerCase());
  formData.append('preferred_color', preferred_color.toLowerCase());
  formData.append('style_preference', style_preference.toLowerCase());

  let response: Response;
  try {
    response = await fetch(RECOMMEND_ENDPOINT, {
      method: 'POST',
      body: formData,
    });
  } catch (err: unknown) {
    const error = err as Error;
    throw new Error(
      `Unable to connect to the recommendation service at ${RECOMMEND_ENDPOINT}. Please verify the backend server is running. (${error.message})`
    );
  }

  if (!response.ok) {
    let errorDetail = `Request failed with status ${response.status}`;
    try {
      const errJson = await response.json();
      if (errJson.detail) {
        errorDetail = typeof errJson.detail === 'string' ? errJson.detail : JSON.stringify(errJson.detail);
      } else if (errJson.message) {
        errorDetail = errJson.message;
      }
    } catch {
      // Ignore JSON parse errors for non-JSON responses
    }
    throw new Error(errorDetail);
  }

  const data: RecommendResponse = await response.json();
  return data;
}
