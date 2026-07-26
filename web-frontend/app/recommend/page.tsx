'use client';

import React, { useState, useRef } from 'react';
import { PageTransition } from '@/components/ui/page-transition';
import { RecommendationForm, RecommendationFormData } from '@/components/recommend/RecommendationForm';
import { GarmentCard } from '@/components/recommend/GarmentCard';
import { getFashionRecommendations } from '@/lib/services/recommendationApi';
import type { RecommendResponse } from '@/lib/types';
import { Sparkles, AlertCircle, Shirt } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { toast } from 'sonner';

export default function RecommendPage() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [recommendResponse, setRecommendResponse] = useState<RecommendResponse | null>(null);
  const [activePersonFile, setActivePersonFile] = useState<File | null>(null);

  const resultsRef = useRef<HTMLDivElement>(null);

  const handleSubmit = async (formData: RecommendationFormData) => {
    if (!formData.personFile) {
      toast.error('Please upload a person image first.');
      return;
    }

    setIsLoading(true);
    setError(null);
    setActivePersonFile(formData.personFile);

    try {
      const response = await getFashionRecommendations({
        person_image: formData.personFile,
        occasion: formData.occasion,
        preferred_color: formData.preferredColor,
        style_preference: formData.stylePreference,
      });

      setRecommendResponse(response);
      toast.success(
        `Found ${response.recommendations.length} recommendations matching your style!`
      );

      // Smooth scroll to results
      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 100);
    } catch (err: unknown) {
      const errorObj = err as Error;
      const errorMsg = errorObj.message || 'Failed to fetch outfit recommendations.';
      setError(errorMsg);
      toast.error(errorMsg);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <PageTransition>
      <div className="container mx-auto px-4 py-8 max-w-6xl space-y-10 pb-20">
        {/* Page Header */}
        <div className="text-center space-y-3 max-w-2xl mx-auto">
          <Badge
            variant="outline"
            className="px-3 py-1 border-primary/40 text-primary bg-primary/5 text-xs font-semibold rounded-full"
          >
            <Sparkles className="w-3.5 h-3.5 mr-1.5 inline-block" />
            AI Fashion Intelligence
          </Badge>

          <h1 className="text-3xl sm:text-4xl md:text-5xl font-extrabold tracking-tight bg-gradient-to-r from-blue-600 via-purple-600 to-indigo-600 bg-clip-text text-transparent">
            Fashion Recommendation
          </h1>

          <p className="text-muted-foreground text-sm sm:text-base leading-relaxed">
            Upload your photo and receive AI-powered outfit recommendations.
          </p>
        </div>

        {/* Input Form Card */}
        <div className="max-w-3xl mx-auto">
          <RecommendationForm onSubmit={handleSubmit} isLoading={isLoading} />
        </div>

        {/* Error Alert */}
        {error && (
          <div className="max-w-3xl mx-auto">
            <Alert variant="destructive" className="rounded-xl border-destructive/50 bg-destructive/10">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle className="font-semibold">Recommendation Error</AlertTitle>
              <AlertDescription className="text-xs sm:text-sm mt-1">
                {error}
              </AlertDescription>
            </Alert>
          </div>
        )}

        {/* Results Section */}
        <div ref={resultsRef} className="space-y-6 pt-4">
          {recommendResponse && (
            <>
              <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between border-b pb-4 gap-2">
                <div>
                  <h2 className="text-2xl font-bold flex items-center gap-2">
                    <Shirt className="w-6 h-6 text-primary" />
                    Recommended Outfits
                  </h2>
                  <p className="text-xs text-muted-foreground">
                    Based on your photo, {recommendResponse.query_parameters.occasion} occasion,{' '}
                    {recommendResponse.query_parameters.preferred_color} color preference, and{' '}
                    {recommendResponse.query_parameters.style_preference} style.
                  </p>
                </div>

                <Badge variant="secondary" className="text-xs px-3 py-1 font-semibold">
                  {recommendResponse.recommendations.length} Items Found
                </Badge>
              </div>

              {recommendResponse.recommendations.length > 0 ? (
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
                  {recommendResponse.recommendations.map((rec) => (
                    <GarmentCard
                      key={rec.garment_id || rec.rank}
                      recommendation={rec}
                      personFile={activePersonFile}
                    />
                  ))}
                </div>
              ) : (
                <div className="text-center p-12 border border-dashed rounded-2xl bg-muted/20 space-y-3">
                  <Shirt className="w-12 h-12 text-muted-foreground mx-auto opacity-50" />
                  <h3 className="font-semibold text-lg">No garments found</h3>
                  <p className="text-xs text-muted-foreground max-w-sm mx-auto">
                    Try adjusting your preferred color or style parameters to discover more matches.
                  </p>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </PageTransition>
  );
}
