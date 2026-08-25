'use client';

import React, { useState, useRef } from 'react';
import Image from 'next/image';
import { useRouter } from 'next/navigation';
import { PageTransition } from '@/components/ui/page-transition';
import { RecommendationForm, FormSubmitData } from '@/components/recommend/RecommendationForm';
import { RecommendationGrid } from '@/components/recommend/RecommendationGrid';
import { getFashionRecommendations } from '@/lib/services/recommendationApi';
import type { RecommendationResponse, GarmentRecommendation } from '@/lib/types/recommendation';
import type { ClothType } from '@/lib/types';
import { useTryonStore } from '@/lib/tryon-store';
import { useVtonStore } from '@/lib/store/useVtonStore';
import { Sparkles, AlertCircle, CheckCircle2 } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Card } from '@/components/ui/card';
import { toast } from 'sonner';

function resolveGarmentUrl(url?: string | null): string | null {
  if (!url) return null;

  if (/^https?:\/\/res\.cloudinary\.com/i.test(url)) {
    return url;
  }

  const host = typeof window !== 'undefined' && window.location.hostname
    ? window.location.hostname
    : '127.0.0.1';

  const backendBase = `http://${host}:5000`;
  const cleanUrl = url.replace(/^https?:\/\/(127\.0\.0\.1|localhost):(8000|5000)\//i, `${backendBase}/`);

  if (/^https?:\/\//i.test(cleanUrl)) {
    return cleanUrl;
  }

  const cleanPath = cleanUrl.startsWith('/') ? cleanUrl : `/${cleanUrl}`;
  return `${backendBase}${cleanPath}`;
}

export default function RecommendPage() {
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [recommendResponse, setRecommendResponse] = useState<RecommendationResponse | null>(null);
  const [personFile, setPersonFile] = useState<File | null>(null);

  // Selected Garment State
  const [selectedGarment, setSelectedGarment] = useState<GarmentRecommendation | null>(null);
  const [selectedGarmentId, setSelectedGarmentId] = useState<string | null>(null);
  const [selectedGarmentImage, setSelectedGarmentImage] = useState<string | null>(null);

  const resultsRef = useRef<HTMLDivElement>(null);

  const handleSubmit = async (formData: FormSubmitData) => {
    if (!formData.personFile) {
      toast.error('Please upload a person image first.');
      return;
    }

    setPersonFile(formData.personFile);
    setIsLoading(true);
    setError(null);

    try {
      const response = await getFashionRecommendations({
        person_image: formData.personFile,
        gender: formData.preferences.gender,
        occasion: formData.preferences.occasion,
        preferred_color: formData.preferences.preferredColor,
        preferred_style: formData.preferences.preferredStyle,
        style_preference: formData.preferences.stylePreference,
        body_type: formData.preferences.bodyType,
        skin_tone: formData.preferences.skinTone,
      });

      setRecommendResponse(response);
      toast.success(
        `Found ${response.recommendations.length} recommendations matching your style!`
      );

      // Smooth scroll to results
      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 150);
    } catch (err: unknown) {
      const errorObj = err as Error;
      const errorMsg = errorObj.message || 'Failed to fetch outfit recommendations.';
      setError(errorMsg);
      toast.error(errorMsg);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSelectGarment = async (garment: GarmentRecommendation) => {
    console.log('🔬 [handleSelectGarment START]', { garment });
    const id = garment.garment_id || String(garment.rank);
    const rawImgUrl = garment.cutout_url || garment.garment_url || garment.image_url || null;
    const resolvedImgUrl = resolveGarmentUrl(rawImgUrl);
    console.log('🔬 [handleSelectGarment URLs]', { rawImgUrl, resolvedImgUrl });

    setSelectedGarment(garment);
    setSelectedGarmentId(id);
    setSelectedGarmentImage(resolvedImgUrl);

    try {
      useTryonStore.getState().setMode('photo');

      const vtonStore = useVtonStore.getState();
      vtonStore.setPath('NORMAL');

      if (personFile) {
        await vtonStore.setBody(personFile);
      }

      if (resolvedImgUrl) {
        await vtonStore.setGarmentUrl(resolvedImgUrl, id);
      }

      const catLower = (garment.category || '').toLowerCase();
      const clothType: ClothType =
        (garment.tryon_payload?.cloth_type as ClothType) ||
        (catLower.includes('bottom') || catLower.includes('pant') || catLower.includes('trouser') || catLower.includes('skirt')
          ? 'lower'
          : catLower.includes('dress') || catLower.includes('overall')
            ? 'overall'
            : 'upper');

      vtonStore.setOptions({ clothType });
      vtonStore.setStep(personFile && resolvedImgUrl ? 'GENERATE' : 'BODY');

      toast.success(`Selected "${garment.name || 'Garment'}" for Try-On preview! Redirecting...`);
      router.push('/try-on');
    } catch (err) {
      console.error('Failed VTON handoff:', err);
      router.push('/try-on');
    }
  };

  const queryParams = recommendResponse?.query_parameters;
  const querySummary = queryParams
    ? `Filtered by ${queryParams.gender || 'unisex'} • ${queryParams.occasion || 'casual'} occasion • ${queryParams.preferred_color || 'preferred'} color • ${queryParams.preferred_style || 'style'} preference.`
    : undefined;

  return (
    <PageTransition>
      <div className="container mx-auto px-4 py-8 max-w-6xl space-y-10 pb-20">
        {/* Page Header */}
        <div className="text-center space-y-3 max-w-2xl mx-auto">
          <Badge
            variant="outline"
            className="px-3.5 py-1 border-primary/40 text-primary bg-primary/5 text-xs font-bold rounded-full shadow-sm"
          >
            <Sparkles className="w-3.5 h-3.5 mr-1.5 inline-block text-primary" />
            AI Fashion Intelligence Engine
          </Badge>

          <h1 className="text-3xl sm:text-4xl md:text-5xl font-extrabold tracking-tight bg-gradient-to-r from-blue-600 via-purple-600 to-indigo-600 bg-clip-text text-transparent">
            Outfit Recommendation
          </h1>

          <p className="text-muted-foreground text-sm sm:text-base leading-relaxed">
            Upload your photo and select your style preferences to receive personalized outfit recommendations powered by AI.
          </p>
        </div>

        {/* Input Form Section */}
        <div className="max-w-4xl mx-auto">
          <RecommendationForm onSubmit={handleSubmit} isLoading={isLoading} />
        </div>

        {/* Selected Garment Stage 5.2 Banner */}
        {selectedGarment && (
          <div className="max-w-4xl mx-auto">
            <Card className="p-4 rounded-2xl border border-primary/30 bg-primary/5 flex items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                {selectedGarmentImage ? (
                  <div className="relative w-12 h-12 rounded-xl border border-primary/20 overflow-hidden bg-background shrink-0">
                    <Image
                      src={selectedGarmentImage}
                      alt={selectedGarment.name || 'Selected Garment'}
                      fill
                      className="object-contain p-1"
                      unoptimized
                    />
                  </div>
                ) : (
                  <div className="w-10 h-10 rounded-xl bg-primary text-primary-foreground flex items-center justify-center font-bold">
                    <CheckCircle2 className="w-5 h-5" />
                  </div>
                )}
                <div>
                  <h4 className="font-bold text-sm text-foreground flex items-center gap-2">
                    Garment Selected: {selectedGarment.name}
                  </h4>
                  <p className="text-xs text-muted-foreground">
                    Saved in React state (ID: {selectedGarmentId}). Ready for Stage 5.2 Virtual Try-On execution.
                  </p>
                </div>
              </div>
              <Badge variant="outline" className="text-xs font-semibold px-3 py-1 border-primary/40 text-primary">
                Stage 5.1 Ready
              </Badge>
            </Card>
          </div>
        )}

        {/* Error Alert Section */}
        {error && (
          <div className="max-w-4xl mx-auto">
            <Alert variant="destructive" className="rounded-2xl border-destructive/50 bg-destructive/10">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle className="font-bold">Recommendation Error</AlertTitle>
              <AlertDescription className="text-xs sm:text-sm mt-1 leading-relaxed">
                {error}
              </AlertDescription>
            </Alert>
          </div>
        )}

        {/* Recommendation Results Grid Section */}
        <div ref={resultsRef} className="max-w-5xl mx-auto">
          <RecommendationGrid
            recommendations={recommendResponse?.recommendations || null}
            isLoading={isLoading}
            selectedGarmentId={selectedGarmentId}
            onSelectGarment={handleSelectGarment}
            querySummary={querySummary}
          />
        </div>
      </div>
    </PageTransition>
  );
}
