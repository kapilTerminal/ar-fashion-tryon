'use client';

import React, { useState } from 'react';
import Image from 'next/image';
import { useRouter } from 'next/navigation';
import { Sparkles, Bookmark, Star, Tag, Palette, Shield, Shirt, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import type { GarmentRecommendation, ClothType } from '@/lib/types';
import { useVtonStore } from '@/lib/store/useVtonStore';
import { useTryonStore } from '@/lib/tryon-store';
import { toast } from 'sonner';

interface GarmentCardProps {
  recommendation: GarmentRecommendation;
  personFile: File | null;
}

export const GarmentCard: React.FC<GarmentCardProps> = ({
  recommendation,
  personFile,
}) => {
  const router = useRouter();
  const [isLoadingTryOn, setIsLoadingTryOn] = useState(false);
  const [imageError, setImageError] = useState(false);

  const {
    name,
    category,
    color,
    style,
    occasion,
    similarity_score,
    garment_url,
    cutout_url,
    tryon_payload,
  } = recommendation;

  // Use cutout_url if available, else garment_url
  const imageUrl = cutout_url || garment_url;

  // Calculate percentage similarity
  const similarityPct = Math.round(similarity_score * 100);

  // Rating default: 5 stars
  const rating = recommendation.rating || 5;

  const handleTryOn = async () => {
    if (!personFile) {
      toast.error('Please upload your person photo first before trying on.');
      return;
    }

    setIsLoadingTryOn(true);
    try {
      // 1. Fetch recommended garment image as Blob/File
      const response = await fetch(imageUrl);
      if (!response.ok) {
        throw new Error('Failed to load recommended garment image');
      }
      const blob = await response.blob();
      const sanitizedName = name.replace(/[^a-z0-9]/gi, '_').toLowerCase();
      const garmentFile = new File([blob], `${sanitizedName}.png`, {
        type: blob.type || 'image/png',
      });

      // 2. Access Zustand stores
      const vtonStore = useVtonStore.getState();
      const tryonStore = useTryonStore.getState();

      // Switch mode to photo
      tryonStore.setMode('photo');

      // Set try-on path to NORMAL (single garment)
      vtonStore.setPath('NORMAL');

      // Set user's person image
      await vtonStore.setBody(personFile);

      // Set recommended garment file (skip auto classification as payload already specifies cloth_type)
      await vtonStore.setGarmentFile(garmentFile, true);

      // Determine cloth type from backend payload
      const clothType = (tryon_payload?.cloth_type || category || 'upper') as ClothType;
      vtonStore.setOptions({ clothType });

      // Advance step to PREVIEW mode
      vtonStore.setStep('PREVIEW');

      toast.success(`Loaded "${name}" into Try-On Studio! Redirecting...`);

      // 3. Navigate to existing Try-On page
      router.push('/try-on?mode=photo');
    } catch (err: unknown) {
      const error = err as Error;
      toast.error(`Failed to launch Virtual Try-On: ${error.message}`);
    } finally {
      setIsLoadingTryOn(false);
    }
  };

  return (
    <Card className="group relative flex flex-col overflow-hidden rounded-2xl border border-border/60 bg-card hover:shadow-xl transition-all duration-300 hover:border-primary/40">
      {/* Top Badge Overlay */}
      <div className="absolute top-3 left-3 right-3 z-10 flex items-center justify-between pointer-events-none">
        <Badge className="bg-background/90 backdrop-blur-md text-foreground border border-border shadow-sm text-xs font-semibold px-2.5 py-1">
          #{recommendation.rank} Match
        </Badge>
        <Badge
          className="bg-primary/90 backdrop-blur-md text-primary-foreground font-bold text-xs px-2.5 py-1 shadow-sm flex items-center gap-1"
        >
          <Sparkles className="w-3 h-3" />
          {similarityPct}% Score
        </Badge>
      </div>

      {/* Garment Image Area */}
      <div className="relative w-full h-64 sm:h-72 bg-muted/30 overflow-hidden flex items-center justify-center p-4">
        {!imageError && imageUrl ? (
          <Image
            src={imageUrl}
            alt={name}
            fill
            className="object-contain p-4 group-hover:scale-105 transition-transform duration-300"
            onError={() => setImageError(true)}
            unoptimized
          />
        ) : (
          <div className="flex flex-col items-center text-muted-foreground">
            <Shirt className="w-12 h-12 mb-2 opacity-50" />
            <span className="text-xs">Image unavailable</span>
          </div>
        )}
      </div>

      {/* Content Area */}
      <div className="flex flex-col flex-1 p-5 space-y-4">
        {/* Rating & Name */}
        <div>
          <div className="flex items-center gap-1 mb-1.5 text-amber-500">
            {Array.from({ length: 5 }).map((_, i) => (
              <Star
                key={i}
                className={`w-4 h-4 ${
                  i < rating ? 'fill-amber-400 text-amber-400' : 'text-muted-foreground/30'
                }`}
              />
            ))}
            <span className="text-xs text-muted-foreground font-medium ml-1">
              (5.0)
            </span>
          </div>

          <h3 className="font-bold text-lg leading-snug line-clamp-2 text-foreground group-hover:text-primary transition-colors">
            {name}
          </h3>
        </div>

        {/* Metadata Badges Grid */}
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div className="flex items-center gap-1.5 text-muted-foreground bg-muted/40 px-2.5 py-1.5 rounded-lg border border-border/40">
            <Shirt className="w-3.5 h-3.5 text-primary shrink-0" />
            <span className="capitalize truncate font-medium">{category}</span>
          </div>

          <div className="flex items-center gap-1.5 text-muted-foreground bg-muted/40 px-2.5 py-1.5 rounded-lg border border-border/40">
            <Palette className="w-3.5 h-3.5 text-blue-500 shrink-0" />
            <span className="capitalize truncate font-medium">{color}</span>
          </div>

          <div className="flex items-center gap-1.5 text-muted-foreground bg-muted/40 px-2.5 py-1.5 rounded-lg border border-border/40">
            <Tag className="w-3.5 h-3.5 text-purple-500 shrink-0" />
            <span className="capitalize truncate font-medium">{style}</span>
          </div>

          <div className="flex items-center gap-1.5 text-muted-foreground bg-muted/40 px-2.5 py-1.5 rounded-lg border border-border/40">
            <Shield className="w-3.5 h-3.5 text-emerald-500 shrink-0" />
            <span className="capitalize truncate font-medium">{occasion}</span>
          </div>
        </div>

        {/* Similarity Score Progress Bar */}
        <div className="space-y-1 pt-1">
          <div className="flex justify-between text-xs font-medium">
            <span className="text-muted-foreground">Match Score</span>
            <span className="text-primary font-bold">{similarityPct}%</span>
          </div>
          <div className="w-full bg-muted rounded-full h-2 overflow-hidden">
            <div
              className="bg-gradient-to-r from-blue-500 to-purple-600 h-full rounded-full transition-all duration-500"
              style={{ width: `${similarityPct}%` }}
            />
          </div>
        </div>

        {/* Action Buttons */}
        <div className="pt-2 flex items-center gap-2">
          {/* Large Try On Button */}
          <Button
            type="button"
            onClick={handleTryOn}
            disabled={isLoadingTryOn}
            className="flex-1 h-11 font-semibold text-sm bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white shadow-md hover:shadow-lg transition-all"
          >
            {isLoadingTryOn ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Preparing...
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4 mr-2" />
                Try On
              </>
            )}
          </Button>

          {/* Save Button (Disabled Placeholder) */}
          <Button
            type="button"
            variant="outline"
            disabled
            className="h-11 px-3 border-border/60 opacity-60 cursor-not-allowed"
            title="Save feature coming soon"
          >
            <Bookmark className="w-4 h-4 text-muted-foreground" />
          </Button>
        </div>
      </div>
    </Card>
  );
};
