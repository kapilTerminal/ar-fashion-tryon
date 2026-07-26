'use client';

import React, { useState } from 'react';
import Image from 'next/image';
import { Sparkles, Tag, Palette, Shield, Shirt, CheckCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';
import type { GarmentRecommendation } from '@/lib/types/recommendation';

interface GarmentCardProps {
  recommendation: GarmentRecommendation;
  isSelected?: boolean;
  onSelectGarment?: (garment: GarmentRecommendation) => void;
}

export const GarmentCard: React.FC<GarmentCardProps> = ({
  recommendation,
  isSelected = false,
  onSelectGarment,
}) => {
  const [imageError, setImageError] = useState(false);

  const {
    name,
    category,
    color,
    primary_color,
    style,
    occasion,
    similarity_score,
    final_score,
    garment_url,
    cutout_url,
    image_url,
  } = recommendation;

  // Determine display image URL
  const imageUrl = cutout_url || garment_url || image_url;

  // Primary color fallback
  const displayColor = primary_color || color || 'Unspecified';

  // Similarity percentage
  const similarityPct = Math.round(similarity_score * 100);

  // Final score percentage (fallback to similarity score if final_score not sent)
  const scoreVal = final_score !== undefined ? final_score : similarity_score;
  const finalScorePct = Math.round(scoreVal * 100);

  // Helper function to generate Match Badge text & star rating
  const getMatchBadge = (pct: number) => {
    if (pct >= 95) {
      return { label: 'Excellent Match', stars: '★★★★★', color: 'bg-emerald-500/10 text-emerald-600 border-emerald-500/30' };
    }
    if (pct >= 90) {
      return { label: 'Great Match', stars: '★★★★☆', color: 'bg-blue-500/10 text-blue-600 border-blue-500/30' };
    }
    if (pct >= 80) {
      return { label: 'Good Match', stars: '★★★★', color: 'bg-indigo-500/10 text-indigo-600 border-indigo-500/30' };
    }
    if (pct >= 70) {
      return { label: 'Suitable', stars: '★★★', color: 'bg-amber-500/10 text-amber-600 border-amber-500/30' };
    }
    return { label: 'Compatible', stars: '★★', color: 'bg-purple-500/10 text-purple-600 border-purple-500/30' };
  };

  const matchBadge = getMatchBadge(finalScorePct);

  const handleTryOnSelect = () => {
    if (onSelectGarment) {
      onSelectGarment(recommendation);
    }
  };

  return (
    <Card
      className={cn(
        'group relative flex flex-col overflow-hidden rounded-3xl border transition-all duration-300 bg-card/90 backdrop-blur-sm',
        'hover:-translate-y-1 hover:shadow-2xl hover:border-primary/40',
        isSelected
          ? 'ring-2 ring-primary border-primary shadow-xl bg-primary/5'
          : 'border-border/60 shadow-md'
      )}
    >
      {/* Top Badges Overlay */}
      <div className="absolute top-3 left-3 right-3 z-10 flex items-center justify-between pointer-events-none">
        <Badge className="bg-background/90 backdrop-blur-md text-foreground border border-border shadow-sm text-xs font-semibold px-2.5 py-1 rounded-xl">
          #{recommendation.rank} Match
        </Badge>
        {isSelected && (
          <Badge className="bg-primary text-primary-foreground font-bold text-xs px-2.5 py-1 shadow-md flex items-center gap-1 rounded-xl">
            <CheckCircle className="w-3.5 h-3.5" />
            Selected for Try-On
          </Badge>
        )}
      </div>

      {/* Garment Product Image */}
      <div className="relative w-full h-64 sm:h-72 bg-gradient-to-b from-muted/30 to-muted/60 overflow-hidden flex items-center justify-center p-4">
        {!imageError && imageUrl ? (
          <Image
            src={imageUrl}
            alt={name || 'Recommended Garment'}
            fill
            className="object-contain p-4 group-hover:scale-105 transition-transform duration-300"
            onError={() => setImageError(true)}
            unoptimized
          />
        ) : (
          <div className="flex flex-col items-center justify-center p-6 text-center text-muted-foreground">
            <Shirt className="w-12 h-12 mb-2 opacity-40" />
            <span className="text-xs font-medium">No Image Available</span>
          </div>
        )}
      </div>

      {/* Card Content Area */}
      <div className="flex flex-col flex-1 p-5 space-y-4">
        {/* Match Star Rating Badge */}
        <div className="flex items-center justify-between gap-2">
          <Badge variant="outline" className={cn('text-xs font-bold px-2.5 py-1 rounded-xl border', matchBadge.color)}>
            <span className="mr-1.5">{matchBadge.stars}</span>
            {matchBadge.label}
          </Badge>

          <span className="text-xs font-bold text-primary">
            {finalScorePct}% Final Score
          </span>
        </div>

        {/* Product Name */}
        <div>
          <h3 className="font-bold text-base sm:text-lg leading-snug line-clamp-2 text-foreground group-hover:text-primary transition-colors">
            {name || 'Fashion Garment'}
          </h3>
        </div>

        {/* Product Metadata Badges Grid */}
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div className="flex items-center gap-1.5 text-muted-foreground bg-muted/40 px-2.5 py-1.5 rounded-xl border border-border/40">
            <Shirt className="w-3.5 h-3.5 text-primary shrink-0" />
            <span className="capitalize truncate font-medium">{category}</span>
          </div>

          <div className="flex items-center gap-1.5 text-muted-foreground bg-muted/40 px-2.5 py-1.5 rounded-xl border border-border/40">
            <Palette className="w-3.5 h-3.5 text-blue-500 shrink-0" />
            <span className="capitalize truncate font-medium">{displayColor}</span>
          </div>

          <div className="flex items-center gap-1.5 text-muted-foreground bg-muted/40 px-2.5 py-1.5 rounded-xl border border-border/40">
            <Tag className="w-3.5 h-3.5 text-purple-500 shrink-0" />
            <span className="capitalize truncate font-medium">{style}</span>
          </div>

          <div className="flex items-center gap-1.5 text-muted-foreground bg-muted/40 px-2.5 py-1.5 rounded-xl border border-border/40">
            <Shield className="w-3.5 h-3.5 text-emerald-500 shrink-0" />
            <span className="capitalize truncate font-medium">{occasion}</span>
          </div>
        </div>

        {/* Similarity Score Progress Bar */}
        <div className="space-y-1 pt-1">
          <div className="flex justify-between text-xs font-semibold">
            <span className="text-muted-foreground">Similarity Score</span>
            <span className="text-foreground">{similarityPct}%</span>
          </div>
          <div className="w-full bg-muted rounded-full h-2 overflow-hidden">
            <div
              className="bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-600 h-full rounded-full transition-all duration-500"
              style={{ width: `${similarityPct}%` }}
            />
          </div>
        </div>

        {/* Try On Button */}
        <div className="pt-2">
          <Button
            type="button"
            onClick={handleTryOnSelect}
            variant={isSelected ? 'default' : 'secondary'}
            className={cn(
              'w-full h-11 font-bold text-sm rounded-2xl transition-all duration-200 shadow-md',
              isSelected
                ? 'bg-primary text-primary-foreground hover:bg-primary/90'
                : 'bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white'
            )}
          >
            {isSelected ? (
              <>
                <CheckCircle className="w-4 h-4 mr-2" />
                Selected for Try-On
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4 mr-2" />
                Try On
              </>
            )}
          </Button>
        </div>
      </div>
    </Card>
  );
};
