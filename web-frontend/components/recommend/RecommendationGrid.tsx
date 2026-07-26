'use client';

import React from 'react';
import { Shirt, Sparkles } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { Badge } from '@/components/ui/badge';
import { GarmentCard } from './GarmentCard';
import type { GarmentRecommendation } from '@/lib/types/recommendation';

interface RecommendationGridProps {
  recommendations: GarmentRecommendation[] | null;
  isLoading: boolean;
  selectedGarmentId: string | null;
  onSelectGarment: (garment: GarmentRecommendation) => void;
  querySummary?: string;
}

export const RecommendationGrid: React.FC<RecommendationGridProps> = ({
  recommendations,
  isLoading,
  selectedGarmentId,
  onSelectGarment,
  querySummary,
}) => {
  if (isLoading) {
    return (
      <div className="space-y-6 pt-4">
        <div className="flex items-center justify-between border-b pb-4">
          <div className="space-y-2">
            <Skeleton className="h-7 w-56 rounded-lg" />
            <Skeleton className="h-4 w-80 rounded-lg" />
          </div>
          <Skeleton className="h-7 w-28 rounded-full" />
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
          {Array.from({ length: 6 }).map((_, i) => (
            <Card key={i} className="p-4 space-y-4 rounded-3xl border border-border/60">
              <Skeleton className="w-full h-64 rounded-2xl" />
              <div className="space-y-2">
                <Skeleton className="h-5 w-3/4 rounded-lg" />
                <Skeleton className="h-4 w-1/2 rounded-lg" />
              </div>
              <div className="grid grid-cols-2 gap-2">
                <Skeleton className="h-8 rounded-xl" />
                <Skeleton className="h-8 rounded-xl" />
              </div>
              <Skeleton className="h-11 w-full rounded-2xl" />
            </Card>
          ))}
        </div>
      </div>
    );
  }

  if (!recommendations) {
    return null;
  }

  if (recommendations.length === 0) {
    return (
      <div className="text-center p-12 sm:p-16 border border-dashed rounded-3xl bg-muted/20 space-y-4 max-w-2xl mx-auto my-8">
        <div className="w-16 h-16 rounded-full bg-muted flex items-center justify-center mx-auto opacity-60">
          <Shirt className="w-8 h-8 text-muted-foreground" />
        </div>
        <h3 className="font-bold text-xl text-foreground">No recommendations yet</h3>
        <p className="text-sm text-muted-foreground max-w-md mx-auto leading-relaxed">
          We couldn&apos;t find any outfits matching your exact preferences. Try adjusting your preferred color, style, or occasion parameters.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6 pt-4">
      {/* Results Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between border-b pb-4 gap-3">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-2 text-foreground">
            <Sparkles className="w-6 h-6 text-primary" />
            Recommended Outfits
          </h2>
          {querySummary && (
            <p className="text-xs text-muted-foreground mt-1 leading-relaxed">
              {querySummary}
            </p>
          )}
        </div>

        <Badge variant="secondary" className="text-xs px-3 py-1.5 font-bold rounded-full border border-border">
          {recommendations.length} Items Found
        </Badge>
      </div>

      {/* Cards Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
        {recommendations.map((rec, index) => (
          <GarmentCard
            key={`${rec.garment_id}-${index}`}
            recommendation={rec}
            isSelected={selectedGarmentId === rec.garment_id}
            onSelectGarment={onSelectGarment}
          />
        ))}
      </div>
    </div>
  );
};
