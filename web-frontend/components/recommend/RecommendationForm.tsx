'use client';

import React, { useState } from 'react';
import { Sparkles, Calendar, Palette, Shirt, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { PersonImageUploader } from './PersonImageUploader';
import { toast } from 'sonner';

export interface RecommendationFormData {
  personFile: File | null;
  previewUrl: string | null;
  occasion: string;
  preferredColor: string;
  stylePreference: string;
}

interface RecommendationFormProps {
  onSubmit: (data: RecommendationFormData) => void;
  isLoading: boolean;
}

const OCCASIONS = ['Casual', 'Formal', 'Office', 'Party', 'Wedding'];
const COLORS = ['Black', 'White', 'Blue', 'Red', 'Green', 'Brown'];
const STYLES = ['Streetwear', 'Minimal', 'Formal', 'Traditional', 'Sport'];

export const RecommendationForm: React.FC<RecommendationFormProps> = ({
  onSubmit,
  isLoading,
}) => {
  const [personFile, setPersonFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [occasion, setOccasion] = useState<string>('Casual');
  const [preferredColor, setPreferredColor] = useState<string>('Black');
  const [stylePreference, setStylePreference] = useState<string>('Streetwear');

  const handleImageSelected = (file: File | null, url: string | null) => {
    setPersonFile(file);
    setPreviewUrl(url);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (!personFile) {
      toast.error('Please upload your photo to get outfit recommendations.');
      return;
    }

    onSubmit({
      personFile,
      previewUrl,
      occasion,
      preferredColor,
      stylePreference,
    });
  };

  return (
    <Card className="p-6 sm:p-8 rounded-2xl border border-border/60 shadow-lg bg-card/80 backdrop-blur-sm">
      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Step 1: Upload Person Image */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label className="text-base font-semibold flex items-center gap-2">
              <span className="w-6 h-6 rounded-full bg-primary/10 text-primary flex items-center justify-center text-xs font-bold">
                1
              </span>
              Upload Person Photo
            </Label>
            <span className="text-xs text-muted-foreground">Required (PNG, JPG, JPEG)</span>
          </div>

          <PersonImageUploader
            previewUrl={previewUrl}
            onImageSelected={handleImageSelected}
            disabled={isLoading}
          />
        </div>

        {/* Step 2: Select Preferences (3 Dropdowns) */}
        <div className="space-y-3 pt-2">
          <Label className="text-base font-semibold flex items-center gap-2">
            <span className="w-6 h-6 rounded-full bg-primary/10 text-primary flex items-center justify-center text-xs font-bold">
              2
            </span>
            Customize Preferences
          </Label>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Dropdown 1: Occasion */}
            <div className="space-y-2">
              <label className="text-xs font-medium text-muted-foreground flex items-center gap-1.5">
                <Calendar className="w-3.5 h-3.5 text-primary" />
                Occasion
              </label>
              <Select
                value={occasion}
                onValueChange={setOccasion}
                disabled={isLoading}
              >
                <SelectTrigger className="w-full h-11 rounded-xl bg-background border-input">
                  <SelectValue placeholder="Select Occasion" />
                </SelectTrigger>
                <SelectContent className="rounded-xl">
                  {OCCASIONS.map((item) => (
                    <SelectItem key={item} value={item} className="text-sm">
                      {item}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Dropdown 2: Preferred Color */}
            <div className="space-y-2">
              <label className="text-xs font-medium text-muted-foreground flex items-center gap-1.5">
                <Palette className="w-3.5 h-3.5 text-blue-500" />
                Preferred Color
              </label>
              <Select
                value={preferredColor}
                onValueChange={setPreferredColor}
                disabled={isLoading}
              >
                <SelectTrigger className="w-full h-11 rounded-xl bg-background border-input">
                  <SelectValue placeholder="Select Color" />
                </SelectTrigger>
                <SelectContent className="rounded-xl">
                  {COLORS.map((item) => (
                    <SelectItem key={item} value={item} className="text-sm">
                      {item}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Dropdown 3: Style Preference */}
            <div className="space-y-2">
              <label className="text-xs font-medium text-muted-foreground flex items-center gap-1.5">
                <Shirt className="w-3.5 h-3.5 text-purple-500" />
                Style Preference
              </label>
              <Select
                value={stylePreference}
                onValueChange={setStylePreference}
                disabled={isLoading}
              >
                <SelectTrigger className="w-full h-11 rounded-xl bg-background border-input">
                  <SelectValue placeholder="Select Style" />
                </SelectTrigger>
                <SelectContent className="rounded-xl">
                  {STYLES.map((item) => (
                    <SelectItem key={item} value={item} className="text-sm">
                      {item}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
        </div>

        {/* Submit Button */}
        <div className="pt-4">
          <Button
            type="submit"
            disabled={isLoading || !personFile}
            className="w-full h-12 text-base font-semibold rounded-xl bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white shadow-lg hover:shadow-xl transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? (
              <>
                <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                Finding Recommendations...
              </>
            ) : (
              <>
                <Sparkles className="w-5 h-5 mr-2" />
                Get Recommendations
              </>
            )}
          </Button>
        </div>
      </form>
    </Card>
  );
};
