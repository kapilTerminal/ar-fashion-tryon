'use client';

import React, { useState } from 'react';
import { Sparkles, Calendar, Palette, Shirt, Loader2, User, SlidersHorizontal, Sun, Layers } from 'lucide-react';
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
import {
  GENDER_OPTIONS,
  OCCASION_OPTIONS,
  COLOR_OPTIONS,
  STYLE_OPTIONS,
  STYLE_PREFERENCE_OPTIONS,
  BODY_TYPE_OPTIONS,
  SKIN_TONE_OPTIONS,
  DEFAULT_PREFERENCES,
} from '@/lib/constants/recommendationOptions';
import type { UserPreferences } from '@/lib/types/recommendation';
import { toast } from 'sonner';

export interface FormSubmitData {
  personFile: File | null;
  previewUrl: string | null;
  preferences: UserPreferences;
}

interface RecommendationFormProps {
  onSubmit: (data: FormSubmitData) => void;
  isLoading: boolean;
}

export const RecommendationForm: React.FC<RecommendationFormProps> = ({
  onSubmit,
  isLoading,
}) => {
  const [personFile, setPersonFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  
  const [gender, setGender] = useState<string>(DEFAULT_PREFERENCES.gender);
  const [occasion, setOccasion] = useState<string>(DEFAULT_PREFERENCES.occasion);
  const [preferredColor, setPreferredColor] = useState<string>(DEFAULT_PREFERENCES.preferredColor);
  const [preferredStyle, setPreferredStyle] = useState<string>(DEFAULT_PREFERENCES.preferredStyle);
  const [stylePreference, setStylePreference] = useState<string>(DEFAULT_PREFERENCES.stylePreference);
  const [bodyType, setBodyType] = useState<string>(DEFAULT_PREFERENCES.bodyType);
  const [skinTone, setSkinTone] = useState<string>(DEFAULT_PREFERENCES.skinTone);

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
      preferences: {
        gender,
        occasion,
        preferredColor,
        preferredStyle,
        stylePreference,
        bodyType,
        skinTone,
      },
    });
  };

  return (
    <Card className="p-6 sm:p-8 rounded-3xl border border-border/60 shadow-xl bg-card/90 backdrop-blur-md">
      <form onSubmit={handleSubmit} className="space-y-8">
        {/* Step 1: Upload Person Image */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <Label className="text-base font-bold flex items-center gap-2">
              <span className="w-7 h-7 rounded-full bg-primary/10 text-primary flex items-center justify-center text-xs font-extrabold">
                1
              </span>
              Upload Person Photo
            </Label>
            <span className="text-xs text-muted-foreground">Required</span>
          </div>

          <PersonImageUploader
            previewUrl={previewUrl}
            onImageSelected={handleImageSelected}
            disabled={isLoading}
          />
        </div>

        {/* Step 2: Select Preferences */}
        <div className="space-y-4 pt-2">
          <Label className="text-base font-bold flex items-center gap-2">
            <span className="w-7 h-7 rounded-full bg-primary/10 text-primary flex items-center justify-center text-xs font-extrabold">
              2
            </span>
            Customize Style Preferences
          </Label>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {/* Field 1: Gender */}
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-muted-foreground flex items-center gap-1.5">
                <User className="w-3.5 h-3.5 text-primary" />
                Gender Target
              </label>
              <Select value={gender} onValueChange={setGender} disabled={isLoading}>
                <SelectTrigger className="w-full h-11 rounded-xl bg-background border-input">
                  <SelectValue placeholder="Select Gender" />
                </SelectTrigger>
                <SelectContent className="rounded-xl">
                  {GENDER_OPTIONS.map((opt) => (
                    <SelectItem key={opt.value} value={opt.value} className="text-sm">
                      {opt.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Field 2: Occasion */}
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-muted-foreground flex items-center gap-1.5">
                <Calendar className="w-3.5 h-3.5 text-blue-500" />
                Occasion
              </label>
              <Select value={occasion} onValueChange={setOccasion} disabled={isLoading}>
                <SelectTrigger className="w-full h-11 rounded-xl bg-background border-input">
                  <SelectValue placeholder="Select Occasion" />
                </SelectTrigger>
                <SelectContent className="rounded-xl">
                  {OCCASION_OPTIONS.map((opt) => (
                    <SelectItem key={opt.value} value={opt.value} className="text-sm">
                      {opt.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Field 3: Preferred Color */}
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-muted-foreground flex items-center gap-1.5">
                <Palette className="w-3.5 h-3.5 text-purple-500" />
                Preferred Color
              </label>
              <Select value={preferredColor} onValueChange={setPreferredColor} disabled={isLoading}>
                <SelectTrigger className="w-full h-11 rounded-xl bg-background border-input">
                  <SelectValue placeholder="Select Color" />
                </SelectTrigger>
                <SelectContent className="rounded-xl">
                  {COLOR_OPTIONS.map((opt) => (
                    <SelectItem key={opt.value} value={opt.value} className="text-sm">
                      {opt.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Field 4: Preferred Style */}
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-muted-foreground flex items-center gap-1.5">
                <Shirt className="w-3.5 h-3.5 text-emerald-500" />
                Preferred Style
              </label>
              <Select value={preferredStyle} onValueChange={setPreferredStyle} disabled={isLoading}>
                <SelectTrigger className="w-full h-11 rounded-xl bg-background border-input">
                  <SelectValue placeholder="Select Preferred Style" />
                </SelectTrigger>
                <SelectContent className="rounded-xl">
                  {STYLE_OPTIONS.map((opt) => (
                    <SelectItem key={opt.value} value={opt.value} className="text-sm">
                      {opt.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Field 5: Style Preference */}
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-muted-foreground flex items-center gap-1.5">
                <SlidersHorizontal className="w-3.5 h-3.5 text-indigo-500" />
                Style Preference
              </label>
              <Select value={stylePreference} onValueChange={setStylePreference} disabled={isLoading}>
                <SelectTrigger className="w-full h-11 rounded-xl bg-background border-input">
                  <SelectValue placeholder="Select Aesthetic" />
                </SelectTrigger>
                <SelectContent className="rounded-xl">
                  {STYLE_PREFERENCE_OPTIONS.map((opt) => (
                    <SelectItem key={opt.value} value={opt.value} className="text-sm">
                      {opt.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Field 6: Body Type */}
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-muted-foreground flex items-center gap-1.5">
                <Layers className="w-3.5 h-3.5 text-amber-500" />
                Body Type
              </label>
              <Select value={bodyType} onValueChange={setBodyType} disabled={isLoading}>
                <SelectTrigger className="w-full h-11 rounded-xl bg-background border-input">
                  <SelectValue placeholder="Select Body Type" />
                </SelectTrigger>
                <SelectContent className="rounded-xl">
                  {BODY_TYPE_OPTIONS.map((opt) => (
                    <SelectItem key={opt.value} value={opt.value} className="text-sm">
                      {opt.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Field 7: Skin Tone */}
            <div className="space-y-1.5 sm:col-span-2 lg:col-span-1">
              <label className="text-xs font-semibold text-muted-foreground flex items-center gap-1.5">
                <Sun className="w-3.5 h-3.5 text-rose-500" />
                Skin Tone / Undertone
              </label>
              <Select value={skinTone} onValueChange={setSkinTone} disabled={isLoading}>
                <SelectTrigger className="w-full h-11 rounded-xl bg-background border-input">
                  <SelectValue placeholder="Select Skin Tone" />
                </SelectTrigger>
                <SelectContent className="rounded-xl">
                  {SKIN_TONE_OPTIONS.map((opt) => (
                    <SelectItem key={opt.value} value={opt.value} className="text-sm">
                      {opt.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
        </div>

        {/* Submit Button */}
        <div className="pt-2">
          <Button
            type="submit"
            disabled={isLoading || !personFile}
            className="w-full h-12 text-base font-bold rounded-2xl bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white shadow-lg hover:shadow-xl transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? (
              <>
                <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                Generating Outfit Recommendations...
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
