'use client';

import React, { useRef, useState } from 'react';
import Image from 'next/image';
import { Upload, X, User, Image as ImageIcon, CheckCircle2, AlertCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';
import { toast } from 'sonner';

interface PersonImageUploaderProps {
  previewUrl: string | null;
  onImageSelected: (file: File | null, previewUrl: string | null) => void;
  disabled?: boolean;
}

const ALLOWED_TYPES = ['image/png', 'image/jpeg', 'image/jpg'];

export const PersonImageUploader: React.FC<PersonImageUploaderProps> = ({
  previewUrl,
  onImageSelected,
  disabled = false,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const validateAndProcessFile = (file: File) => {
    setErrorMsg(null);

    if (!ALLOWED_TYPES.includes(file.type.toLowerCase())) {
      const msg = 'Unsupported file format. Please upload PNG, JPG, or JPEG.';
      setErrorMsg(msg);
      toast.error(msg);
      return;
    }

    // 10MB limit check
    if (file.size > 10 * 1024 * 1024) {
      const msg = 'File size exceeds 10MB limit.';
      setErrorMsg(msg);
      toast.error(msg);
      return;
    }

    const url = URL.createObjectURL(file);
    onImageSelected(file, url);
    toast.success('Person photo uploaded successfully!');
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      validateAndProcessFile(file);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    if (!disabled) setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (disabled) return;

    const file = e.dataTransfer.files?.[0];
    if (file) {
      validateAndProcessFile(file);
    }
  };

  const handleRemove = () => {
    if (previewUrl && previewUrl.startsWith('blob:')) {
      URL.revokeObjectURL(previewUrl);
    }
    setErrorMsg(null);
    onImageSelected(null, null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <Card className="p-4 sm:p-6 border-dashed border-2 relative transition-all duration-200">
      <input
        ref={fileInputRef}
        type="file"
        accept="image/png, image/jpeg, image/jpg"
        onChange={handleFileChange}
        className="hidden"
        disabled={disabled}
      />

      {previewUrl ? (
        <div className="relative flex flex-col items-center space-y-4">
          <div className="relative w-full max-w-xs h-72 sm:h-80 rounded-xl overflow-hidden shadow-lg border border-border/50 bg-black/5 group">
            <Image
              src={previewUrl}
              alt="Person Preview"
              fill
              className="object-cover transition-transform duration-300 group-hover:scale-105"
              unoptimized
            />
            <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity flex items-end justify-between p-3">
              <Badge variant="secondary" className="bg-background/80 backdrop-blur-md text-xs">
                <CheckCircle2 className="w-3 h-3 mr-1 text-emerald-500" />
                Selected
              </Badge>
              <Button
                type="button"
                variant="destructive"
                size="icon"
                onClick={handleRemove}
                disabled={disabled}
                className="h-8 w-8 rounded-full shadow-md"
              >
                <X className="w-4 h-4" />
              </Button>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => fileInputRef.current?.click()}
              disabled={disabled}
              className="text-xs"
            >
              <ImageIcon className="w-3.5 h-3.5 mr-1.5" />
              Change Photo
            </Button>
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={handleRemove}
              disabled={disabled}
              className="text-xs text-muted-foreground hover:text-destructive"
            >
              Remove
            </Button>
          </div>
        </div>
      ) : (
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => !disabled && fileInputRef.current?.click()}
          className={cn(
            'flex flex-col items-center justify-center p-8 sm:p-10 text-center cursor-pointer rounded-xl transition-all duration-200',
            isDragging
              ? 'bg-primary/10 border-primary scale-[0.99]'
              : 'hover:bg-muted/50 border-transparent',
            disabled && 'opacity-50 cursor-not-allowed'
          )}
        >
          <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center mb-4 text-primary transition-transform duration-200 group-hover:scale-110">
            <User className="w-8 h-8" />
          </div>

          <h3 className="font-semibold text-base sm:text-lg mb-1">
            Upload Your Full Body Photo
          </h3>

          <p className="text-xs sm:text-sm text-muted-foreground max-w-sm mb-4">
            Drag & drop your photo here, or click to browse. Standard pose facing forward produces best recommendations.
          </p>

          <div className="flex items-center gap-2 mb-2">
            <Badge variant="outline" className="text-[11px] font-medium">PNG</Badge>
            <Badge variant="outline" className="text-[11px] font-medium">JPG</Badge>
            <Badge variant="outline" className="text-[11px] font-medium">JPEG</Badge>
          </div>

          <Button type="button" variant="secondary" size="sm" disabled={disabled} className="mt-2 text-xs">
            <Upload className="w-3.5 h-3.5 mr-1.5" />
            Select Photo
          </Button>

          {errorMsg && (
            <div className="mt-4 text-xs text-destructive flex items-center gap-1.5 font-medium">
              <AlertCircle className="w-4 h-4" />
              {errorMsg}
            </div>
          )}
        </div>
      )}
    </Card>
  );
};
