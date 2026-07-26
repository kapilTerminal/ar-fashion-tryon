/**
 * Form selection constants for Fashion Recommendation Engine
 * Stage 5.1
 */

export interface SelectOption {
  label: string;
  value: string;
}

export const GENDER_OPTIONS: SelectOption[] = [
  { label: 'Unisex', value: 'unisex' },
  { label: 'Men', value: 'men' },
  { label: 'Women', value: 'women' },
];

export const OCCASION_OPTIONS: SelectOption[] = [
  { label: 'Casual', value: 'casual' },
  { label: 'Formal', value: 'formal' },
  { label: 'Office / Business', value: 'office' },
  { label: 'Party / Evening', value: 'party' },
  { label: 'Wedding', value: 'wedding' },
  { label: 'Activewear / Gym', value: 'activewear' },
  { label: 'Date Night', value: 'date_night' },
];

export const COLOR_OPTIONS: SelectOption[] = [
  { label: 'Black', value: 'black' },
  { label: 'White', value: 'white' },
  { label: 'Blue', value: 'blue' },
  { label: 'Red', value: 'red' },
  { label: 'Green', value: 'green' },
  { label: 'Brown', value: 'brown' },
  { label: 'Pink', value: 'pink' },
  { label: 'Grey', value: 'grey' },
  { label: 'Beige', value: 'beige' },
  { label: 'Navy', value: 'navy' },
  { label: 'Yellow', value: 'yellow' },
];

export const STYLE_OPTIONS: SelectOption[] = [
  { label: 'Casual', value: 'casual' },
  { label: 'Formal', value: 'formal' },
  { label: 'Streetwear', value: 'streetwear' },
  { label: 'Minimalist', value: 'minimalist' },
  { label: 'Vintage / Retro', value: 'vintage' },
  { label: 'Sporty', value: 'sporty' },
  { label: 'Elegant', value: 'elegant' },
];

export const STYLE_PREFERENCE_OPTIONS: SelectOption[] = [
  { label: 'Modern', value: 'modern' },
  { label: 'Classic', value: 'classic' },
  { label: 'Trendy', value: 'trendy' },
  { label: 'Chic', value: 'chic' },
  { label: 'Edgy', value: 'edgy' },
  { label: 'Bohemian', value: 'bohemian' },
];

export const BODY_TYPE_OPTIONS: SelectOption[] = [
  { label: 'Rectangle', value: 'rectangle' },
  { label: 'Hourglass', value: 'hourglass' },
  { label: 'Inverted Triangle', value: 'inverted_triangle' },
  { label: 'Pear', value: 'pear' },
  { label: 'Apple', value: 'apple' },
  { label: 'Athletic', value: 'athletic' },
];

export const SKIN_TONE_OPTIONS: SelectOption[] = [
  { label: 'Fair', value: 'fair' },
  { label: 'Light', value: 'light' },
  { label: 'Medium', value: 'medium' },
  { label: 'Tan', value: 'tan' },
  { label: 'Dark', value: 'dark' },
  { label: 'Warm Undertone', value: 'warm' },
  { label: 'Cool Undertone', value: 'cool' },
];

export const DEFAULT_PREFERENCES = {
  gender: 'unisex',
  occasion: 'casual',
  preferredColor: 'black',
  preferredStyle: 'casual',
  stylePreference: 'modern',
  bodyType: 'rectangle',
  skinTone: 'medium',
};
