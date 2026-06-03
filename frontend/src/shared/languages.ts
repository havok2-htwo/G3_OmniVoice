// Curated synthesis languages shown in the synthesis UIs (Demo + Admin Quick
// Synthesis). The backend maps these labels to the ISO codes that
// OmniVoice.generate(language=...) resolves; "Auto" lets the model detect the
// language from the text. Keep this list in sync with SUPPORTED_LANGUAGES in
// backend/src/omnivoice_tts_server/api/router_v2.py and LANGUAGE_LABEL_TO_CODE
// in backend/src/omnivoice_tts_server/runtime_v2.py.
export const SYNTH_LANGUAGE_OPTIONS = [
  "Auto",
  "Deutsch",
  "English",
  "Français",
  "Español",
  "Italiano",
  "Nederlands",
  "Polski",
  "Português",
  "Türkçe",
  "Русский",
  "Українська",
  "中文",
  "日本語",
  "한국어",
] as const;
