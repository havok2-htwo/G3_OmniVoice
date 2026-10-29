import type { MingDeliveryFields, MingEmotion, MingLevel, MingSpeakingRate, ReferenceMode } from "./api";

export const MING_EMOTION_OPTIONS: Array<{ value: MingEmotion; label: string }> = [
  { value: "happy", label: "froehlich" },
  { value: "sad", label: "traurig" },
  { value: "angry", label: "wuetend" },
  { value: "fearful", label: "aengstlich" },
  { value: "disgusted", label: "angeekelt" },
  { value: "surprised", label: "ueberrascht" },
  { value: "neutral", label: "neutral" },
];

export const MING_RATE_OPTIONS: Array<{ value: MingSpeakingRate; label: string }> = [
  { value: "slow", label: "langsam" },
  { value: "medium", label: "normal" },
  { value: "fast", label: "schnell" },
];

export const MING_PITCH_OPTIONS: Array<{ value: MingLevel; label: string }> = [
  { value: "low", label: "tief" },
  { value: "medium", label: "mittel" },
  { value: "high", label: "hoch" },
];

export const MING_VOLUME_OPTIONS: Array<{ value: MingLevel; label: string }> = [
  { value: "low", label: "leise" },
  { value: "medium", label: "mittel" },
  { value: "high", label: "laut" },
];

export const REFERENCE_MODE_OPTIONS: Array<{ value: ReferenceMode; label: string }> = [
  { value: "auto", label: "auto" },
  { value: "full", label: "voll" },
  { value: "speaker", label: "nur Sprecher" },
];

/** Ming's inline paralinguistic tags (case-sensitive upstream; the server maps other spellings). The first ones are
 * shown directly, the rest behind "mehr". */
export const MING_TAGS: Array<{ tag: string; label: string; common?: boolean }> = [
  { tag: "[Laughter]", label: "Lachen", common: true },
  { tag: "[Sigh]", label: "Seufzen", common: true },
  { tag: "[Breathing]", label: "Atmen", common: true },
  { tag: "[Uhm]", label: "Aehm", common: true },
  { tag: "[Cough]", label: "Husten", common: true },
  { tag: "[Surprise-ah]", label: "Ah! (ueberrascht)" },
  { tag: "[Surprise-oh]", label: "Oh! (ueberrascht)" },
  { tag: "[Dissatisfaction-hnn]", label: "Hmpf (unzufrieden)" },
  { tag: "[Confirmation-en]", label: "Mhm (zustimmend)" },
  { tag: "[Question-ah]", label: "Ah? (fragend)" },
  { tag: "[Question-oh]", label: "Oh? (fragend)" },
  { tag: "[Question-en]", label: "Hm? (fragend)" },
  { tag: "[Question-ei]", label: "Ei? (fragend)" },
  { tag: "[Question-huh]", label: "Hae? (fragend)" },
];

export type MingDeliveryDraft = {
  emotion: "" | MingEmotion;
  speaking_rate: "" | MingSpeakingRate;
  pitch: "" | MingLevel;
  volume: "" | MingLevel;
  reference_mode: "" | ReferenceMode;
};

export const EMPTY_MING_DELIVERY: MingDeliveryDraft = { emotion: "", speaking_rate: "", pitch: "", volume: "", reference_mode: "" };

/** A structured caption (emotion/rate/pitch/volume) cannot be combined with a free-form instruction on Ming. */
export function hasStructuredMingControl(draft: MingDeliveryDraft): boolean {
  return Boolean(draft.emotion || draft.speaking_rate || draft.pitch || draft.volume);
}

/** What the server's "auto" reference mode resolves to for this draft (ming_runtime.resolve_reference_mode). */
export function resolvedReferenceMode(draft: MingDeliveryDraft): "full" | "speaker" {
  if (draft.reference_mode === "full" || draft.reference_mode === "speaker") return draft.reference_mode;
  return hasStructuredMingControl(draft) ? "speaker" : "full";
}

/** Request fields for the set controls only; unset controls stay off the wire. */
export function mingDeliveryFields(draft: MingDeliveryDraft): MingDeliveryFields {
  const fields: MingDeliveryFields = {};
  if (draft.emotion) fields.emotion = draft.emotion;
  if (draft.speaking_rate) fields.speaking_rate = draft.speaking_rate;
  if (draft.pitch) fields.pitch = draft.pitch;
  if (draft.volume) fields.volume = draft.volume;
  if (draft.reference_mode) fields.reference_mode = draft.reference_mode;
  return fields;
}
