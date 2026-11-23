import type { MingDeliveryFields, MingEmotion, MingLevel, MingSpeakingRate, ReferenceMode } from "./api";

export const MING_EMOTION_OPTIONS: Array<{ value: MingEmotion; label: string }> = [
  { value: "happy", label: "happy" },
  { value: "sad", label: "sad" },
  { value: "angry", label: "angry" },
  { value: "fearful", label: "fearful" },
  { value: "disgusted", label: "disgusted" },
  { value: "surprised", label: "surprised" },
  { value: "neutral", label: "neutral" },
];

export const MING_RATE_OPTIONS: Array<{ value: MingSpeakingRate; label: string }> = [
  { value: "slow", label: "slow" },
  { value: "medium", label: "normal" },
  { value: "fast", label: "fast" },
];

export const MING_PITCH_OPTIONS: Array<{ value: MingLevel; label: string }> = [
  { value: "low", label: "low" },
  { value: "medium", label: "medium" },
  { value: "high", label: "high" },
];

export const MING_VOLUME_OPTIONS: Array<{ value: MingLevel; label: string }> = [
  { value: "low", label: "quiet" },
  { value: "medium", label: "medium" },
  { value: "high", label: "loud" },
];

export const REFERENCE_MODE_OPTIONS: Array<{ value: ReferenceMode; label: string }> = [
  { value: "auto", label: "auto" },
  { value: "full", label: "full" },
  { value: "speaker", label: "speaker only" },
];

/** Ming's inline paralinguistic tags (case-sensitive upstream; the server maps other spellings). The first ones are
 * shown directly, the rest behind "more". */
export const MING_TAGS: Array<{ tag: string; label: string; common?: boolean }> = [
  { tag: "[Laughter]", label: "Laughter", common: true },
  { tag: "[Sigh]", label: "Sigh", common: true },
  { tag: "[Breathing]", label: "Breathing", common: true },
  { tag: "[Uhm]", label: "Uhm", common: true },
  { tag: "[Cough]", label: "Cough", common: true },
  { tag: "[Surprise-ah]", label: "Ah! (surprised)" },
  { tag: "[Surprise-oh]", label: "Oh! (surprised)" },
  { tag: "[Dissatisfaction-hnn]", label: "Hmpf (dissatisfied)" },
  { tag: "[Confirmation-en]", label: "Mhm (agreeing)" },
  { tag: "[Question-ah]", label: "Ah? (asking)" },
  { tag: "[Question-oh]", label: "Oh? (asking)" },
  { tag: "[Question-en]", label: "Hm? (asking)" },
  { tag: "[Question-ei]", label: "Ei? (asking)" },
  { tag: "[Question-huh]", label: "Huh? (asking)" },
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
