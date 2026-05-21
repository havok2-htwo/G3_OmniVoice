export const DEFAULT_VOICE_DESIGN_INSTRUCT = "female, young adult, moderate pitch";

export const VOICE_DESIGN_GROUPS = [
  {
    id: "gender",
    label: "Gender",
    options: [
      { value: "female", label: "Female / 女" },
      { value: "male", label: "Male / 男" },
    ],
  },
  {
    id: "age",
    label: "Age",
    options: [
      { value: "child", label: "Child / 儿童" },
      { value: "teenager", label: "Teenager / 少年" },
      { value: "young adult", label: "Young Adult / 青年" },
      { value: "middle-aged", label: "Middle-aged / 中年" },
      { value: "elderly", label: "Elderly / 老年" },
    ],
  },
  {
    id: "pitch",
    label: "Pitch",
    options: [
      { value: "very low pitch", label: "Very Low Pitch / 极低音调" },
      { value: "low pitch", label: "Low Pitch / 低音调" },
      { value: "moderate pitch", label: "Moderate Pitch / 中音调" },
      { value: "high pitch", label: "High Pitch / 高音调" },
      { value: "very high pitch", label: "Very High Pitch / 极高音调" },
    ],
  },
  {
    id: "style",
    label: "Style",
    options: [{ value: "whisper", label: "Whisper / 耳语" }],
  },
  {
    id: "accent",
    label: "English Accent",
    options: [
      { value: "american accent", label: "American Accent" },
      { value: "australian accent", label: "Australian Accent" },
      { value: "british accent", label: "British Accent" },
      { value: "canadian accent", label: "Canadian Accent" },
      { value: "chinese accent", label: "Chinese Accent" },
      { value: "indian accent", label: "Indian Accent" },
      { value: "japanese accent", label: "Japanese Accent" },
      { value: "korean accent", label: "Korean Accent" },
      { value: "portuguese accent", label: "Portuguese Accent" },
      { value: "russian accent", label: "Russian Accent" },
    ],
  },
  {
    id: "dialect",
    label: "Chinese Dialect",
    options: [
      { value: "河南话", label: "Henan Dialect / 河南话" },
      { value: "陕西话", label: "Shaanxi Dialect / 陕西话" },
      { value: "四川话", label: "Sichuan Dialect / 四川话" },
      { value: "贵州话", label: "Guizhou Dialect / 贵州话" },
      { value: "云南话", label: "Yunnan Dialect / 云南话" },
      { value: "桂林话", label: "Guilin Dialect / 桂林话" },
      { value: "济南话", label: "Jinan Dialect / 济南话" },
      { value: "石家庄话", label: "Shijiazhuang Dialect / 石家庄话" },
      { value: "甘肃话", label: "Gansu Dialect / 甘肃话" },
      { value: "宁夏话", label: "Ningxia Dialect / 宁夏话" },
      { value: "青岛话", label: "Qingdao Dialect / 青岛话" },
      { value: "东北话", label: "Northeast Dialect / 东北话" },
    ],
  },
] as const;

type VoiceDesignGroupId = (typeof VOICE_DESIGN_GROUPS)[number]["id"];

const OPTION_TO_GROUP = new Map<string, VoiceDesignGroupId>(
  VOICE_DESIGN_GROUPS.flatMap((group) => group.options.map((option) => [option.value, group.id] as const)),
);

function parseKnownSelections(instruct: string): Partial<Record<VoiceDesignGroupId, string>> {
  const selections: Partial<Record<VoiceDesignGroupId, string>> = {};
  instruct
    .split(/[,\uFF0C]/)
    .map((item) => item.trim().toLowerCase())
    .filter(Boolean)
    .forEach((item) => {
      const groupId = OPTION_TO_GROUP.get(item);
      if (groupId) selections[groupId] = item;
    });
  return selections;
}

export function voiceDesignValueForGroup(instruct: string, groupId: VoiceDesignGroupId): string {
  return parseKnownSelections(instruct)[groupId] || "";
}

export function setVoiceDesignValue(instruct: string, groupId: VoiceDesignGroupId, value: string): string {
  const selections = parseKnownSelections(instruct);
  if (value) {
    selections[groupId] = value;
    if (groupId === "accent") delete selections.dialect;
    if (groupId === "dialect") delete selections.accent;
  } else {
    delete selections[groupId];
  }
  const next = VOICE_DESIGN_GROUPS.map((group) => selections[group.id]).filter(Boolean);
  return next.length ? next.join(", ") : DEFAULT_VOICE_DESIGN_INSTRUCT;
}

export function voiceDesignInstructOrDefault(instruct: string): string {
  const selections = parseKnownSelections(instruct);
  const next = VOICE_DESIGN_GROUPS.map((group) => selections[group.id]).filter(Boolean);
  return next.length ? next.join(", ") : DEFAULT_VOICE_DESIGN_INSTRUCT;
}
