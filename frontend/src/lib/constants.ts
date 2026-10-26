export const API_BASE_URL = (import.meta as any).env?.VITE_API_BASE_URL || "http://localhost:8001/api/v1";

export const THEME_COLORS = [
  { id: "violet", name: "Aurora Violet", gradient: "from-[#7C6CFF]/80 via-[#5CC8FF]/60 to-[#121A33]", bg: "#7C6CFF" },
  { id: "amber", name: "Lamp Amber", gradient: "from-[#FFB454]/80 via-[#FF6584]/60 to-[#121A33]", bg: "#FFB454" },
  { id: "mint", name: "Emerald Mint", gradient: "from-[#3DDC97]/80 via-[#5CC8FF]/60 to-[#121A33]", bg: "#3DDC97" },
  { id: "sky", name: "Celestial Sky", gradient: "from-[#5CC8FF]/80 via-[#7C6CFF]/60 to-[#121A33]", bg: "#5CC8FF" },
  { id: "rose", name: "Twilight Rose", gradient: "from-[#FF6584]/80 via-[#7C6CFF]/60 to-[#121A33]", bg: "#FF6584" },
  { id: "coral", name: "Scholar Coral", gradient: "from-[#FF6B6B]/80 via-[#FFB454]/60 to-[#121A33]", bg: "#FF6B6B" },
];

export const ANSWER_STYLES = [
  { id: "concise", label: "Concise", description: "Direct 2-3 sentence answer with direct citations." },
  { id: "detailed", label: "Detailed", description: "Comprehensive breakdown with step-by-step logic." },
  { id: "exam-5-mark", label: "Exam (5-Mark)", description: "Structured for exam marking scheme (Definitions, Key Points, Conclusion)." },
  { id: "explain-like-beginner", label: "Beginner Friendly", description: "Intuitive analogies with jargon simplified." },
];
