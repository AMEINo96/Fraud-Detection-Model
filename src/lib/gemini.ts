import { GoogleGenAI } from "@google/genai";

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

export async function analyzeFraudResults(metrics: any, samples: any[]) {
  const prompt = `
    As a Fraud Detection Analyst, analyze the following model performance metrics and specific fraud samples.
    
    Metrics:
    - Accuracy: ${metrics.accuracy.toFixed(4)}
    - Precision: ${metrics.precision.toFixed(4)}
    - Recall: ${metrics.recall.toFixed(4)}
    - F1 Score: ${metrics.f1.toFixed(4)}
    - Confusion Matrix: TP: ${metrics.tp}, TN: ${metrics.tn}, FP: ${metrics.fp}, FN: ${metrics.fn}
    
    Samples identified as fraudulent (first 3):
    ${JSON.stringify(samples.slice(0, 3), null, 2)}
    
    Provide a concise technical report on:
    1. The effectiveness of the current model.
    2. The danger of identified False Negatives or False Positives in this specific context.
    3. Suggestions for feature engineering or model improvement.
    
    Keep the tone professional and data-driven.
  `;

  try {
    const result = await ai.models.generateContent({
      model: "gemini-3-flash-preview",
      contents: prompt,
    });
    return result.text;
  } catch (error) {
    console.error("Gemini analysis failed:", error);
    return "Analysis failed due to API error.";
  }
}
