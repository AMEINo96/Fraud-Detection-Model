/**
 * Simple Logistic Regression for Fraud Detection
 */

export class LogisticRegression {
  weights: number[] = [];
  bias: number = 0;
  learningRate: number = 0.1;
  epochs: number = 50;

  constructor(featureCount: number) {
    this.weights = new Array(featureCount).fill(0);
  }

  sigmoid(z: number): number {
    return 1 / (1 + Math.exp(-z));
  }

  predict(features: number[]): number {
    let z = this.bias;
    for (let i = 0; i < features.length; i++) {
      z += this.weights[i] * features[i];
    }
    return this.sigmoid(z);
  }

  train(features: number[][], labels: number[], onProgress?: (epoch: number, loss: number) => void) {
    const m = features.length;
    const n = features[0].length;

    for (let epoch = 0; epoch < this.epochs; epoch++) {
      let totalLoss = 0;
      const dw = new Array(n).fill(0);
      let db = 0;

      for (let i = 0; i < m; i++) {
        const yPred = this.predict(features[i]);
        const yTrue = labels[i];
        const error = yPred - yTrue;

        // Gradient computation
        for (let j = 0; j < n; j++) {
          dw[j] += error * features[i][j];
        }
        db += error;

        // Binary Cross Entropy Loss
        totalLoss += -(yTrue * Math.log(yPred + 1e-15) + (1 - yTrue) * Math.log(1 - yPred + 1e-15));
      }

      // Update weights and bias
      for (let j = 0; j < n; j++) {
        this.weights[j] -= (this.learningRate * dw[j]) / m;
      }
      this.bias -= (this.learningRate * db) / m;

      if (onProgress) {
        onProgress(epoch, totalLoss / m);
      }
    }
  }
}

export function evaluateModel(predictions: number[], actual: number[], threshold: number = 0.5) {
  let tp = 0;
  let tn = 0;
  let fp = 0;
  let fn = 0;

  for (let i = 0; i < predictions.length; i++) {
    const isFraudPred = predictions[i] >= threshold;
    const isFraudActual = actual[i] === 1;

    if (isFraudPred && isFraudActual) tp++;
    else if (!isFraudPred && !isFraudActual) tn++;
    else if (isFraudPred && !isFraudActual) fp++;
    else fn++;
  }

  const precision = tp / (tp + fp) || 0;
  const recall = tp / (tp + fn) || 0;
  const f1 = (2 * precision * recall) / (precision + recall) || 0;
  const accuracy = (tp + tn) / (tp + tn + fp + fn);

  return { tp, tn, fp, fn, precision, recall, f1, accuracy };
}
