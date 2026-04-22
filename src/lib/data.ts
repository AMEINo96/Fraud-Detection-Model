/**
 * Synthetic Data Generator for Credit Card Fraud Detection
 */

export interface Transaction {
  id: string;
  time: number;
  amount: number;
  // Anonymized variables V1-V28 (simulated as PCA components)
  v: number[];
  isFraud: boolean;
}

export function generateSyntheticData(count: number, fraudRatio: number = 0.0017): Transaction[] {
  const transactions: Transaction[] = [];
  
  for (let i = 0; i < count; i++) {
    const isFraud = Math.random() < fraudRatio;
    const time = i * 2; // Simulated seconds from start
    
    // Amount usually has a right-skewed distribution
    // Normal: $5-$100, Fraud: can be small or huge
    let amount = 0;
    if (isFraud) {
      amount = Math.random() < 0.3 ? Math.random() * 20 : Math.random() * 2000;
    } else {
      amount = Math.exp(Math.random() * 4) + 5;
    }
    
    // Generate V1-V28
    // Normal transactions cluster around certain values
    // Fraud transactions are outliers
    const v: number[] = Array.from({ length: 28 }, (_, j) => {
      if (isFraud) {
        // Higher variance and different mean for fraud
        return (Math.random() - 0.5) * 10 + (Math.random() > 0.5 ? 5 : -5);
      }
      return (Math.random() - 0.5) * 4;
    });

    transactions.push({
      id: `TX-${i.toString().padStart(6, '0')}`,
      time,
      amount,
      v,
      isFraud
    });
  }
  
  return transactions;
}

export function preprocessData(data: Transaction[]) {
  // Simple standardization of Amount and Time
  const amounts = data.map(t => t.amount);
  const meanAmount = amounts.reduce((a, b) => a + b, 0) / amounts.length;
  const stdAmount = Math.sqrt(amounts.reduce((a, b) => a + Math.pow(b - meanAmount, 2), 0) / amounts.length);

  const times = data.map(t => t.time);
  const meanTime = times.reduce((a, b) => a + b, 0) / times.length;
  const stdTime = Math.sqrt(times.reduce((a, b) => a + Math.pow(b - meanTime, 2), 0) / times.length);

  return data.map(t => ({
    ...t,
    scaledAmount: (t.amount - meanAmount) / (stdAmount || 1),
    scaledTime: (t.time - meanTime) / (stdTime || 1),
    features: [
      (t.amount - meanAmount) / (stdAmount || 1),
      (t.time - meanTime) / (stdTime || 1),
      ...t.v
    ]
  }));
}
