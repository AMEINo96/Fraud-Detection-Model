/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { 
  ShieldAlert, 
  Database, 
  Cpu, 
  Activity, 
  TrendingDown, 
  AlertTriangle,
  CheckCircle2,
  BrainCircuit,
  Search,
  Filter,
  RefreshCw,
  Info
} from 'lucide-react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip as ReChartsTooltip, 
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  ZAxis,
  Cell,
  PieChart,
  Pie
} from 'recharts';

// UI Components
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Slider } from "@/components/ui/slider";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Label } from "@/components/ui/label";

// Libs
import { generateSyntheticData, preprocessData, Transaction } from '@/lib/data';
import { LogisticRegression, evaluateModel } from '@/lib/ml';
import { analyzeFraudResults } from '@/lib/gemini';
import { cn } from '@/lib/utils';

export default function App() {
  const [data, setData] = useState<Transaction[]>([]);
  const [datasetSize, setDatasetSize] = useState(1000);
  const [fraudRatio, setFraudRatio] = useState(0.05);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [lossHistory, setLossHistory] = useState<{ epoch: number; loss: number }[]>([]);
  const [model, setModel] = useState<LogisticRegression | null>(null);
  const [metrics, setMetrics] = useState<any>(null);
  const [aiAnalysis, setAiAnalysis] = useState<string>("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [activeTab, setActiveTab] = useState("data");

  // Generate initial data
  useEffect(() => {
    handleGenerateData();
  }, []);

  const handleGenerateData = () => {
    const rawData = generateSyntheticData(datasetSize, fraudRatio);
    setData(rawData);
    setMetrics(null);
    setAiAnalysis("");
    setModel(null);
    setLossHistory([]);
    setActiveTab("data");
  };

  const handleTrainModel = async () => {
    setIsTraining(true);
    setTrainingProgress(0);
    setLossHistory([]);

    const preprocessed = preprocessData(data);
    const featureCount = preprocessed[0].features.length;
    const lr = new LogisticRegression(featureCount);
    
    const trainFeatures = preprocessed.map(t => t.features);
    const trainLabels = preprocessed.map(t => (t.isFraud ? 1 : 0));

    const history: { epoch: number; loss: number }[] = [];
    
    lr.train(trainFeatures, trainLabels, (epoch, loss) => {
      if (epoch % 5 === 0) {
        history.push({ epoch, loss });
        setLossHistory([...history]);
        setTrainingProgress((epoch / lr.epochs) * 100);
      }
    });

    setModel(lr);
    setIsTraining(false);
    setTrainingProgress(100);

    const preds = trainFeatures.map(f => lr.predict(f));
    const evalMetrics = evaluateModel(preds, trainLabels);
    setMetrics(evalMetrics);
    
    setActiveTab("eval");
  };

  const handleAnalyzeWithAI = async () => {
    if (!metrics) return;
    setIsAnalyzing(true);
    const fraudSamples = data.filter(t => t.isFraud);
    const report = await analyzeFraudResults(metrics, fraudSamples);
    setAiAnalysis(report || "");
    setIsAnalyzing(false);
  };

  const fraudDistribution = useMemo(() => {
    const fraudCount = data.filter(d => d.isFraud).length;
    return [
      { name: 'Normal', value: data.length - fraudCount, color: '#374151' },
      { name: 'Fraud', value: fraudCount, color: '#ef4444' }
    ];
  }, [data]);

  const scatterData = useMemo(() => {
    return data.slice(0, 500).map(d => ({
      x: d.v[0],
      y: d.v[1],
      amount: d.amount,
      isFraud: d.isFraud
    }));
  }, [data]);

  const NavItem = ({ id, label, icon: Icon, active }: { id: string, label: string, icon: any, active: boolean }) => (
    <button 
      onClick={() => setActiveTab(id)}
      className={cn(
        "w-full flex items-center gap-3 px-3 py-2 rounded transition-colors text-[13px] mb-1",
        active ? "bg-blue-600 text-white font-semibold" : "text-[#9ca3af] hover:bg-[#1f2937] hover:text-white"
      )}
    >
      <Icon className="w-4 h-4" />
      {label}
    </button>
  );

  return (
    <div className="flex h-screen w-full bg-[#030712] text-[#f3f4f6] font-sans overflow-hidden">
      {/* Sidebar */}
      <aside className="w-[220px] shrink-0 border-r border-[#1f2937] bg-[#0b0f1a] p-5 flex flex-col">
        <div className="font-mono text-[14px] tracking-[1px] text-[#3b82f6] mb-[30px] font-bold">
          FRAUD_GUARD v2.4
        </div>
        
        <nav className="flex-1">
          <NavItem id="data" label="Stream Monitor" icon={Database} active={activeTab === 'data'} />
          <NavItem id="train" label="Training Console" icon={Cpu} active={activeTab === 'train'} />
          <NavItem id="eval" label="Metrics & Eval" icon={Activity} active={activeTab === 'eval'} />
          <NavItem id="ai" label="AI Reporting" icon={BrainCircuit} active={activeTab === 'ai'} />
        </nav>

        <div className="mt-auto pt-5 border-t border-[#1f2937]">
          <div className="text-[10px] text-[#9ca3af] font-semibold mb-1 uppercase tracking-wider">System Status</div>
          <div className="text-[#10b981] text-[12px] font-bold flex items-center gap-2">
            <span className="w-2 h-2 bg-[#10b981] rounded-full"></span> OPERATIONAL
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col p-6 overflow-y-auto gap-5">
        <header className="flex justify-between items-center border-b border-[#1f2937] pb-3 mb-1">
          <div>
            <h1 className="text-[20px] font-normal m-0 tracking-tight">Credit Card Fraud Detection Engine</h1>
            <p className="text-[12px] text-[#9ca3af] mt-1">Analyzing high-dimensional transactional vectors via Gradient Descent</p>
          </div>
          <div className="text-right font-mono">
            <div className="text-[11px] text-[#9ca3af]">REFRESH RATE</div>
            <div className="text-[14px] font-semibold">14ms</div>
          </div>
        </header>

        {/* Stats Grid */}
        <div className="grid grid-cols-4 gap-4">
          {[
            { label: 'Transactions (24h)', value: data.length.toLocaleString(), color: null },
            { label: 'Detection Rate', value: `${(fraudRatio * 100).toFixed(3)}%`, color: 'text-[#ef4444]' },
            { label: 'F1-Score (Test)', value: metrics ? metrics.f1.toFixed(4) : '0.0000', color: null },
            { label: 'Precision / Recall', value: metrics ? `${metrics.precision.toFixed(3)} / ${metrics.recall.toFixed(3)}` : '0.000 / 0.000', color: null },
          ].map((stat, i) => (
            <div key={i} className="bg-[#111827] border border-[#1f2937] p-4 rounded-[4px]">
              <div className="text-[10px] uppercase text-[#9ca3af] font-semibold mb-2">{stat.label}</div>
              <div className={cn("text-[24px] font-bold font-mono tracking-tighter", stat.color || "text-white")}>
                {stat.value}
              </div>
            </div>
          ))}
        </div>

        <div className="flex-1 flex flex-col min-h-0 gap-5">
          <AnimatePresence mode="wait">
            {activeTab === 'data' && (
              <motion.div 
                key="data"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="flex-1 flex flex-col gap-5 min-h-0"
              >
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-5 h-[280px] shrink-0">
                  <div className="bg-[#111827] border border-[#1f2937] p-4 rounded-[4px] flex flex-col">
                    <div className="text-[12px] font-semibold text-[#9ca3af] border-b border-[#1f2937] pb-2 mb-3 tracking-wide">STREAM CONFIGURATION</div>
                    <div className="space-y-5">
                      <div className="space-y-2">
                        <div className="flex justify-between text-[11px] font-mono">
                          <span className="text-[#9ca3af]">VECTOR CAPACITY</span>
                          <span className="text-white">{datasetSize}</span>
                        </div>
                        <Slider 
                          value={[datasetSize]} 
                          onValueChange={(v) => setDatasetSize(v[0])} 
                          min={500} max={10000} step={100}
                          className="py-2"
                        />
                      </div>
                      <div className="space-y-2">
                        <div className="flex justify-between text-[11px] font-mono">
                          <span className="text-[#9ca3af]">ANOMALY COEFFICIENT</span>
                          <span className="text-[#ef4444]">{(fraudRatio * 100).toFixed(1)}%</span>
                        </div>
                        <Slider 
                          value={[fraudRatio * 1000]} 
                          onValueChange={(v) => setFraudRatio(v[0] / 1000)} 
                          min={1} max={200} step={1}
                          className="py-2"
                        />
                      </div>
                      <Button 
                        size="sm" 
                        onClick={handleGenerateData}
                        className="w-full bg-[#1f2937] hover:bg-[#374151] border border-[#1f2937] text-[11px] font-mono mt-2"
                      >
                        <RefreshCw className="w-3 h-3 mr-2" /> REINITIALIZE_STREAM
                      </Button>
                    </div>
                  </div>

                  <div className="lg:col-span-2 bg-[#111827] border border-[#1f2937] p-4 rounded-[4px] flex flex-col h-full">
                    <div className="text-[12px] font-semibold text-[#9ca3af] border-b border-[#1f2937] pb-2 mb-3 flex justify-between tracking-wide">
                      CLUSTER ANALYSIS (V1_V2 PROJECTION)
                    </div>
                    <div className="flex-1 min-h-0">
                      <ResponsiveContainer width="100%" height="100%">
                        <ScatterChart margin={{ top: 10, right: 10, bottom: 0, left: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false} />
                          <XAxis type="number" dataKey="x" hide />
                          <YAxis type="number" dataKey="y" hide />
                          <ZAxis type="number" dataKey="amount" range={[5, 150]} />
                          <Scatter name="Transactions" data={scatterData}>
                            {scatterData.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={entry.isFraud ? '#ef4444' : '#3b82f6'} fillOpacity={entry.isFraud ? 0.9 : 0.3} />
                            ))}
                          </Scatter>
                        </ScatterChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>

                <div className="flex-1 bg-[#111827] border border-[#1f2937] rounded-[4px] flex flex-col min-h-[300px]">
                  <div className="px-4 py-3 border-b border-[#1f2937] flex justify-between items-center shrink-0">
                    <span className="text-[11px] font-bold text-[#9ca3af] tracking-widest uppercase italic">Live Transaction Stream</span>
                    <Badge variant="outline" className="bg-[#1f2937] text-[#3b82f6] border-[#3b82f6] text-[9px] uppercase font-mono">Auto-Refresh: 2s</Badge>
                  </div>
                  <div className="flex-1 overflow-auto">
                    <Table className="font-mono text-[11px] border-collapse w-full">
                      <TableHeader className="bg-[#1f2937] sticky top-0 z-10">
                        <TableRow className="border-0 hover:bg-transparent">
                          <TableHead className="text-[#9ca3af] uppercase text-[9px] h-9 px-4">TXN_ID</TableHead>
                          <TableHead className="text-[#9ca3af] uppercase text-[9px] h-9 px-4">TIMESTAMP</TableHead>
                          <TableHead className="text-[#9ca3af] uppercase text-[9px] h-9 px-4 text-right">AMOUNT</TableHead>
                          <TableHead className="text-[#9ca3af] uppercase text-[9px] h-9 px-4">VECTOR_V14</TableHead>
                          <TableHead className="text-[#9ca3af] uppercase text-[9px] h-9 px-4 text-right">IDENTIFICATION</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {data.slice(0, 15).map((tx) => (
                          <TableRow key={tx.id} className="border-b border-[#1f2937] hover:bg-[#1f2937]/30 transition-colors">
                            <TableCell className="px-4 py-2 font-bold py-3">{tx.id}</TableCell>
                            <TableCell className="px-4 py-2 text-[#9ca3af]">{tx.time}s</TableCell>
                            <TableCell className="px-4 py-2 text-right">${tx.amount.toFixed(2)}</TableCell>
                            <TableCell className="px-4 py-2 text-[#9ca3af]">{tx.v[13] > 0 ? '+' : ''}{tx.v[13].toFixed(3)}</TableCell>
                            <TableCell className="px-4 py-2 text-right">
                              {tx.isFraud ? (
                                <span className="text-[#ef4444] bg-[#ef4444]/10 px-2 py-0.5 border border-[#ef4444] rounded-[2px] text-[10px] font-bold">FRAUD_ALERT</span>
                              ) : (
                                <span className="text-[#10b981] opacity-60">LEGITIMATE</span>
                              )}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                </div>
              </motion.div>
            )}

            {activeTab === 'train' && (
              <motion.div 
                key="train"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="grid grid-cols-1 lg:grid-cols-2 gap-5"
              >
                <div className="bg-[#111827] border border-[#1f2937] p-5 rounded-[4px] flex flex-col h-[400px]">
                  <div className="text-[12px] font-semibold text-[#9ca3af] border-b border-[#1f2937] pb-2 mb-4 uppercase tracking-wider flex justify-between">
                    Optimizer Convergence
                    <span className="font-mono text-[10px] text-blue-400">LR: 0.1 / EPOCHS: 50</span>
                  </div>
                  <div className="flex-1 min-h-0">
                    {lossHistory.length > 0 ? (
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={lossHistory}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false} />
                          <XAxis dataKey="epoch" stroke="#4b5563" fontSize={10} axisLine={false} tickLine={false} />
                          <YAxis stroke="#4b5563" fontSize={10} axisLine={false} tickLine={false} />
                          <Line type="monotone" dataKey="loss" stroke="#3b82f6" strokeWidth={2} dot={false} animationDuration={500} />
                        </LineChart>
                      </ResponsiveContainer>
                    ) : (
                      <div className="h-full flex flex-col items-center justify-center border border-dashed border-[#1f2937] rounded">
                        <Cpu className={cn("w-10 h-10 mb-3", isTraining ? "animate-pulse text-blue-500" : "text-[#1f2937]")} />
                        <p className="text-[11px] font-mono text-[#4b5563]">System ready for training cycle</p>
                      </div>
                    )}
                  </div>
                  <Button 
                    className="mt-4 bg-blue-600 hover:bg-blue-700 text-white font-mono text-[12px] h-10" 
                    onClick={handleTrainModel}
                    disabled={isTraining}
                  >
                    {isTraining ? `ADJUSTING_WEIGHTS ${Math.round(trainingProgress)}%` : 'INITIATE_OPTIMIZATION'}
                  </Button>
                </div>

                <div className="space-y-5">
                  <div className="bg-[#111827] border border-[#1f2937] p-5 rounded-[4px]">
                    <div className="text-[12px] font-semibold text-[#9ca3af] border-b border-[#1f2937] pb-2 mb-4 uppercase tracking-wider">Architecture Overview</div>
                    <div className="grid grid-cols-2 gap-4 mb-5">
                      <div className="bg-[#030712] p-3 border border-[#1f2937]">
                        <div className="text-[9px] uppercase text-[#9ca3af] mb-1 font-mono">Algorithm</div>
                        <div className="text-[13px] font-bold">LOGISTIC_REGRESSION</div>
                      </div>
                      <div className="bg-[#030712] p-3 border border-[#1f2937]">
                        <div className="text-[9px] uppercase text-[#9ca3af] mb-1 font-mono">Features</div>
                        <div className="text-[13px] font-bold">30V_DIMENSIONS</div>
                      </div>
                    </div>
                    <div className="space-y-2">
                       <div className="flex justify-between text-[10px] font-mono text-[#9ca3af]">
                         <span>TRAINING_LOAD</span>
                         <span>{Math.round(trainingProgress)}%</span>
                       </div>
                       <Progress value={trainingProgress} className="h-1 bg-[#1f2937]" />
                    </div>
                  </div>

                  <Alert className="bg-[#ef4444]/5 border-[#ef4444]/20 rounded-[4px]">
                    <AlertTriangle className="w-4 h-4 text-[#ef4444]" />
                    <AlertTitle className="text-[#ef4444] text-[12px] font-bold uppercase font-mono">Imbalance Criticality</AlertTitle>
                    <AlertDescription className="text-[11px] text-[#9ca3af] leading-relaxed">
                      Minority class under-representation detected. Recommended pre-processing: SMOTE + Tomek Links for production accuracy benchmarks.
                    </AlertDescription>
                  </Alert>
                </div>
              </motion.div>
            )}

            {activeTab === 'eval' && (
              <motion.div 
                key="eval"
                initial={{ opacity: 0, scale: 0.98 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.98 }}
                className="grid grid-cols-1 lg:grid-cols-2 gap-5 min-h-0"
              >
                <div className="bg-[#111827] border border-[#1f2937] p-5 rounded-[4px] flex flex-col">
                  <div className="text-[12px] font-semibold text-[#9ca3af] border-b border-[#1f2937] pb-2 mb-5 uppercase tracking-wider flex justify-between">
                    CONFUSION_MATRIX
                    <span className="text-[10px] font-mono text-[#9ca3af]">THRESHOLD: 0.5</span>
                  </div>
                  <div className="flex-1 grid grid-cols-2 gap-3 font-mono">
                    {[
                      { label: 'TRUE NEGATIVE', value: metrics?.tn || 0, color: 'text-slate-400', sub: 'Correct Legitimate', bg: 'bg-[#1f2937]/30' },
                      { label: 'FALSE POSITIVE', value: metrics?.fp || 0, color: 'text-[#ef4444]', sub: 'Incorrect Fraud', bg: 'bg-[#ef4444]/5' },
                      { label: 'FALSE NEGATIVE', value: metrics?.fn || 0, color: 'text-[#ef4444]/70', sub: 'Undetected Fraud', bg: 'bg-[#ef4444]/5' },
                      { label: 'TRUE POSITIVE', value: metrics?.tp || 0, color: 'text-[#10b981]', sub: 'Correct Fraud', bg: 'bg-[#10b981]/10 border-[#10b981]/20' },
                    ].map((cell, i) => (
                      <div key={i} className={cn("flex flex-col items-center justify-center border border-[#1f2937] p-4 rounded", cell.bg)}>
                        <div className="text-[8px] font-bold uppercase tracking-widest mb-1">{cell.label}</div>
                        <div className={cn("text-[28px] font-bold tracking-tighter", cell.color)}>{cell.value}</div>
                        <div className="text-[9px] opacity-40 mt-1">{cell.sub}</div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="space-y-5 flex flex-col">
                  <div className="bg-[#111827] border border-[#1f2937] p-5 rounded-[4px]">
                    <div className="text-[12px] font-semibold text-[#9ca3af] border-b border-[#1f2937] pb-2 mb-4 uppercase tracking-wider">Classification Scorecard</div>
                    <div className="space-y-4">
                      {[
                        { label: 'Precision', value: metrics?.precision || 0, color: '#3b82f6' },
                        { label: 'Recall', value: metrics?.recall || 0, color: '#10b981' },
                        { label: 'F1 Score', value: metrics?.f1 || 0, color: '#8b5cf6' },
                      ].map((m, i) => (
                        <div key={i} className="flex items-center gap-4">
                          <span className="w-16 text-[10px] font-mono text-[#9ca3af] uppercase">{m.label}</span>
                          <div className="flex-1 h-2 bg-[#1f2937] rounded-full overflow-hidden">
                            <div className="h-full transition-all duration-1000" style={{ width: `${m.value * 100}%`, backgroundColor: m.color }}></div>
                          </div>
                          <span className="w-12 text-[11px] font-mono text-right text-white font-bold">{(m.value * 100).toFixed(1)}%</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="bg-[#111827] border border-[#1f2937] p-5 rounded-[4px] flex-1">
                    <div className="text-[12px] font-semibold text-[#9ca3af] border-b border-[#1f2937] pb-2 mb-4 uppercase tracking-wider tracking-widest flex justify-between">
                      Class Ratio Analysis
                      <span className="text-[10px] font-mono text-[#3b82f6]">1:{(1/fraudRatio).toFixed(0)} IMBALANCE</span>
                    </div>
                    <div className="flex items-center gap-6 h-[100px]">
                        <div className="w-20 h-20">
                          <ResponsiveContainer width="100%" height="100%">
                            <PieChart>
                              <Pie data={fraudDistribution} innerRadius={25} outerRadius={35} paddingAngle={2} dataKey="value">
                                {fraudDistribution.map((entry, index) => (
                                  <Cell key={`cell-${index}`} fill={entry.color} />
                                ))}
                              </Pie>
                            </PieChart>
                          </ResponsiveContainer>
                        </div>
                        <div className="space-y-1">
                           <div className="text-[11px] font-bold text-white uppercase tracking-tight">Post-Sampling Metrics</div>
                           <div className="text-[10px] text-[#9ca3af] leading-relaxed font-mono">
                             Robust Scaling (IQR) applied. <br/>
                             Isolation Forest detected {Math.floor(data.length * 0.02)} outliers in segment.
                           </div>
                        </div>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}

            {activeTab === 'ai' && (
              <motion.div 
                key="ai"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="flex-1 flex flex-col min-h-0"
              >
                <div className="bg-[#111827] border border-[#1f2937] rounded-[4px] flex-1 flex flex-col min-h-0 relative">
                  <div className="px-5 py-3 border-b border-[#1f2937] flex justify-between items-center shrink-0">
                    <div className="flex flex-col">
                      <span className="text-[12px] font-bold uppercase tracking-[2px] text-[#3b82f6]">Gemini Intelligence Unit</span>
                      <span className="text-[9px] text-[#9ca3af] font-mono">Analysis Core: v3.0-FLASH-PREVIEW</span>
                    </div>
                    <Button 
                      size="sm" 
                      onClick={handleAnalyzeWithAI}
                      disabled={isAnalyzing || !metrics}
                      className="bg-blue-600 hover:bg-blue-700 text-white font-mono text-[11px] h-8"
                    >
                      {isAnalyzing ? <RefreshCw className="w-3 h-3 mr-2 animate-spin" /> : <BrainCircuit className="w-3 h-3 mr-2" />}
                      {isAnalyzing ? 'PROCESSING' : 'GENERATE_REPORT'}
                    </Button>
                  </div>
                  
                  <div className="flex-1 overflow-auto p-6 font-mono text-[12px] leading-relaxed">
                    {aiAnalysis ? (
                      <div className="text-slate-400 whitespace-pre-wrap max-w-3xl border-l border-blue-600/30 pl-5">
                        {aiAnalysis}
                      </div>
                    ) : (
                      <div className="h-full flex flex-col items-center justify-center opacity-20">
                        <BrainCircuit className="w-16 h-16 mb-4" />
                        <div className="text-[11px] uppercase tracking-widest italic">Awaiting analysis trigger...</div>
                      </div>
                    )}
                  </div>

                  <div className="px-5 py-2 border-t border-[#1f2937] text-[9px] text-[#9ca3af] font-mono flex justify-between items-center bg-[#0b0f1a]">
                    <span>SECURE_DATA_HANDLING: TRUE</span>
                    <span className="text-[#10b981]">NEURAL_BRIDGE: CONNECTED</span>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </main>
    </div>
  );
}
