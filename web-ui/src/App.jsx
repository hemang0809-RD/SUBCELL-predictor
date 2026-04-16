import { Fragment, useMemo, useState } from 'react';
import {
  Atom,
  Beaker,
  CircleHelp,
  FileUp,
  FlaskConical,
  Microscope,
  Play,
  Sparkles,
} from 'lucide-react';
import {
  Bar,
  BarChart,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

const CLASSES = ['Nucleus', 'Cytoplasm', 'Mitochondrion'];
const CLASS_COLORS = {
  Nucleus: '#3b82f6',
  Cytoplasm: '#22c55e',
  Mitochondrion: '#f59e0b',
};

const EXAMPLE_INPUT = `>A0A0C5B5G6 | Nucleus
MRWQEMGYIFYPRKLR
>A1A4S6 | Cytoplasm
MGLQPLEFSDCYLDSPWFRERIRAHEAELERTNKFIKELIKDGKNLIAATKSLSVAQRKFAHSLRDFKFEFIGDAVTDDERCIDASLREFSNFLKNLEEQREIMALSVTETLIKPLEKFRKEQLGAVKEEKKKFDKETEKNYSLID
>A0A096LP01 | Mitochondrion
MYRNEFTAWYRRMSVVYGIGTWSVLGSLLYYSRTMAKSSVDQKDGSASEVPSELSERPKGFYVETVVTYKEDFVPNTEKILNYWKSWTGGPGTEP`;

const locationBlurb = {
  Nucleus: 'Stores DNA and hosts transcriptional regulation for gene expression.',
  Cytoplasm: 'Main metabolic compartment where translation and enzyme networks operate.',
  Mitochondrion: 'Energy-producing organelle with oxidative phosphorylation machinery.',
};

const cleanSequence = (value = '') => value.replace(/\s+/g, '').toUpperCase().replace(/[^ACDEFGHIKLMNPQRSTVWY]/g, '');

const parseSequences = (rawText) => {
  const lines = rawText.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  const records = [];
  let current = null;

  lines.forEach((line, index) => {
    if (line.startsWith('>')) {
      if (current?.sequence) {
        records.push({ ...current, sequence: cleanSequence(current.sequence) });
      }
      current = { name: line.slice(1).trim() || `Sequence ${records.length + 1}`, sequence: '' };
      return;
    }

    if (!current) {
      records.push({ name: `Sequence ${index + 1}`, sequence: cleanSequence(line) });
      return;
    }

    current.sequence += line;
  });

  if (current?.sequence) {
    records.push({ ...current, sequence: cleanSequence(current.sequence) });
  }

  return records.filter((item) => item.sequence.length > 0);
};

const generateProbabilities = () => {
  const preferredIndex = Math.floor(Math.random() * CLASSES.length);
  const logits = CLASSES.map((_, index) => (Math.random() * 0.8 + (index === preferredIndex ? 1.4 : 0.15)));
  const sum = logits.reduce((acc, value) => acc + value, 0);
  const probs = logits.map((value) => value / sum);

  return CLASSES.reduce((acc, key, idx) => {
    acc[key] = probs[idx];
    return acc;
  }, {});
};

const truncate = (value, length = 30) => (value.length <= length ? value : `${value.slice(0, length)}…`);

function App() {
  const [inputText, setInputText] = useState('');
  const [results, setResults] = useState([]);
  const [expandedRows, setExpandedRows] = useState({});

  const averageConfidence = useMemo(() => {
    if (!results.length) return 0;
    const total = results.reduce((sum, row) => sum + row.confidence, 0);
    return total / results.length;
  }, [results]);

  const distribution = useMemo(() => {
    const bucket = CLASSES.map((name) => ({ name, value: 0 }));
    results.forEach((row) => {
      const found = bucket.find((item) => item.name === row.predictedLocation);
      if (found) found.value += 1;
    });
    return bucket;
  }, [results]);

  const handlePredict = () => {
    const parsed = parseSequences(inputText);
    const nextResults = parsed.map((item, idx) => {
      const probs = generateProbabilities();
      const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1]);
      const predictedLocation = sorted[0][0];
      return {
        id: `${item.name}-${idx}`,
        name: item.name,
        sequence: item.sequence,
        predictedLocation,
        confidence: sorted[0][1],
        probabilities: probs,
      };
    });
    setResults(nextResults);
    setExpandedRows({});
  };

  const handleUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const text = await file.text();
    setInputText(text);
  };

  const toggleExpanded = (id) => {
    setExpandedRows((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  return (
    <div className="min-h-screen bg-[var(--bg)] text-[var(--fg)]">
      <div className="mx-auto grid w-full max-w-7xl gap-6 p-4 md:grid-cols-12 md:p-8">
        <section className="space-y-6 md:col-span-8">
          <div className="rounded-2xl border border-slate-200/70 bg-[var(--card)] p-5 shadow-sm dark:border-slate-700">
            <div className="mb-4 flex items-center gap-3">
              <FlaskConical className="text-sky-500" />
              <h1 className="text-xl font-semibold">SubCell-Predictor Mock Inference UI</h1>
            </div>
            <p className="mb-4 text-sm text-slate-600 dark:text-slate-300">
              Paste protein sequences as raw strings (one per line) or FASTA. Predictions are mocked client-side to emulate
              production behavior.
            </p>
            <textarea
              value={inputText}
              onChange={(event) => setInputText(event.target.value)}
              placeholder=">P12345\nMSTNPKPQRK..."
              className="h-56 w-full rounded-xl border border-slate-300 bg-transparent p-3 font-mono text-sm outline-none ring-sky-400 focus:ring-2 dark:border-slate-600"
            />
            <div className="mt-4 flex flex-wrap gap-3">
              <button
                type="button"
                onClick={() => setInputText(EXAMPLE_INPUT)}
                className="inline-flex items-center gap-2 rounded-lg bg-slate-800 px-4 py-2 text-sm font-medium text-white hover:bg-slate-700 dark:bg-slate-200 dark:text-slate-900"
              >
                <Sparkles size={16} /> Try Example
              </button>
              <label className="inline-flex cursor-pointer items-center gap-2 rounded-lg border border-slate-300 px-4 py-2 text-sm hover:bg-slate-50 dark:border-slate-600 dark:hover:bg-slate-800/60">
                <FileUp size={16} /> Upload .fasta
                <input type="file" accept=".fasta,.fa,.txt" className="hidden" onChange={handleUpload} />
              </label>
              <button
                type="button"
                onClick={handlePredict}
                className="inline-flex items-center gap-2 rounded-lg bg-sky-600 px-4 py-2 text-sm font-medium text-white hover:bg-sky-500"
              >
                <Play size={16} /> Predict
              </button>
            </div>
          </div>

          <div className="rounded-2xl border border-slate-200/70 bg-[var(--card)] p-5 shadow-sm dark:border-slate-700">
            <div className="mb-4 flex items-center gap-2">
              <Microscope className="text-emerald-500" size={18} />
              <h2 className="font-semibold">Results Table</h2>
            </div>
            {!results.length && <p className="text-sm text-slate-500">No predictions yet. Add sequences and click Predict.</p>}
            {!!results.length && (
              <div className="overflow-x-auto">
                <table className="min-w-full text-left text-sm">
                  <thead>
                    <tr className="border-b border-slate-200 text-slate-500 dark:border-slate-700">
                      <th className="py-2 pr-3">Name</th>
                      <th className="py-2 pr-3">Sequence</th>
                      <th className="py-2 pr-3">Predicted Location</th>
                      <th className="py-2">Confidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.map((row) => (
                      <Fragment key={row.id}>
                        <tr
                          onClick={() => toggleExpanded(row.id)}
                          className="cursor-pointer border-b border-slate-100 transition hover:bg-slate-50/80 dark:border-slate-800 dark:hover:bg-slate-800/40"
                          style={{ backgroundColor: `${CLASS_COLORS[row.predictedLocation]}14` }}
                        >
                          <td className="py-3 pr-3 align-top">{row.name}</td>
                          <td className="py-3 pr-3 font-mono text-xs align-top">{truncate(row.sequence)}</td>
                          <td className="py-3 pr-3 align-top">
                            <span className="rounded-full px-2 py-1 text-xs font-medium" style={{ color: CLASS_COLORS[row.predictedLocation] }}>
                              {row.predictedLocation}
                            </span>
                          </td>
                          <td className="py-3 align-top">
                            <div className="h-2.5 w-36 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-700">
                              <div
                                className="h-full rounded-full"
                                style={{ width: `${(row.confidence * 100).toFixed(1)}%`, backgroundColor: CLASS_COLORS[row.predictedLocation] }}
                              />
                            </div>
                            <div className="mt-1 text-xs text-slate-500">{(row.confidence * 100).toFixed(1)}%</div>
                          </td>
                        </tr>
                        {expandedRows[row.id] && (
                          <tr className="border-b border-slate-100 dark:border-slate-800">
                            <td colSpan={4} className="px-2 py-3">
                              <div className="h-24 w-full rounded-lg border border-slate-200 p-2 dark:border-slate-700">
                                <ResponsiveContainer width="100%" height="100%">
                                  <BarChart
                                    data={[
                                      {
                                        name: 'Probabilities',
                                        ...row.probabilities,
                                      },
                                    ]}
                                    layout="vertical"
                                    margin={{ top: 10, right: 20, left: 20, bottom: 0 }}
                                  >
                                    <XAxis type="number" tickFormatter={(value) => `${Math.round(value * 100)}%`} domain={[0, 1]} />
                                    <YAxis type="category" dataKey="name" hide />
                                    <Tooltip formatter={(value) => `${(value * 100).toFixed(2)}%`} />
                                    {CLASSES.map((name) => (
                                      <Bar key={name} dataKey={name} stackId="a" fill={CLASS_COLORS[name]} radius={[4, 4, 4, 4]} />
                                    ))}
                                  </BarChart>
                                </ResponsiveContainer>
                              </div>
                            </td>
                          </tr>
                        )}
                      </Fragment>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {results.length >= 3 && (
            <div className="rounded-2xl border border-slate-200/70 bg-[var(--card)] p-5 shadow-sm dark:border-slate-700">
              <div className="mb-2 flex items-center gap-2">
                <Beaker size={18} className="text-purple-500" />
                <h2 className="font-semibold">Batch Summary</h2>
              </div>
              <div className="grid gap-4 md:grid-cols-2">
                <div className="h-52 rounded-xl border border-slate-200 dark:border-slate-700">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie data={distribution} dataKey="value" nameKey="name" innerRadius={45} outerRadius={72} paddingAngle={2}>
                        {distribution.map((entry) => (
                          <Cell key={entry.name} fill={CLASS_COLORS[entry.name]} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
                <div className="rounded-xl border border-slate-200 p-4 dark:border-slate-700">
                  <p className="text-sm text-slate-500">Average confidence</p>
                  <p className="mt-2 text-3xl font-semibold">{(averageConfidence * 100).toFixed(1)}%</p>
                  <div className="mt-4 space-y-2">
                    {distribution.map((item) => (
                      <div key={item.name} className="flex items-center justify-between text-sm">
                        <span className="flex items-center gap-2">
                          <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: CLASS_COLORS[item.name] }} />
                          {item.name}
                        </span>
                        <span>{item.value} sequences</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}
        </section>

        <aside className="space-y-6 md:col-span-4">
          <div className="rounded-2xl border border-slate-200/70 bg-[var(--card)] p-5 shadow-sm dark:border-slate-700">
            <h2 className="mb-3 flex items-center gap-2 font-semibold">
              <Atom size={18} className="text-pink-500" /> Compartment Guide
            </h2>
            <svg viewBox="0 0 260 180" className="mb-4 w-full rounded-xl border border-slate-200 bg-white p-3 dark:border-slate-700 dark:bg-slate-900">
              <ellipse cx="130" cy="92" rx="110" ry="70" fill="#e2e8f0" />
              <ellipse cx="130" cy="92" rx="86" ry="50" fill="#f8fafc" />
              <circle cx="96" cy="84" r="22" fill="#93c5fd" />
              <ellipse cx="160" cy="78" rx="20" ry="12" fill="#fde68a" />
              <ellipse cx="160" cy="112" rx="20" ry="12" fill="#fde68a" />
              <text x="84" y="88" fontSize="10" fill="#1e3a8a">Nucleus</text>
              <text x="145" y="80" fontSize="9" fill="#92400e">Mito</text>
              <text x="75" y="138" fontSize="10" fill="#166534">Cytoplasm</text>
            </svg>
            <div className="space-y-2 text-sm">
              {CLASSES.map((name) => (
                <div key={name}>
                  <p className="font-medium" style={{ color: CLASS_COLORS[name] }}>{name}</p>
                  <p className="text-slate-600 dark:text-slate-300">{locationBlurb[name]}</p>
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-2xl border border-slate-200/70 bg-[var(--card)] p-5 shadow-sm dark:border-slate-700">
            <h2 className="mb-3 flex items-center gap-2 font-semibold">
              <CircleHelp size={18} className="text-cyan-500" /> Model Metadata
            </h2>
            <p className="text-sm text-slate-600 dark:text-slate-300">
              Gradient Boosting classifier, 432 features (AAC + DPC + Physicochemical), trained on 10,661 annotated human
              proteins from UniProt.
            </p>
            <div className="mt-4 space-y-2 rounded-xl border border-slate-200 p-3 text-sm dark:border-slate-700">
              <div className="flex justify-between"><span>AAC</span><span>20 features</span></div>
              <div className="flex justify-between"><span>DPC</span><span>400 features</span></div>
              <div className="flex justify-between"><span>Physicochemical</span><span>12 features</span></div>
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}

export default App;
