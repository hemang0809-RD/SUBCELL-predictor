/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        mono: ['"JetBrains Mono"', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
      colors: {
        nucleus: '#3b82f6',
        cytoplasm: '#22c55e',
        mitochondrion: '#f59e0b',
      },
    },
  },
  plugins: [],
};
