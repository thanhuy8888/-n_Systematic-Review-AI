/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./*.{js,ts,jsx,tsx}",
    "./components/**/*.{js,ts,jsx,tsx}",
    "./services/**/*.{js,ts,jsx,tsx}"
  ],
  theme: {
    extend: {
      colors: {
        'vnu-blue': '#0054A6',
        'vnu-yellow': '#FDB813',
        'vnu-white': '#FFFFFF',
      }
    },
  },
  plugins: [],
}
