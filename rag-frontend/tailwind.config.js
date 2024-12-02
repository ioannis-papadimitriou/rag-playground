/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',  // Add this line
  theme: {
    extend: {
      maxHeight: {
        'dynamic': 'calc(100vh - 200px)',
      }
    },
  },
  plugins: [],
}