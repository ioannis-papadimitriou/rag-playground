import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/chat': 'http://localhost:5000',
      '/settings': 'http://localhost:5000',
      '/new_session': 'http://localhost:5000',
      '/switch_session': 'http://localhost:5000',
      '/delete_session': 'http://localhost:5000'
    }
  }
})