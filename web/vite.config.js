import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base: 기본 "/"(로컬·Docker·Vercel). GitHub Pages 빌드는 VITE_BASE=/wafer-defect-suite/ 주입.
export default defineConfig({
  plugins: [react()],
  base: process.env.VITE_BASE || "/",
});
