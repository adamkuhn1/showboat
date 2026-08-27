import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base: "./" keeps built asset paths relative so each app works when served
// standalone AND when embedded in the portfolio shell via iframe.
export default defineConfig({
  plugins: [react()],
  base: "./",
});
