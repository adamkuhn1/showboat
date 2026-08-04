import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["src/**/*.test.ts", "training/**/*.test.ts"],
    environment: "node",
  },
});
