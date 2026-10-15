import { defineConfig } from "vitest/config";


export default defineConfig({
  test: {
    include: ["src/**/*.test.ts"],
    // config/env.ts requires MONGO_URI; unit tests never connect to it.
    env: { MONGO_URI: "mongodb://127.0.0.1:27017/unit-test" },
  },
});



