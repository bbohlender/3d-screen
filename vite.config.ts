import { defineConfig } from "vite";

// https://vitejs.dev/config/
export default defineConfig({
    base: "/depth-prediction",
    build: {
        target: "esnext",
    }
});
