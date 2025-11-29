// vite-env.d.ts

// By declaring `process` in the global scope, we inform TypeScript that this
// variable exists, even in a browser environment. This resolves the `TS2580`
// compile-time error. Vite's `define` configuration handles the actual
// replacement of `process.env.API_KEY` with a string value at build time.
declare global {
  // FIX: Replaced `var process` with an augmentation of the `NodeJS.ProcessEnv`
  // interface. This avoids redeclaring the `process` variable, which is likely
  // already defined by another type declaration (e.g., from `@types/node`),
  // and correctly adds the `API_KEY` type to `process.env`.
  namespace NodeJS {
    interface ProcessEnv {
      API_KEY: string;
    }
  }
}

// Adding an empty export statement ensures that this file is treated as a module,
// which is necessary for `declare global` to work correctly.
export {};
