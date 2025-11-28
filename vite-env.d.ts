// FIX: Removed reference to 'vite/client' which was causing errors.
// Added definitions for ImportMetaEnv and process.env to comply with usage.

export {};

declare global {
  interface ImportMetaEnv {
    readonly VITE_API_KEY: string;
    [key: string]: any;
  }

  interface ImportMeta {
    readonly env: ImportMetaEnv;
  }

  // FIX: Replaced redeclaration of 'process' with augmentation of the NodeJS namespace
  // to avoid a conflict with existing global type definitions for 'process'. This correctly
  // adds the API_KEY type to process.env for use in the client-side application.
  namespace NodeJS {
    interface ProcessEnv {
      API_KEY: string;
    }
  }
}
