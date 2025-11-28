// FIX: Removed reference to 'vite/client' which was causing errors.
// Added definitions for ImportMetaEnv and process.env to comply with usage.

interface ImportMetaEnv {
  readonly VITE_API_KEY: string;
  [key: string]: any;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

declare var process: {
  env: {
    API_KEY: string;
    [key: string]: string | undefined;
  }
};
