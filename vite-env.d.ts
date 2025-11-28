// Fixed: Removed reference to vite/client to resolve type definition error.
declare namespace NodeJS {
  interface ProcessEnv {
    API_KEY?: string;
    [key: string]: string | undefined;
  }
}
