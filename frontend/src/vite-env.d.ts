/// <reference types="vite/client" />


interface ImportMetaEnv {
  readonly VITE_API_URL?: string;
  readonly VITE_USE_MOCK?: string;
  readonly VITE_MOCK_HEALTH?: string;
  readonly VITE_MOCK_NO_RESULTS?: string;
}


interface ImportMeta {
  readonly env: ImportMetaEnv;
}



