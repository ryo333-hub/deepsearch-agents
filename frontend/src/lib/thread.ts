const STORAGE_KEY = "deepsearch.thread_id";

export function createThreadId(): string {
  if (crypto.randomUUID) {
    return crypto.randomUUID();
  }

  return `manual-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export function getStoredThreadId(): string {
  // An explicit prepared-Demo link selects its session; KB access still uses
  // the backend's existing session validation. Consume once so New Session works.
  const url = new URL(window.location.href);
  const prepared = url.searchParams.get("thread_id");
  if (prepared && /^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$/.test(prepared)) {
    storeThreadId(prepared);
    url.searchParams.delete("thread_id");
    window.history.replaceState(null, "", url);
    return prepared;
  }
  const existing = window.localStorage.getItem(STORAGE_KEY);
  if (existing) {
    return existing;
  }

  const threadId = createThreadId();
  window.localStorage.setItem(STORAGE_KEY, threadId);
  return threadId;
}

export function storeThreadId(threadId: string): void {
  window.localStorage.setItem(STORAGE_KEY, threadId);
}
