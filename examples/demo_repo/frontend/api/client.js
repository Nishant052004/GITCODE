/**
 * Browser API client for acme-store. Handles auth headers, JSON parsing and
 * retrying failed requests with exponential backoff.
 */
const DEFAULT_RETRIES = 3;

export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

export class ApiClient {
  constructor(baseUrl, tokenProvider = () => null) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.tokenProvider = tokenProvider;
  }

  async request(path, { method = "GET", body, retries = DEFAULT_RETRIES } = {}) {
    let lastError;
    for (let attempt = 0; attempt < retries; attempt++) {
      try {
        return await this.attempt(path, method, body);
      } catch (err) {
        // 4xx errors are deterministic - retrying will not help.
        if (err instanceof ApiError && err.status < 500) throw err;
        lastError = err;
        await sleep(backoffMillis(attempt));
      }
    }
    throw lastError;
  }

  async attempt(path, method, body) {
    const headers = { "Content-Type": "application/json" };
    const token = this.tokenProvider();
    if (token) headers["Authorization"] = `Bearer ${token}`;
    const resp = await fetch(this.baseUrl + path, {
      method,
      headers,
      body: body ? JSON.stringify(body) : undefined,
    });
    if (!resp.ok) throw new ApiError(resp.status, `request failed: ${resp.status}`);
    return resp.json();
  }

  async login(email, password) {
    const data = await this.request("/auth/login", {
      method: "POST",
      body: { email, password },
      retries: 1,
    });
    return data.token;
  }

  async chargeInvoice(invoiceId) {
    return this.request(`/invoices/${invoiceId}/charge`, { method: "POST" });
  }
}

function backoffMillis(attempt) {
  return Math.min(8000, 250 * Math.pow(2, attempt));
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
