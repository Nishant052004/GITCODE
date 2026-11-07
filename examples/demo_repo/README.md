# acme-store

A tiny demo e-commerce library used as the sample repo for the
Codebase Intelligence & Developer Copilot. It is small on purpose: the whole
pipeline (ingest -> chunk -> embed -> index -> query) can run against it in
seconds on a laptop, and the eval ground truth in `eval/ground_truth.json`
points at these files.

Modules:
- `src/acme/auth.py` - password hashing, login, user lookup
- `src/acme/payments.py` - invoices, charging, refunds
- `src/acme/search.py` - inverted index for product search
- `src/acme/storage.py` - sqlite connection + user repository
- `src/acme/utils/logger.py` - JSON logging setup
- `frontend/api/client.js` - browser API client with retries
- `tests/test_auth.py` - auth tests
