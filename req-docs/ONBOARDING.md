# Onboarding Guide

## What is this project and what problem does it solve?
Requests is a Python HTTP library that provides a simple, elegant, and high-level API for making HTTP/1.1 requests. It solves the complexity of native HTTP libraries (such as manual query string formatting, form-encoding, cookie persistence, and connection pooling) by offering an intuitive interface for tasks like sending GET/POST requests, handling sessions, managing authentication, and parsing responses.

## How do I install and run it?
Not documented in the repository beyond standard Python setup tools (e.g., `setup.py` requiring Python 3.10+).

## Where should I start reading, and why those files?
To understand the codebase effectively, read the files in the following order:

1. **`src/requests/__init__.py`**
   This is the package entry point. It initializes the package, checks third-party dependency compatibility, configures warnings and logging, and re-exports the entire public API.

2. **`src/requests/api.py`**
   Defines the core functional user-facing API functions (such as `get`, `post`, `request`, etc.). It demonstrates how requests are instantiated by creating a temporary `Session` and forwarding calls.

3. **`src/requests/sessions.py`**
   Implements the `Session` class and redirect mixins. It manages and persists settings, cookies, authentication states, and redirects across multiple requests.

4. **`src/requests/models.py`**
   Defines the foundational data models (`Request`, `PreparedRequest`, `Response`) and mixing classes responsible for request encoding and hooks.

5. **`src/requests/adapters.py`**
   Defines transport adapters (like `HTTPAdapter`) that bridge Requests with `urllib3` to handle connection pooling, retries, proxies, and SSL/TLS verification.

6. **`src/requests/auth.py`**
   Implements authentication mechanisms including HTTP Basic Auth, Proxy Auth, and Digest Auth.

7. **`src/requests/utils.py`**
   Provides a wide collection of utility functions for header parsing, encoding detection, proxy handling, and netrc lookup.

## How is it tested?
Not explicitly detailed in the readable files, though test files exist in the `tests/` directory (e.g., `tests/test_requests.py`, `tests/test_adapters.py`, etc.).
