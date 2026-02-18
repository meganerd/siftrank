# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.x.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in siftrank, please report it by emailing the maintainers. Please do not open a public issue for security vulnerabilities.

## Known Limitations

### TOCTOU Race Condition in Path Validation (Mitigated)

**Description:**

A Time-of-Check-Time-of-Use (TOCTOU) race condition previously existed between path validation in `validateInputPath()` and subsequent file access operations. This has been mitigated by implementing file descriptor passing.

**Implementation:**

The TOCTOU window has been eliminated by opening files during validation and passing the file descriptor through the call chain:

1. `validateInputPath()` opens the file immediately and returns the open `*os.File` descriptor
2. `file.Stat()` is called on the open descriptor (not the path) to avoid re-resolution
3. The descriptor is passed to `RankFromFile()` and `loadDocumentsFromFile()`, which use it directly
4. For directory input, `RankFromFiles()` opens each enumerated file immediately before loading documents
5. The redundant `validatePath()` call in `loadDocumentsFromFile()` has been removed

This approach makes the check-and-use atomic — the file that was validated is the same file that is read.

**Previous Attack Vector (Now Mitigated):**

1. User runs: `siftrank -f /tmp/safe.txt`
2. `validateInputPath()` opens /tmp/safe.txt and returns the FD
3. ~~**[RACE WINDOW]** Attacker replaces file~~ — The open FD still references the original file
4. `loadDocumentsFromFile()` reads from the original FD, not the replaced path

**Residual Risk:**

- **Directory enumeration gap:** When processing a directory, there is a brief window between `enumerateFiles()` returning paths and `RankFromFiles()` opening each file. This window is minimal and constrained by the same threat model considerations below.
- **Symlink resolution:** `validateInputPath()` does not use `O_NOFOLLOW` (not portable in Go). Symlinks are followed at open time. However, once the FD is obtained, the opened file cannot be swapped.

**Threat Model:**

siftrank is designed as a single-user CLI tool that processes local files. The threat model assumes:

- **Trusted local environment** - Users run siftrank in environments they control
- **No privilege escalation** - Tool does not run with elevated privileges
- **User-owned data** - Input files are owned by the executing user

In this threat model, an attacker with local filesystem write access already has significant capabilities beyond what this TOCTOU vulnerability would provide.

**Defense-in-Depth Mitigations:**

1. **File descriptor passing** - Files are opened during validation and the FD is reused throughout the call chain
2. **FD-based Stat** - `file.Stat()` on the open descriptor avoids path re-resolution
3. **Error propagation** - File access failures are caught and propagated with sanitized error messages
4. **No privilege escalation** - Tool runs with user's existing permissions, cannot access files the user couldn't already access
5. **Resource limits** - MaxFilesPerDirectory (1000) and MaxDocuments (10000) limits prevent resource exhaustion

**Severity Assessment: Mitigated**

The primary TOCTOU race condition has been eliminated through file descriptor passing. The residual directory enumeration gap is classified as **Low** severity due to:

- Exploitation requires local filesystem write access (high attacker capability requirement)
- CLI tool threat model limits exposure (no remote exploitation vector)
- The window between enumeration and open is minimal (microseconds)

---

**Last Updated:** 2026-02-18
