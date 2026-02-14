# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.x.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in siftrank, please report it by emailing the maintainers. Please do not open a public issue for security vulnerabilities.

## Known Limitations

### TOCTOU Race Condition in Path Validation (Low Severity)

**Description:**

A Time-of-Check-Time-of-Use (TOCTOU) race condition exists between path validation in `validateInputPath()` and subsequent file access operations. This creates a narrow time window where an attacker with local filesystem write access could modify files or symlink targets after validation but before file reading.

**Attack Vector:**

1. Attacker provides a legitimate file path to siftrank
2. Path validation completes successfully
3. During the window between validation and file access, attacker:
   - Replaces the file with a symlink to a sensitive file (e.g., `/etc/shadow`)
   - Modifies a symlink target to point to a different location
4. File reading operation accesses the modified target

**Exploitability Constraints:**

- **Requires local filesystem write access** - Attacker must have permissions to modify files in the target directory
- **Narrow time window** - Race condition window is typically milliseconds
- **CLI tool context** - siftrank runs as a local command-line tool, not a networked service
- **User permissions only** - File access is constrained by the user's existing permissions

**Threat Model:**

siftrank is designed as a single-user CLI tool that processes local files. The threat model assumes:

- **Trusted local environment** - Users run siftrank in environments they control
- **No privilege escalation** - Tool does not run with elevated privileges
- **User-owned data** - Input files are owned by the executing user

In this threat model, an attacker with local filesystem write access already has significant capabilities beyond what this TOCTOU vulnerability would provide.

**Mitigations (Defense-in-Depth):**

1. **Secondary validation** - File access operations in `loadDocumentsFromFile()` perform additional validation and error handling
2. **Error propagation** - File access failures are caught and propagated with sanitized error messages
3. **No privilege escalation** - Tool runs with user's existing permissions, cannot access files the user couldn't already access
4. **Resource limits** - MaxFilesPerDirectory (1000) and MaxDocuments (10000) limits prevent resource exhaustion

**Severity Assessment: Low**

This vulnerability is classified as **Low** severity because:

- Exploitation requires local filesystem write access (high attacker capability requirement)
- CLI tool threat model limits exposure (no remote exploitation vector)
- Existing permissions constrain impact (no privilege escalation)
- Defense-in-depth mitigations reduce practical exploitability

**Recommendation:**

For users concerned about this limitation:

- Run siftrank only in directories you control
- Avoid processing files from untrusted users on multi-user systems
- Use file permissions to restrict write access to input directories

**Future Improvements:**

Potential optimizations to reduce the TOCTOU window:

- File descriptor passing: Open files during validation and pass descriptors to avoid re-opening
- Atomic validation: Use `os.Open()` with `O_NOFOLLOW` during validation and retain the file descriptor

These optimizations are not currently implemented as they add complexity without significantly reducing risk in the current threat model.

---

**Last Updated:** 2026-02-14
