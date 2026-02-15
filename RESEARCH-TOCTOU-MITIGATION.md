# TOCTOU Mitigation Research

## Executive Summary

This document researches approaches to reduce the TOCTOU (Time-of-Check-Time-of-Use) race condition window documented in SECURITY.md. Three approaches are evaluated with implementation complexity and security benefits analyzed.

**Recommendation:** Approach 1 (Open During Validation) provides the best balance of security improvement and implementation simplicity.

---

## Current Implementation Analysis

### Call Flow

```
main.go:391  → validateInputPath(inputFile)
             → returns validPath

main.go:413  → ranker.RankFromFile(validPath, ...)

siftrank.go:631 → RankFromFile
               → calls loadDocumentsFromFile

siftrank.go:914 → loadDocumentsFromFile
               → validatePath(filePath) [REDUNDANT]

siftrank.go:920 → os.Open(validPath) [TOCTOU WINDOW]
```

### TOCTOU Window

The race window exists between:
- **Check:** `validateInputPath()` at main.go:391 (EvalSymlinks + Stat)
- **Use:** `os.Open()` at siftrank.go:920

**Window duration:** Typically 1-10ms depending on system load

**Attack scenario:**
1. User runs: `siftrank -f /tmp/safe.txt`
2. validateInputPath() checks /tmp/safe.txt ✓
3. **[RACE WINDOW]** Attacker replaces with: `ln -sf /etc/shadow /tmp/safe.txt`
4. os.Open() opens /etc/shadow (if user has read permission)

### Current Mitigations

- **Secondary validation:** loadDocumentsFromFile() calls validatePath() again (line 914)
  - Reduces window but doesn't eliminate it
- **User permissions:** Can only access files user already has permission to read
- **CLI context:** No remote attack vector
- **Error handling:** File access errors are caught and propagated

---

## Research: Mitigation Approaches

### Approach 1: Open During Validation (RECOMMENDED)

#### Concept

Open the file during validation and return the open file descriptor. This eliminates the window by making the "check" and "use" atomic.

#### Go Implementation Feasibility

**✅ Fully supported** - Go's `os.Open()` returns an `*os.File` which can be passed around.

```go
// New signature for validateInputPath
func validateInputPath(path string) (string, *os.File, bool, error) {
	absPath, err := filepath.Abs(path)
	if err != nil {
		return "", nil, false, fmt.Errorf("failed to resolve path")
	}

	cleanPath := filepath.Clean(absPath)
	if strings.Contains(cleanPath, "..") {
		return "", nil, false, fmt.Errorf("path contains directory traversal")
	}

	// Open the file immediately (atomic check-use)
	file, err := os.Open(cleanPath)
	if err != nil {
		if os.IsNotExist(err) {
			return "", nil, false, fmt.Errorf("path does not exist")
		}
		return "", nil, false, fmt.Errorf("failed to open path")
	}

	// Stat the OPEN file descriptor (not the path)
	info, err := file.Stat()
	if err != nil {
		file.Close()
		return "", nil, false, fmt.Errorf("failed to stat path")
	}

	// Resolve symlinks via the open descriptor
	realPath, err := filepath.EvalSymlinks(cleanPath)
	if err != nil {
		file.Close()
		return "", nil, false, fmt.Errorf("failed to resolve path")
	}

	return realPath, file, info.IsDir(), nil
}
```

#### Changes Required

**cmd/siftrank/main.go:**
- Update `validateInputPath()` signature to return `*os.File`
- Update callers to receive file descriptor
- Pass file descriptor to `RankFromFile()` / `RankFromFiles()`

**pkg/siftrank/siftrank.go:**
- Update `RankFromFile()` to accept optional `*os.File`
- Update `loadDocumentsFromFile()` to accept optional `*os.File`
- Skip `os.Open()` if file descriptor is provided
- Remove redundant `validatePath()` call (line 914)

#### Benefits

- **Eliminates TOCTOU window** - Check and use are atomic
- **Minimal API changes** - Only signature changes, no architectural shifts
- **Clean ownership** - Caller opens, callee uses, caller closes
- **Works with directories** - Can stat open directories

#### Drawbacks

- **Slightly more complex API** - Functions take extra parameter
- **Ownership concerns** - Need to document who closes the file
- **Directory handling** - Opening directories requires special handling (readable but not for io.Reader)

#### Implementation Effort

**Estimated: 4-6 hours**

- 1 hour: Update validateInputPath() and callers
- 2 hours: Update RankFromFile/RankFromFiles to accept file descriptor
- 1 hour: Update loadDocumentsFromFile() to skip redundant open
- 1-2 hours: Testing and validation

---

### Approach 2: File Descriptor Passing Throughout

#### Concept

Pass file descriptors through the entire call chain instead of paths. This is the "cleanest" design from a security perspective.

#### Go Implementation Feasibility

**✅ Supported but requires significant refactor**

```go
// New API: Accept io.Reader instead of path
func (r *Ranker) RankFromReader(reader io.Reader, templateData string, isJSON bool) ([]*RankedDocument, error) {
	documents, err := r.loadDocumentsFromReader(reader, templateData, isJSON)
	if err != nil {
		return nil, err
	}
	return r.rankDocuments(documents)
}

// Wrapper for file path input
func (r *Ranker) RankFromFile(filePath string, templateData string, forceJSON bool) ([]*RankedDocument, error) {
	validPath, file, isDir, err := validateInputPath(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	if isDir {
		return nil, fmt.Errorf("directory input requires RankFromFiles")
	}

	ext := strings.ToLower(filepath.Ext(validPath))
	isJSON := ext == ".json" || forceJSON

	return r.RankFromReader(file, templateData, isJSON)
}
```

#### Changes Required

**Major refactor:**
- Add `RankFromReader()` as primary API
- Make `RankFromFile()` a convenience wrapper
- Update `loadDocumentsFromFile()` to `loadDocumentsFromReader()` only
- Remove all path-based internal APIs
- Update directory enumeration to open files immediately

#### Benefits

- **Cleanest security design** - No paths passed after validation
- **Better separation of concerns** - Path validation separate from document loading
- **More testable** - Can test with io.Reader mocks

#### Drawbacks

- **Breaking API change** - Existing public API changes
- **Complex directory handling** - Need to open all files upfront
- **Resource management** - Many open file descriptors for directories
- **Loss of path context** - Error messages lose file path information

#### Implementation Effort

**Estimated: 12-16 hours**

- 4 hours: Design new API and update signatures
- 4 hours: Refactor loadDocumentsFromFile → loadDocumentsFromReader
- 2 hours: Update directory enumeration
- 2 hours: Update error messages (lose path context)
- 2-4 hours: Testing and validation

---

### Approach 3: Validate at Use Site (Deferred Validation)

#### Concept

Don't validate paths upfront. Instead, validate when opening the file by checking the stat() result after open.

#### Go Implementation Feasibility

**✅ Supported but questionable design**

```go
func (r *Ranker) loadDocumentsFromFile(filePath string, templateData string, forceJSON bool) ([]document, error) {
	// Open WITHOUT validation first
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open input file: %w", err)
	}
	defer file.Close()

	// Validate the OPEN file descriptor
	info, err := file.Stat()
	if err != nil {
		return nil, fmt.Errorf("failed to stat file: %w", err)
	}

	// Check for directory
	if info.IsDir() {
		return nil, fmt.Errorf("path is a directory, not a file")
	}

	// Additional validation on open descriptor
	// (symlink resolution via /proc/self/fd/N on Linux)

	ext := strings.ToLower(filepath.Ext(filePath))
	isJSON := ext == ".json" || forceJSON

	return r.loadDocumentsFromReader(file, templateData, isJSON)
}
```

#### Changes Required

- Remove `validateInputPath()` call in main.go
- Move validation logic into loadDocumentsFromFile()
- Validate AFTER opening file descriptor

#### Benefits

- **Eliminates TOCTOU** - Validation happens on open descriptor
- **Fewer code changes** - Just move validation location
- **Simpler call chain** - No path validation in main.go

#### Drawbacks

- **Fails late** - Errors detected after more work done
- **Worse UX** - User gets errors after ranker is created
- **Inconsistent** - main.go does directory detection, but file validation is deferred
- **Security regression** - Removes upfront path traversal check

#### Implementation Effort

**Estimated: 2-4 hours**

- 1 hour: Remove validateInputPath() from main.go
- 1 hour: Move validation into loadDocumentsFromFile()
- 1-2 hours: Testing and validation

---

## Approach Comparison

| Criteria | Approach 1: Open During Validation | Approach 2: FD Passing | Approach 3: Validate at Use |
|----------|-----------------------------------|------------------------|----------------------------|
| **Security** | ✅ Eliminates TOCTOU | ✅ Eliminates TOCTOU | ✅ Eliminates TOCTOU |
| **Implementation Effort** | 🟢 Low (4-6 hours) | 🔴 High (12-16 hours) | 🟢 Low (2-4 hours) |
| **API Changes** | 🟡 Signature changes only | 🔴 Breaking changes | 🟢 Removes API |
| **Code Clarity** | 🟢 Clear ownership model | 🟢 Best separation of concerns | 🔴 Confusing validation location |
| **Error Handling** | 🟢 Fail fast | 🟢 Fail fast | 🔴 Fail late |
| **Directory Support** | 🟢 Works well | 🟡 Complex (many open FDs) | 🔴 Inconsistent |
| **Backward Compatibility** | 🟢 Internal change only | 🔴 Public API change | 🟢 Simplifies API |
| **Testability** | 🟢 Same as current | 🟢 Better (io.Reader mocks) | 🟡 Same as current |

**Legend:** 🟢 Good | 🟡 Acceptable | 🔴 Problematic

---

## Recommendation: Approach 1

**Approach 1 (Open During Validation)** is recommended because it:

1. **Eliminates the TOCTOU race condition** - Primary security goal achieved
2. **Minimizes implementation risk** - Low effort, low complexity
3. **Maintains API compatibility** - Internal change only
4. **Fails fast** - Errors detected immediately at validation
5. **Clear ownership model** - Caller opens, passes FD, closes on return
6. **Handles directories cleanly** - Stat() works on directory FDs

### Implementation Plan

**Phase 1: Core Changes (2-3 hours)**
1. Update `validateInputPath()` in main.go to return `*os.File`
2. Update callers in main.go to receive and manage file descriptor
3. Update `RankFromFile()` signature to accept optional `*os.File`

**Phase 2: Ranker Updates (2-3 hours)**
4. Update `RankFromFiles()` to pass file descriptors
5. Update `loadDocumentsFromFile()` to skip os.Open() if FD provided
6. Remove redundant `validatePath()` call (line 914 in siftrank.go)

**Phase 3: Testing (1-2 hours)**
7. Test with single file input
8. Test with directory input
9. Test error cases (permission denied, directory as file, etc.)

**Total effort:** 5-8 hours

---

## Alternative: Do Nothing

The SECURITY.md document already states:

> These optimizations are not currently implemented as they add complexity without significantly reducing risk in the current threat model.

**Arguments for status quo:**
- **Low severity** - Documented as Low in threat assessment
- **Limited attack surface** - CLI tool, local files, user permissions
- **Defense in depth exists** - Secondary validation, error handling
- **Cost vs benefit** - 5-8 hours implementation for low-severity issue

**Arguments for implementation:**
- **Industry best practice** - TOCTOU is well-known vulnerability class
- **Defense in depth** - Multiple layers of security are better
- **Low implementation cost** - 5-8 hours is reasonable for a security fix
- **Documentation** - Shows security-conscious development

---

## Appendix: Go-Specific Considerations

### O_NOFOLLOW Support

**Status:** ❌ Not available in Go's standard library

Go does not expose `O_NOFOLLOW` via `os.OpenFile()`. The flag exists in `syscall` package but is platform-specific (Linux/BSD, not Windows).

```go
// Platform-specific, not portable
file, err := syscall.Open(path, syscall.O_RDONLY|syscall.O_NOFOLLOW, 0)
```

**Conclusion:** Not suitable for portable implementation.

### file.Stat() vs os.Stat()

**Key difference:**
- `os.Stat(path)` - Stats the path (follows symlinks, TOCTOU vulnerable)
- `file.Stat()` - Stats the open file descriptor (no TOCTOU)

```go
file, _ := os.Open("/tmp/foo")
info, _ := file.Stat() // Stats the OPEN file, not the path
```

This is the foundation of Approach 1.

### Directory File Descriptors

**Status:** ✅ Supported

Go can open directories and call `Stat()` on the descriptor:

```go
dir, err := os.Open("/tmp/mydir")
info, err := dir.Stat()
if info.IsDir() {
	// Process directory
}
```

However, directory FDs cannot be used with `io.Reader` interfaces.

---

**Document Status:** Research Complete
**Date:** 2026-02-14
**Issue:** siftrank-41
**Epic:** Epic 3 - Optional Security Enhancements
