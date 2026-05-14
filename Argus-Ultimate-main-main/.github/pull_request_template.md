## Summary
<!-- What does this PR do? One sentence. -->

## Type of Change
- [ ] Bug fix
- [ ] New feature / enhancement
- [ ] Refactor / code hygiene
- [ ] Config / infra change
- [ ] Documentation

## Testing
- [ ] Unit tests pass (`pytest tests/ -m "not slow"`)
- [ ] Integration tests pass (`pytest tests_unified/`)
- [ ] Manually tested on paper mode (`python main.py paper --capital 1000 --cycles 3`)
- [ ] No hardcoded credentials or secrets

## Risk Assessment
<!-- For trading logic changes: describe the risk impact -->
- Risk modules affected: <!-- e.g. position sizing, circuit breakers, stop loss -->
- Config changes: <!-- yes/no; which keys -->
- Breaking changes: <!-- yes/no -->

## Checklist
- [ ] Code follows repo style (`ruff check .` passes)
- [ ] New or changed logic has tests
- [ ] `CHANGELOG.md` updated if this is a user-visible change
- [ ] PR title is descriptive (conventional commit format preferred)
