# Access Control Policy

Owner: Security  
Review cadence: Quarterly

## Control Objectives
1. Enforce least privilege for keys and operators.
2. Separate duties across development, approval, and live operations.
3. Require MFA and auditable approvals for privileged actions.

## Requirements
1. Live deployment requires independent approver separate from code author.
2. Secrets rotation metadata must be current and verified at startup.
3. Over-permissioned exchange keys fail readiness validation.
