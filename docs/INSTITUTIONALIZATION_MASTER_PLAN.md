# Institutionalization Master Plan

## Objective
Make Argus provably institutional-ready by combining technical controls, governance controls, and independently verifiable evidence.

This plan does not broaden trading permissions. Live trading remains gated and off by default.

## Institutional-Ready Exit Criteria
Argus is considered institutional-ready only when all items below are true:

1. `30d` soak evidence passes with no unresolved reconciliation breaks.
2. Walk-forward and promotion gates pass on current production candidate.
3. No duplicate intents/orders under retry/disconnect chaos tests.
4. Reconciliation freeze ownership and operator acknowledgement controls are in force.
5. Security key policy and least-privilege checks pass.
6. Governance artifacts are approved by accountable owners.
7. Disaster recovery test proves target `RTO/RPO`.
8. Manual legal/compliance attestations are approved and on file.

## Workstreams
1. Governance and compliance program.
2. Independent risk and limit governance.
3. Segregation of duties and approval workflow.
4. Execution quality and best-ex policy evidence.
5. Data quality, lineage, and deterministic replay.
6. Model risk management and promotion controls.
7. SRE reliability, SLOs, and incident process.
8. Security and secrets governance.
9. Treasury and counterparty controls.
10. External audit and investor reporting pack.

## In-Repo Delivery Artifacts
1. Program backlog: `docs/institutional/program_backlog.json`
2. Evidence checks manifest: `docs/institutional/evidence_manifest.json`
3. RACI matrix: `docs/institutional/RACI.md`
4. 90-day schedule: `docs/institutional/PLAN_90_DAYS.md`
5. Readiness checker: `scripts/institutional_readiness_check.py`
6. Windows runner: `ops/windows/Run-InstitutionalReadiness.ps1`

## Required Human Approvals (External to Code)
1. Licensing/registration legal attestation.
2. Compliance officer program approval.
3. Risk committee sign-off of limit hierarchy.
4. Security officer sign-off of key policy and pentest closure.
5. Operations owner sign-off of DR drill results.

If any required approval is missing, institutional readiness must remain `FAIL`.
