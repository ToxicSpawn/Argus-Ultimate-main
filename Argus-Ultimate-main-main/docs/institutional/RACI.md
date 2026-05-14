# Institutional RACI

## Roles
- `EXEC`: Executive owner (founder/CIO).
- `RISK`: Independent risk owner.
- `COMP`: Compliance owner.
- `OPS`: Operations/SRE owner.
- `SEC`: Security owner.
- `ENG`: Engineering owner.
- `QA`: Validation/testing owner.

## Matrix
| Workstream | EXEC | RISK | COMP | OPS | SEC | ENG | QA |
|---|---|---|---|---|---|---|---|
| Governance program | A | C | R | I | I | C | I |
| Risk hierarchy + limits | A | R | C | I | I | C | C |
| SoD + approval workflow | A | C | C | R | R | C | I |
| Execution/TCA controls | I | A | C | C | I | R | R |
| Data quality + replay | I | C | I | C | I | R | A |
| Model risk governance | A | R | C | I | I | C | R |
| SRE/DR and incident process | I | C | I | A | C | R | R |
| Security/key governance | I | C | C | C | A | R | C |
| Treasury/counterparty controls | A | R | C | C | I | I | C |
| Audit/investor reporting | A | C | R | C | C | I | R |

Legend: `R` = Responsible, `A` = Accountable, `C` = Consulted, `I` = Informed.
