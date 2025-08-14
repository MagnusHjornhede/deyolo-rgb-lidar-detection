# NOTICE — DEYOLO License Obligations (Thesis2025)

- This repository’s original code is licensed under **MIT** (see `LICENSE`).
- The project **uses DEYOLO** (chips96/DEYOLO), which is licensed under **GNU AGPL-3.0**.

## Be observant of the following (AGPL-3.0):
1. **Modified program distribution**  
   If any **DEYOLO code is modified** and the modified program is **distributed**, you must provide the **complete corresponding source** of that modified version to recipients.

2. **Modified program offered as a network service (AGPL §13)**  
   If a **modified** DEYOLO-based program is **run on a server** and users **interact with it remotely** (e.g., web/API), you must provide those users access to the **complete corresponding source** of that modified version.

3. **Combined works**  
   If DEYOLO code is **combined** with other code into a **single program** that you distribute, AGPL copyleft may extend to the combined work. Keep DEYOLO code clearly separated and retain its license/file headers.

4. **Unmodified external use**  
   If DEYOLO is used **unmodified** as an **external dependency** and you do **not** distribute a combined program or run a modified service, your original code remains under **MIT**.

## Practical checklist
- Keep upstream **copyright and license notices** for any DEYOLO files.
- If vendoring DEYOLO, include its **LICENSE** file (e.g., `external/DEYOLO/LICENSE`).
- Mark files with **SPDX** headers:
  - Our code: `SPDX-License-Identifier: MIT`
  - Vendored/derived DEYOLO files: `SPDX-License-Identifier: AGPL-3.0-only`
- Pretrained **weights/datasets** may carry **separate terms**; verify before redistribution.

For clarity only — not legal advice.
