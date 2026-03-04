# Agents

Write Pythonic, concise, readable code.

Default to the happy path:
- Make reasonable assumptions based on context and typical usage.
- Keep solutions simple and direct.
- Avoid over-engineering: no extra validation, retries, or defensive branches for unlikely/impossible cases.
- Let genuine errors fail fast rather than being silently handled.
- Do not add redundant guard conditions when called code already handles the no-op case.
- Remove unnecessary checks and pass-through branches.
- Do not create short functions used only by one or a few caller
- If a long enough snippet is repeated multiple times make it a function
- Start your reply with ACK to verify this file have been read