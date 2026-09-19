#!/usr/bin/env python3
from policies import PSP_LOGICAL_COMPUTE, SEARCH_CHECKPOINTS, SURVIVORS, valid_policies


rows = valid_policies()
assert len(rows) == len(SEARCH_CHECKPOINTS) * len(SURVIVORS) == 57
assert {row.notation for row in rows} >= {"20\u21921@10", "9\u21923@10", "5\u21923@28"}
assert all(row.logical_compute <= 256 for row in rows)
assert all(row.M > row.K and row.M <= 25 for row in rows)
assert PSP_LOGICAL_COMPUTE == 256
print(f"budget check PASS: {len(rows)} fixed policies; PSP compute {PSP_LOGICAL_COMPUTE}")
