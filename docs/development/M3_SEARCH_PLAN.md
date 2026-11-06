# M3 search and library actions implementation plan

> **For agentic workers:** read `CLAUDE.md` first. Task-by-task, red/green tests, self
> review, then independent agent review. Root coordinates file leases and merges
> only after current-head received/responded Copilot, Windows CI and independent
> review. Steps use checkbox syntax. **Status: in progress — query-core checkpoint,
> 2026-11-06; UI attempts recorded, complete tasks/group/milestone acceptance pending.**
> R58 bounded UI/50ms checkpoint approved, query p95 FAILED; complete tasks/group/milestone remain open.
> Root issued GO
> after M2 A merge `53712c5beedafc2d40e2abf10c13362a1c3c9f78` and contracts merge
> `73d3cb384bd422de6899d8c7b761f3f3159ea3f1`. M3 front: `codex/m3-search`,
> `C:/Users/savar/.codex/worktrees/2710/here`, chat
> `01a1af04-e779-7363-be61-c840651b0b18`. Root retains leases/integration/merges.

**Goal:** A `SearchService` with typed queries, safe FTS5 access, stable pagination
and a fallback, plus a Qt library tab that lists, filters, searches and edits titles
without blocking the UI.
**Architecture:** `src/memory/search.py` reads the M2 catalog and index through the
repository; `src/ui/library.py` renders pages delivered by `BackgroundJobs` with
request ids and debounce. No SQL or FTS syntax leaves the memory package.
**Tech Stack:** Python 3.13, `sqlite3` with FTS5, Pydantic DTOs, PySide6, pytest,
pytest-qt; no new runtime dependency.
**Spec:** `docs/development/M3_SEARCH_SPEC.md`.
**Plan (Spanish):** `docs/MILESTONE_3_PLAN.md`.

## Global constraints

- Accepted foundation: PR #21 merged as `53712c5beedafc2d40e2abf10c13362a1c3c9f78`
  at 2026-11-06T03:22:06Z, from reviewed head
  `8a498e207812592ff84a0acc5ba4e9f392e02373`. Product source
  `c59ae331ebf124400ef3e476e5f0fcc029e83999` retains schema 2, marker 3,
  receipt 3 and R36–38, with R40/R43/R44 corrections inherited below. Independent
  reviews accepted C1–C5; the final C5 verdict approved source c59ae33 and committed
  documentation 3fdbd05. Received Copilot review 5795325438 on exact head 8a498e2
  confirmed no open issues; all five observations have individual replies/resolved
  threads. Complete Windows CI 42155415514 passed 980 cases/two wheel-environment
  skips in 307.46s; package CI 42155415501 passed on that same head. Fresh local
  installed-wheel distribution checks passed all four cases in 7.51s. These are
  automatic synthetic evidence, not hardware/provider/clean-install certification.
  Contract PR #22 merged as `73d3cb384bd422de6899d8c7b761f3f3159ea3f1` at
  2026-11-06T03:59:54Z from reviewed head `96f76dbbcb5ab4ba92866d9acf73c2673e140ee8`:
  independent approval, received/responded Copilot 5795547739, exact Windows tests
  42159238848 (980/two wheel skips/304.66s), package 42159238847 and guarded merge.
  Both fronts exist and have GO; these documentation gates do not accept B/M3 code.
- `src/memory/` imports neither Qt nor Typer; `src/ui/` never runs SQL or reads
  session files.
- User text never reaches `MATCH` unquoted; every token is quoted and escaped.
- Deleting meetings, historical revisions and, by default, partial revisions never
  appear in results.
- No provider call anywhere; everything works with `OPENAI_API_KEY` unset.
- Tests use synthetic fixtures and temporary roots; the 1,000/100,000 fixture is
  generated, never recorded.
- Touched files leave ruff clean and formatted; no new mypy errors in them (baseline
  is not clean, see `CLAUDE.md` §6).
- Root alone assigns exclusive file leases and performs merges. Shared
  `src/memory/models.py`, `repository.py`, `database.py` writes are serialized with
  M2 B; no alternate layout or duplicate helper is introduced to bypass a lease.
- Migrations, global docs, `pyproject.toml`, `uv.lock` and version changes have one writer at a
  time. Preserve migration 1/2 checksums; M3 A does not perform a milestone version bump.
- Reserve isolated execution windows for deadline-sensitive suites and benchmarks;
  no competing tests, builds or mypy during those runs. Independent work must honor
  the root resource lease as well as file ownership.

## Execution checkpoint — 2026-11-06

Task 1's independent query core is focused-verified and independently **APPROVED
CORE** at `e4d2469c31b6053d0f432151c9a254f620eab404`; complete Task 1 acceptance
is pending. Focus2 observed 177 passes/one opt-in benchmark skip/26.79s; XML
178/0 failures/0 errors/1 skip, internal 26.699s. Source/tests trees are
`63ce6ddb6b99f5df0deb6db7c7d5857ffe0783c7` /
`f9baed770a46679ea02e128eed1cd23c32d5f4c1`. `/root/task1_review`, independent of
RED/GREEN, found no confirmed Critical/Important/Minor core defect. Continue that
same assessment after actual remaining integrations/evidence; no global review
or final Task 1 approval is recorded.

Frozen R46 DTO/model/error handoffs are integrated and models returned to B.
Only the independent query/list/filter/FTS/hybrid/literal/cursor core is covered.
The later isolated R48 read supplement is available at original
`e0d083ae2aebb0ef1b1d428f6b3beba42ac9030f`, parent
`7d925472fedb6d71bc2ab64dd6cb2256b2801277` and UI-worktree sync
`b3628a705fde1afb69690bb72a0960829615d575`; models blob is
`7ac51f614e272d57479823fbcbb056844a924ca6` in both consumers. It adds coherent
canonical detail/participant reads and the same injected controller getter, no
B4 mutations/snapshots. Parent search.py is unchanged from e4d; its 177 passes and
APPROVED CORE review remain bound to e4d. Root reports an isolated probe and
XML122 without failures. Initial post-handoff M3 runtime was NOT_RUN; subsequent
focus5/6 establish focused synthetic R48 read-consumer evidence, with changes
requested at that stage by the independent UI review. Later accepted B4/title
consumer evidence below supersedes its earlier pending state. Coherent shared B5
R42 suppression, sole reindex facade
and real before-after/no-op equivalence remain pending.
Small synthetic cases do not establish the guarded 1,000/100,000 p95
or Qt heartbeat. Full suite, final integrated mypy and subsequent relevant checks remain open.
Final four-file Ruff check/format passed at e4d after mechanical formatting and an
explicit same-fixture re-export; no assertions/scenarios were weakened.

Task 2's original RED, setup errors and focus1/2 failures remain historical.
Focus3 exact `2389a7f` recorded nine failures/29 passes/18.14s (XML 18.058s).
Status probe exit0/2.973s/jobs idle confirmed enum-to-str transport; GREEN 21ee889
restored the canonical enum. RED a9e022e's public QListWidget helper integrated as
`39a2767`; focus4 on that head first reached real R48 semantic assertions:
three failures/35 passes/13.16s (XML 12.909s), missing roster/scoped speaker names
and Tab traversal. Keyboard probe exit0/2.84s/jobs idle proved default table Tab
cycled columns; disabling that navigation leaves the table for detail/editor.

GREEN `887f2152ae918d9b12e35e813c3150e7e222703a` consumes canonical roster and
exact revision/speaker assignments, reuses that mapping for passive targets and
fixes Tab. Focus5 at that exact clean HEAD passed 38/12.49s (XML 12.434s).
Focus6 ran HEAD 887 plus ONLY the formatter diff, passing 38/12.57s (XML 12.523s),
zero failures/errors/skips. It was not a run at later 825 HEAD. All nine pre-run
Git blobs stayed unchanged and match post-release style commit
`825f3e64570fdd1504050b4259a708376cfb88f6`; source/tests trees are
`15b4872ee5bb3f8d69c4db8f7373103f74bf903d` /
`60430875b7dc44d1b5c012a94d1dc7c86a197f3f`. Final limited nine-file Ruff check/
format passed after one E501/six format findings; only formatting changed six
files, no scenarios/assertions. Both actual R48 consumers complete all roster,
request-scoped speaker/passive-target/raw text/timing/custom-root assertions.
This is focused automatic verification only, not B4 mutation or B5 acceptance.

Original 825 changes-requested review and findings remain immutable. The four new
regressions on exact test-only `7044b12d0097164cea340e61846ba980d1420ac2` observed
four failures/28 deselected/zero errors/skips/12.70s (XML four/internal 12.653s)
before fixes. Same GREEN source-only 7e554f1 addresses accepted existing ingestion
refresh/invalidation/options, separate participant failure/retry/canonical ID
replacement and deduplicated shared-busy feedback; no tests changed. Exact clean
worker `7e554f17ff6e3b89ae95786aa81edec3696b30d4` then passed all 42 cases in
13.37s (XML 13.319s). Nine-file Ruff check/format passed with no edits/repeats.

Same `/root/task2_review` scoped 825..7e returns **APPROVED IMPLEMENTED UI CHECKPOINT**:
R1/R2/R3 ADDRESSED, no new confirmed Critical/Important/Minor. Complete Task 2/group
remain open. Historical parent `5a81c046a5c8550c609aa0a1d2191c8c97fc41ae` integrates
all nine exact UI and three R48 backend blobs, with search.py still reviewed e4d.
Parent source/tests trees are `ae2b4209b4466278a840bb6c8c0b5194071c941c` /
`7c1f2559162a71633e60734eed334eaa2f8baaf5`. This proves bounded UI/DTO contract
equivalence; the 42-case invocation ran at worker 7e, not parent 5a81. No integrated
full/mypy scan follows from it; that stage only prepared the later D1 comparison.

Coordinator released runtime immediately after the latest verification commands,
before reports/scoped review/parent integration; root ACKed the reserved host.
This docs stage has no imports/tests/probes/lint/runtime grant. The latest integrated
B4/types/legacy checkpoint below preserves the earlier source scopes and held gates.

### B4/types and display checkpoint — frozen 5db071a

Accepted B4 96e8 was integrated as c4cc8ca with exact backend and canonical model 7ac
plus unchanged R46. Independent title tests own 4e5869f/main adb0b70 use actual public
ingest_session and FTS. Initial mypy 79→87 adds 9/removes 1 errors; the earlier CRLF
preflight started zero scans. Same-GREEN truthful types e5ff86b/main 75be486 change
only contract/bridge/library, no suppression/config/B4/model/query edit. Verified
comparison reuses revalidated actual 96e8 baseline 65 (raw bytes/provenance, matching
Python 3.13.15/mypy 2.3.1/config 7143), scans changed candidate 67 once, and reports
79→78 errors, ZERO added records/errors, four removed (one error/three notes).
Raw exit 1 is inherited; wrapper 0 is the no-new-errors result for exact D1 only.

UI 44 including both actual current/stale title→version/evidence/FTS paths passes
14.42s/XML internal 14.385s at HEAD75 plus ONLY pinned test formatter diff. Post-release
ed8e5aa matches executed D1/tests 26bc/config 7143; it was not the invocation HEAD.
Same reviewer APPROVED the scoped typing/B4 title-consumer checkpoint at ed8.

Frozen B test-only 075 integrated f3f06bc: three repaired nodes actually fail 3/3.44s
(XML 3/internal 3.387s) for absent B5 purge hook/tombstone restore/delete command;
speaker precondition now passes and CLI help prevents the old false positive.
Separate expected B5 RED, not M3 regression or a corrected 100 aggregate. Legacy 33
at actual f3/D1 instead finds genuine blank-error regression: 3F/30P/10.44s,
XML 33/internal 9.406s. Same GREEN MainWindow-only 7c9f6d4→5db071a changes display
fallback, preserves nonempty characters/structured identity/reason, no tests.
Exact clean 5db/E3 bounded follow-up: 7P/1.92s, XML 7/internal 1.862s; one-file
Ruff/format pass without edits. Same reviewer APPROVED the bounded display fix.
That display window supplies seven-case/Ruff evidence only; the later preview
checkpoint adds no pytest/mypy/full run.

Frozen `5db071a2d034bda94ed1676d166bcce3aa7f7f0b` source/tests/config:
`e3ede58c32f5d848988b94dd8c4a5319512546c3` /
`35efb958f45c9b107f3639fe018c6f0df42b3b79` /
`7143d4deff5acb55e524baa5ee15c101bce9b3b5`. Mechanical sync into the existing docs
worktree preserves these bytes; no runtime here. Coordinator released immediately
after commands before reports/review. At that checkpoint R50–R52 were declared
only; later bounded R50 approval below supersedes that pending state only. No early
app/native adapter/stub/second service. Complete
Task1/Task2/group/M3 remain open: B5 guard/reindex/delete callback/equivalence,
guarded 1k/100k p95<500ms/Qt<50ms, final integrated full/mypy, final B5 preview/global review/
Copilot/CI/package/guarded merge/certification pending. No version bump.
`M3_SEARCH_EVIDENCE.json` retains exact commands/SHAs/artifacts and calibration;
all prior history, core e4 proof/review and acceptance/task blocks remain unchanged.

### Bounded synthetic preview checkpoint — d7009c7

Same E3/tests35/config7143 at actual main `d7009c724da4fb5eca01d8e28fcec56f6529bf3d`:
preview1 exit0/function passes but all text boxes → visual NOT_APPROVED unchanged.
Independent no-here offscreen QLabel probe confirms empty font families/unresolved
Sans Serif/uniform glyph1; process-local existing Segoe UI makes the same 700x80
sample legible with distinct glyphs. Raw isValid=true alone cannot prove rendering.
Preview2 registers/selects ONLY the pinned existing font at 9pt inside the synthetic
QApplication; original composition/assertions/sandbox/audit and source unchanged.
Exit0/entrypoint0, one synthetic row, public Return/raw segment/absent timing/editor,
no command/forbidden operation/worker/credential; list 533x452/detail 1100x800 legible.
Horizontal table scroll, clipped narrow combos and first heading (still clipped
expanded) remain geometry limits; opened detail has no overlap. Same reviewer
APPROVED BOUNDED PREVIEW EVIDENCE, no new finding/fix, not general layout/native/
DPI/accessibility/large-corpus or complete-task approval. Font sample PNGs are not
Biblioteca captures; original preview1 is not overwritten or relabeled.

Separate preview records bind all 3 actual commands/helpers/font/source/PNGs/exits;
no pytest counts/XML invented, timings are not p95/heartbeat. Coordinator released
immediately after EACH command before report/image inspection; root/B host now.
Current E3 evidence is bounded 7/Ruff plus this populated synthetic preview only;
no 33/44/mypy/full repeat, D1 result scopes unchanged. Final B5-integrated preview
and broader gates remain pending. At d700 R50 handoff was held on root P2; later
scoped technical R50 acceptance is below, no direct c9 or later native import,
stub/second service/native adapter or product/style/global-font/config/dependency
change. Complete Task1/Task2/group/M3 remain open, no certification/version bump.
### Bounded R50/R53 checkpoint — actual verified d8

Exact fbe technical scope is R50/reindex/query after H1/H2 only, not Task5/delete/
native/R52 acceptance. Merge 5bb preserves exact B source/canonical models+R46.
Actual new guarded51 RED at 4df: 51F/0E/S/9.10s, XML51/internal8.922; producer
prerequisites pass, reindex/no-op initially unreached. Same GREEN guard-only 75a→965
then 51P/12.30s/XML12.251 at HEAD 965+ONLY formatter literal join; later D6 exact
executed root be5/source bb4/tests 4a21 is not the invocation HEAD. Same Task1 reviewer
SPEC/quality APPROVED bounded immutable D6, zero C/I/M; original e4 review unchanged.
Real aliases/failclosed/Busy/one-context/close and sole B reindex/SECOND no-op now
execute, 3 meetings/3 segments/partial evidence/pages/cursor equivalence; only
missing-FTS capability simulated. TEMP connection-local/no main writes/JSON dep;
full actual FTS/schema digest retained plus B fingerprint; later measured costs below.

R53 separate RED at d21:10F/1 override-control P/0E/S/2.40s, XML11/internal2.337,
5 keyword TypeErrors + 5 missing-method AttributeErrors; trailing assertions unreached.
After same disjoint source authors, actual clean d8/source 0bea/tests de961/config 7143
passed 11/2.38s, XML11/internal2.344, two-file hygiene PASS/no edits. All callback/
identity/canonical-error/zero-resource/readonly/actual same-memory/custom-root
composition assertions complete; actual real report 1 meeting/2 segments, synthetic
3/7 counts identity only. Separate unchanged-source regression, NO R53 repeat:
228P/1 benchmark skip/38.53s, XML229/internal38.253=140 search+37 fallback+51 guard.
Original177 now have current-source functional regression, not full/type/performance.
Same T1 APPROVED R53/R50 interaction and same T2 APPROVED optional UI composition,
bounded SPEC/quality at immutable `d8ec92b3d6aca9ec872b8f3d77d446f7b686f91a`, zero C/I/M.

Scale c16→70af was test-only89-line addition,excluded/unbound from old 11/228;
late notice/author timestamp never imply retrocoverage. Later first functional
CONTROL 1P/4.09s, XML1/internal4.045 at HEAD 70af+ONLY staged safe import-sort/format.
Subsequent 396 exact executed root E913/tests AE2 is not invocation HEAD; source0bea/
config7143 unchanged. Real 220-ID/SQLite-variable-limit 64, full filtered first limit2
listing/hybrid pages, three visible IDs/segments/continuations/closed readers pass.
Complete-only fixture, not large-set partial/fallback/resource/performance proof.
Initial I001/format findings retained as hygiene, not semantic RED; assertions intact.
Scale SAME-T1 bounded control/evidence/docs supplement APPROVED, D1 addressed at
fcdd579; initial Minor retained as history, no full-task inference.

At that functional checkpoint, 23 separate pytest records preserved 17 old + 51 RED + 51 GREEN + R53 RED + 11 GREEN +
regression 228 + scale CONTROL 1,preview 3 unchanged. Prior d8/de961 and HEAD965+format
bindings remain exact; no 52 aggregate or run at post-release 396HEAD is invented.
Coordinator GLOBAL RELEASE immediately after last command BEFORE XML/reports/
review/commit; all 17 protected raw files unchanged. The benchmark helper/existing test and new UI benchmark were preparation ONLY
at that stage; their first actual results are recorded below, no production/B changes. Guarded1k/100k p95/Qt, final types/full/UI/
preview, delete ACK/native/R52, wider B/global/hosted/Copilot/CI/package/merge/
certification/tasks/groups remain pending; no M3B regeneration or version bump.
Composite checkboxes remain unchecked: no partial evidence closes their remaining
clauses. Group/hosted gates, certification and a 0.4.0 bump are not claimed.

### Historical0bea performance checkpoint — query FAILED / UI INCOMPLETE

Actual HEAD0c4098a plus ONLY staged3-file mechanical formatter delta executed two
separate nodes on unchanged source0bea/tests13c5/config7143. Executed entire root
89c628f equals post-release `ec800c271b6a5dafd2ab269dd7cc28edba42db5d`, not an invocation
at ec800HEAD. Initial check0/format-check1; only format3, final check/format0.
All79 raw bindings were unchanged before UI and post-release; no assertion, fixture,
source/config/threshold or between-command edit. Full exact bindings/commands in JSON.

Query 1F/72.70s, XML1/0E/S/internal72.625: valid setup5 warmups plus40 complete
correct public queries, inclusive p95 **1303.482595ms > required <500ms**. Real
1,000 meetings/100,000 segments/98,901 FTS documents,220 B records/221 resolved IDs
including alias/44 visible hits. Windows11 AMD64 Family25 Model80/16 CPUs,
Python3.13.15; setup14.1765594s outside warm intervals. All actual lock/registry/
alias/TEMP/full digest/page/DTO/close work included. Budget FAILED; cost driver not
profiled or fixed. Root's budget-only condition allowed the same-source UI node.

UI 1F/16.17s, XML1/0E/S/internal16.137: canonical Busy library-page requests3/4,
zero SearchPage callbacks/zero completed or rendered queries of8. Final50ms assert
UNREACHED.115 ticks/578.164ms/max11.2069ms are partial diagnostic ONLY, heartbeat
INCOMPLETE. Owned workers drained; retained Qt warnings do not establish cause.
SAME authors prepare static cost/Busy timeline diagnoses; no source/harness fix,
ignored Busy, relaxed threshold or new runtime concession is inferred.

25 separate pytest records preserve all prior23 and entire preview3. Scale/evidence/
docs supplement APPROVED with D1 addressed, without broad/source/performance review
restart. GLOBAL RELEASE immediately after final UI command BEFORE XML/metrics/
analysis/reports/review/style commit. Current full/final types/UI/preview, wider
B5/delete ACK/nonresurrection/native/R52/global/hosted/Copilot/CI/package/merge,
certification/Task1/Task2/group/M3 remain open. Thresholds, acceptance/task checklists,
Closing/M3B overlap and version remain unchanged; docs-role runtime0.

### R55 generation checkpoint — functional approved / p95 still FAILED

The actual two-call causal diagnostic at old source0bea/known synthetic input194
is recorded in `guarded-query-cost-profile-report.md`, not as pytest cases/XML.
It supports targeting repeated generation serialization/framing; nested inclusive
costs are not summed. R55 source e783→4cbc changes only internal generation v2:
fixed256-row batches, all actual ordered SQL inputs and real B fingerprint/snapshot
preserved. This bounds rows, not bytes; large fields remain costly. Static v1→v2
cursor-rejection reasoning is not a newly executed cross-version test.

ALL THREE actual commands ran HEAD `b9f16b02740fb436c9ff69d7db21e2b90fe70568`
plus ONLY staged new-test safe import-sort/mechanical format. Executed root
`e59253d05f38716118a2d543bddd663e89987106`, source
`e198e954658eefa15222476adfdfc06db797b8a8`, tests
`a2acb894f96523dae6e871fc65a9e13d6c7eb4e5`, config7143. Post-release
`1cb2442a69d47a14afc92662288ab579958ab00c` equals that entire executed tree;
it is NOT the invocation HEAD. All80 raw bindings remained identical.

New controls:10P/4.79s, XML10/internal4.614s; first-pass CONTROL, no invented RED.
Actual819-document nonmatching/suppressed FTS-only mutations at255/256/257/final
reject cursors before MATCH; fresh traversal and SECOND real B no-op are covered.
A separate existing regression passed240/46.02s, XML240/internal45.744s:
140 search+37 fallback+51 guard+11 facade+1 scale. New10 were not repeated;
benchmark/opt-in were absent from both functional commands. Fresh R55 evidence
is separate from the unchanged historical0bea records.

Then unchanged query benchmark:1F/52.56s, XML1/internal51.151s. Valid setup11.5409553s
outside warm intervals,5 warmups and40 complete correct public calls on the same
1k/100k/98,901FTS/220records/221IDs/44hits workload. Inclusive p95
**902.4132849997841ms still FAILS unchanged <500ms**. Raw40 and machine/input
context remain in the XML/measurements artifact. The old1303.482595ms failure is
preserved; a lower new measurement is not causal speedup certification.

SAME Task1 FINAL bounded R55 SPEC/functional-quality review is APPROVED, no
confirmed Critical/Important/Minor finding; complete Task1 NOT APPROVED and
performance FAILED. Only new-test hygiene passed after retained I001/format
findings; source Ruff/current mypy/full remain unrun. At that freeze R55 UI observer/heartbeat were
NOT_RUN; their later observed invocation is recorded below, without old0bea transfer. Diagnostic2 was PREPARED ONLY at this freeze; its later executed diagnostic is
recorded below. At that stage passive UI observer still needed execution; no second source
optimization is authorized.

28 separate pytest records preserve all25 prior and the entire preview3. GLOBAL
RELEASE immediately followed the last process BEFORE XML/metrics/reports/review/
commit. All broader B/native/delete/UI/preview/types/full/global/hosted/task/group/
M3 gates stay open, with unchanged thresholds/criteria/version and no certification.
After that 28-record freeze, a separate grant executed diagnostic2 at actual clean
HEAD1cb/sourcee198/tests a2ac/config7143 on known synthetic input205: one real
control5206.9568ms plus one cProfile call2744.1124ms, exit0. Generation1683.1486ms
inclusive/self132.6825 includes788 fetchmany724.7501ms,784 direct dumps about473ms
and5 SQL about298ms; B registry1008.0762ms includes220 path checks about869ms.
Nested inclusive timings are not added. Two unequal calls establish no warm p95,
causal speedup,host-load cause or measurement invalidity. GLOBAL RELEASE preceded
analysis; raw80/DB/registry/source/tests/config unchanged. See the final addendum
of `guarded-query-cost-diagnosis-preparation.md` and its pinned summary in JSON.
This is diagnostic evidence, no new pytest case or acceptance. At that diagnostic2 checkpoint UI observer was
NOT_RUN; later observation follows. Valid40 p95902.413285>500 remains FAILED, with no new source/B fix authority.

After a separate observer-only grant, actual clean HEADf9/sourcee198/tests a2ac/
config7143 ran the wrapped observer526/manifest10123 once:1F/41.36s,
XML1/internal41.290s, exit1. This is OBSERVATIONAL_ONLY_NOT_PERFORMANCE_ACCEPTANCE.
The canonical causal summary records participant request2 owning the lock, obsolete
initial MeetingQuery3 Busy ignored by the view but aborting the all-errors collector,
and current SearchQuery4 Busy accepted during finally, leaving Busy/Retry and0rows.
Zero SearchPages/zero completed of8; partial359ticks/1818.0331ms/max27.155ms is NOT
a heartbeat pass and final50ms assertion was unreached; workers drained. See
`guarded-ui-worker-observer-r55-1-causal-summary.json` and the final addendum of
`guarded-ui-worker-observer-preparation.md`; no93-event table is duplicated here.
GLOBAL RELEASE preceded analysis,raw79/Git unchanged. No retroactive0bea cause or
uninstrumented timing inference; uninstrumented heartbeat acceptance NOT_RUN.
The ledger now retains all28 prior records plus this separate observed invocation.
R57 later executed materializer31F once and conditional warmC4 at clean HEAD9406/
sourcee198/tests a2ac/config7143, both exit0: dedicated synthetic READY input,
raw81/DB/220registry/marker/Git invariant. Exactly5 warmups+control+observed call,
no pytest/cProfile/p95. Warmups2239–2690ms do not reproduce old822–1028ms;
observed wall/CPU2642.9028/2562.5ms,control3438.416/3046.875ms. Inclusive phases
and quantized CPU are not subtracted or used to infer host cause. See canonical
R57 diagnosis addendum/summary and READY/manifests pinned in JSON; no equality
with205 or result/digest cache. GLOBAL RELEASE preceded postchecks/analysis.
Existing p95902.413285>500 remains FAILED. That diagnostic supplied no source
fix/B transfer authorization; the later separate R61 source GO is recorded below.

### R58 worker verification — bounded approval and mechanical integration

At clean author-worker `2b66246198965eaca7a3df5d326837f4851b8942`, source
`12774a1e6702ab943934280d05d0de6897e8378d`/tests a2ac/config7143, two-file
Ruff/check-format passed without edits; unchanged UI44 passed16.36s/XML16.294.
The original UNINSTRUMENTED eight-query gate then passed1/23.56s/XML22.725:
8SearchPages/8render-drains,IDs3–10,errors[],workersdrained;1788ticks,
1790points/1789gaps,complete8941.6032ms window,max8.9216ms<50. Final strict
assertion reached and passed. Exact raw points/gaps remain in canonical
`r58-ui-measurements-1.json`; no gap table is duplicated here.

This pass belongs only to that source/context. Primary31e9/sourcee198 is not the
invocation. SAME Task2 FINAL bounded review APPROVED,no C/I/Minor. Root then
mechanically integrated ONLYlibrary/bridge62+ as38cff90dba7bd2914e44491ec60e8cf58771a2f4,
source12774/tests a2ac/config7143. ALL83Gitblobs match executed2b; rawEOL-only
differences are models/library/search_fixtures, normalized identical. Neither
83rawidentity nor invocation at38 is claimed.
All83 worker raw bindings remain unchanged; disclosed same-blob models/fixture
LFprimary/CRLFworker differences were preserved. GLOBAL RELEASE immediately after
session66172 exit preceded analysis/docs/Git; R57marker/DB unchanged.
Coordination is auxiliary(participants/detail)↔page only: page↔page,title/read,
cross-view and external ownership remain outside; actual Busy/Retry is retained.
31 separate records preserve all29 prior and preview3; no completeTask2/group/
hardware/types/full acceptance. Query p95902.413285>500 remains FAILED;
DF1ADA69 proposal is historical; current proposalA315/source-freeze addendum
records root-authorized isolated R61ab837 SOURCE_PREPARED/NOT_RUN, not integrated.
Independent byte/digest controls are preparing; no byte proof, speedup,500ms
acceptance or B wait is claimed.

## Dependencies and handoff during execution

Accepted A and contract-documentation merges permit task 1's independent
query/list/filter/fallback work alongside M2 B under root's recorded leases.
The root STATUS/dispatch brief and front progress report retain actual ownership;
the older local STATUS header is not the current execution checkpoint.

- Bind query DTO/read helpers to the accepted A schema, including nullable dates,
  `row_version`, active revision visibility, `segments_fts` and R31's
  `revision_audio_assets` identity/ordering. Audio association is retained for the
  target DTO; actual citation resolution stays in M3 B.
  Audio-only reservation may have `ingest_sources.revision_id=NULL`, no historical
  revisions and NULL capture dates preserved by durable source kind; listing,
  sorting/filtering and UI must not invent recording dates from processing times.
  R39 fixes list `started_at DESC NULLS LAST, meeting_id ASC`; search is rank ASC,
  then the same date/id keys and `segment_id ASC`. Unknown dates are included
  without date bounds, excluded if either bound is present; from inclusive/to
  exclusive. Bind every filter/key and apply them in SQL before LIMIT. Implement
  the SPEC's exact mixed-direction keyset branches, versioned query/filter-bound
  cursor with explicit NULL flag/value and no sentinel. Literal fallback uses the
  same date/null rules with uniform rank 0; FTS retains BM25. UI keeps unknown dates.
- Reconcile the existing A FTS capability probe/warning with the B maintenance API;
  observed A defines it in `service.py`. `database.py` is not automatically a probe
  implementation task. A lease is required if the accepted helper needs changing.
- Keep one FTS rebuild owner: B task 5 supplies reindex. Finalize the search facade's
  delegation/result contract when that B contract is accepted; do not add a second
  index implementation while waiting. Reindex integration/equivalence acceptance
  waits for that dependency, without blocking independent query work.
- Title-edit UI and optimistic conflict tests consume accepted B task 4
  `update_meeting`; never implement the B mutation inside search or count a fake as
  the real title-edit integration. Other worker/list/detail work is independent.
- R41 keeps active-complete-only persistent FTS under A/B5 ownership; SearchService
  supplies active-partial opt-in through read-only literal SQL UNION ALL, rank 0,
  with unchanged complete BM25 and a single global R39 page. Missing FTS makes both
  routes literal/rank 0. Use matching title/text fields and escaped LIKE tokens;
  no partial-index writer or second reindexer.
- B5's planned R42 guard suppresses logically accepted deletions before every
  B/M3 read, ordinary reimport and existing rebuild. Final query/detail integration
  waits for its accepted private adapter; independent query work remains possible.
  Do not parse the registry separately or postfilter a page. Invalid/unreadable
  registry is a typed recoverable fail-closed exposure error, never purge authority.
- Preserve R36's fresh-token/new-revision versus exact committed-proof replay,
  receipt-3 operation restoration and receipt-2 compatibility. Marker 3 carries
  token/source kind and `hashes`, including the writer's actual staged `session.json`
  and declared/locally consumed role artifacts before final replace; marker 1/2
  stays compatible. Accepted R38 retains unmatched sealed pending intent with
  existing recovery `error_summary`, no writer claim/old-output ingest/provider.
  Receipt validation isolates invalid siblings; private real-controller deferral
  preserves one cancelled capture publication/identity/audio/error at the existing
  cutover. These are accepted A behavior, not search fixes/public producer APIs.
  R40 retains ordinary callback rollback/cause and failed-rollback backups,
  unchanged BaseException/process-death behavior and a shared default-processor
  MemoryService/custom root; explicit processors stay compatible. Rejected
  completed replay warns without retry; successful/no-marker completed stays hidden.
  R43/R44 preserve valid metadata as bounded nonretryable warnings for malformed
  markers or unresolved legacy catalog owners, without media/credentials/provider
  work, mutation or invented identity/parent. Unknown ownership stops marker/error/
  audio inspection; stale materialization remains fail-closed. These are inherited
  safeguards, not new search or recovery scope.
  Keep native/catalog Busy/path guards fail-closed, the existing leases and
  typed `IngestResult.error_category` (`invalid_input|conflict|io`) and
  `OwnershipBusy(Busy)`; query adapters never parse warning prose.

## Front scopes and leases

| Scope | Owner/dependency after root dispatch | Exclusive write rule |
|---|---|---|
| `src/memory/search.py`; `tests/test_memory_search.py`, `test_memory_search_fallback.py`, `test_memory_search_benchmark.py` | M3 A independent query/list/filter/fallback | May overlap M2 B using accepted A contracts; shared DTO/read-helper writes still wait for leases. |
| `src/memory/models.py`, `repository.py`, `database.py` | Both fronts | Root grants per-file exclusive leases; one migrator, preserved migration-1/2 checksums. |
| `src/memory/service.py`, `artifacts.py`, application processing/recovery and B tests | M2 B persistence/lifecycle | M3 consumes accepted APIs; no copied persistence/probe or alternate module layout. |
| `MeetingPatch`, `update_meeting`, optimistic version/title-index consistency | B task 4 supplies; M3 title editor consumes | Real editor/conflict integration waits for accepted B4; fakes are preview/worker evidence only. |
| `MemoryService.reindex() -> ReindexReport` | B task 5 supplies; search facade delegates | One reindexer; integrate/equivalence-test the accepted B5 operation. |
| R42 suppression guard | B5 supplies; B/M3 read adapters consume | Accepted B5 adapter before real integration; no duplicate parser. Apply before cursor/LIMIT and invalidate pre-delete UI callbacks. |
| `src/ui/library.py`, `tests/test_ui_library.py` | M3 A independent view preparation | Lease common adapters before wiring; B acceptance dependencies still apply. |
| `src/ui/main_window.py`, `app.py`, `bridge.py`, `contract.py`, fake/preview, common UI tests, `src/application/controller.py`, `src/cli.py` | Shared integration or adapters | Root grants exact file leases before either front writes; B confirmation and M3 library wiring are serialized where shared. |
| Global/milestone docs, schema, dependency metadata and versions | Root coordination | Serialized across fronts; M3 A has no milestone bump. |

Reuse accepted prepared synthetic fixtures through one root-owned transfer; do not
regenerate or duplicate fixtures in each front. No new index/persistence/layout is
introduced for concurrency. Deadline suites, benchmarks/heartbeat and heavy builds
need root's quiet resource window without competing tests/builds/mypy.

## Review focus

- Operator-looking user input (`NEAR`, `*`, `^`, `-x`, `title:x`, unbalanced quotes).
- Listing keyset stability across inserts/deletes, and R45 search-generation
  rejection before using ranks changed by any corpus mutation, including nonmatches.
- Filters applied after `LIMIT` by mistake.
- A late response from a superseded request replacing the current result.
- Conflict handling on title edit with a stale `row_version`.
- FTS5-missing path leaving the library unusable.
- NULL dates/active revision on reserved audio-only lineage being replaced by
  processing dates, invented history or an assumed transcript during query/detail.
- A/B FTS capability or reindex logic being duplicated; titles/index becoming stale
  after a B edit; deletion appearing on later pages after a concurrent B operation.
- Shared-file or schema writes overlapping between fronts, and benchmark/deadline
  results measured while other verification is running.
- Opt-in partial hits absent from complete-only FTS, matched title tokens missing
  from literal routes, wildcard input interpreted as syntax, or partials changing
  after reindex. Require the real R41 hybrid/global-page contract.
- R42 barrier accepted while SQL still shows the meeting, DB loss restoring old
  receipts, or an old UI callback re-showing deletion; fail closed on bad registry.

### Task 1: SearchService, queries, pagination and fallback

**Files:** `src/memory/search.py` (new), `src/memory/models.py` (M2, add query/page
DTOs and `CitationTarget` type), `src/memory/repository.py` (M2, read helpers),
`src/memory/database.py` (shared; change only if the accepted contract needs it,
under root lease); tests in
`tests/test_memory_search.py`, `tests/test_memory_search_fallback.py`,
`tests/test_memory_search_benchmark.py` (marked slow, skipped by default).

**Interfaces:** `SearchService(connection_factory)` with
`list_meetings(query: MeetingQuery) -> MeetingPage`,
`search(query: SearchQuery) -> SearchPage | MeetingPage`, `reindex() -> ReindexReport`,
`capabilities() -> SearchCapabilities`; pure helpers `normalize_query_text(str) -> str`
and `build_fts_expression(str) -> str`; typed errors `InvalidInput` (bad cursor or
limit) reusing M2 error types. `SearchHit` carries `meeting_id`, `revision_id`,
`segment_id`, `snippet`, `start_ms`, `end_ms`, `timing_source`, `partial`.
The planned search `reindex` facade delegates to the accepted B maintenance
operation; its exact adapter/result is a dependency to reconcile, not a new writer.
Use the existing accepted FTS probe before adding any capability implementation.
Ruling 47 fixes the empty normalized search branch: delegate to `list_meetings`
with the same filters, limit and cursor and return its `MeetingPage`; nonempty
search returns `SearchPage`. Do not invent a hit/segment or expand visibility for
an audio-only reservation. The UI still dispatches empty text directly to listing,
whose date/ID cursor remains outside the R45 search generation.

- [ ] Write failing tests for `build_fts_expression`: accents, case, quotes, two-letter
  token, operator-looking tokens, unbalanced quotes; assert the raw string never
  appears in the SQL text (capture through a tracing connection).
  Cover literal token matching across the same title/segment fields, with `%`, `_`
  and escape characters treated literally in both partial and missing-FTS routes.
- [ ] Implement normalization and expression building.
- [ ] Write failing tests for listing and search: empty query lists by date then id;
  filters before limit; listing cursor round trip across two pages with an insert in between;
  malformed cursor → `InvalidInput`; `deleting`, historical and partial exclusion;
  `include_partial` badge flag. Include unknown-date audio-only lineage and a
  reservation without a revision; retain unknown fields. Assert list/search exact
  R39 ordering across equal dates, NULL and non-NULL cursor boundaries and equal
  ranks; from inclusion/to exclusion and exclusion of NULL under either date
  bound; identical date rules in FTS/fallback; mismatched cursor query/filter/order
  and malformed NULL flags return `InvalidInput`. Cover listing inserts/deletes
  between pages without OFFSET, sentinels, duplicate rows or invented dates.
  Under R45, stable-corpus ranked pages retain their complete R39 continuation;
  a nonmatching insert, deletion, title/revision/filter-relation change or accepted
  suppression from another connection rejects an old search cursor with structured
  `InvalidInput(corpus_changed)` before returning items. A fresh cursor-free query
  returns the correct current ordering without duplicates/suppressed hits. Check
  reproducible generation across fresh service/connection instances, no hit-only
  generation, no cross-connection data_version shortcut and all FTS/hybrid/fallback
  routes. Existing listing insert/NULL controls must continue without invalidation.
  Mixed complete/partial matches must cross the global LIMIT/rank/date-NULL keyset
  correctly; complete ranks retain BM25, partials are literal rank 0 and badged.
  Assert no historical/deleting/suppressed hits, no per-branch truncation and
  include_partial/backend cursor mismatch rejection. Literal snippets are actual
  text, at most 160 characters, with only measured/absent timing information.
- [ ] Implement queries with `bm25()` ordering, filters in SQL, opaque cursors and
  bounded `snippet()`.
  Bind search cursors to R45's deterministic complete-corpus/visibility generation
  in the same short read snapshot as the page; reject changed generations before
  keyset evaluation. Default to existing metadata/immutable-revision digest, with
  the coherent B5 suppression snapshot; no migration, persistent cache, extra writer
  or connection-local counter treated as a global generation. Measure the unchanged
  p95 budget with this actual generation work included.
  Prove metadata covers the actual logical FTS documents, including filtered or
  suppressed rows still affecting statistics and index-only maintenance changes.
  Canonically order/encode inputs and hold one explicit read transaction through
  generation and full page fetch, retaining existing bounded database handling.
  Add R41's read-only active-partial literal branch only when opted in; UNION ALL
  before global R39 cursor/LIMIT, with common filters/suppression applied first.
  Consume B5's accepted R42 guard through its private adapter before real read
  integration; one coherent suppression snapshot includes pending barriers whose
  SQL hide transaction failed. Never duplicate persistent FTS or registry parsing.
- [ ] Write failing tests for the FTS5-missing path and `reindex()` equivalence;
  reuse the accepted capability probe and implement literal fallback with `degraded`
  flag. Integrate/delegate reindex only after B task 5's contract is accepted, then
  run equivalence tests against its actual index maintenance.
  Compare complete-only and include_partial pages before/after B5 reindex; the
  persistent index remains complete-only and the partial read branch is unchanged.
  Simulate missing FTS for both literal branches/rank 0. Inject invalid/inaccessible
  R42 registry and assert typed recoverable fail-closed reads; rebuild from a
  durable pending/complete barrier must never expose a surviving receipt/FTS row.
- [ ] Generate the 1,000/100,000 synthetic fixture in a slow test; record p95 on the
  documented machine in the report; keep the test opt-in in CI.
- [ ] Run focused tests and the full offscreen suite; self-review; commit as
  "memory: search service, pagination and fallback".

### Task 2: Library tab, workers, filters, editing and empty states

**Files:** `src/ui/library.py` (new), `src/ui/main_window.py`, `src/ui/app.py`,
`src/ui/bridge.py`, `src/ui/contract.py` (extend `VisualController` with library
operations), `src/ui/fake_controller.py`, `src/ui/preview.py`; tests in
`tests/test_ui_library.py`, `tests/test_ui_lifecycle.py`.

**Interfaces:** `LibraryView(QWidget)` with signals `meeting_selected(meeting_id)`,
`segment_selected(CitationTarget)`; `VisualController` gains `list_meetings`,
`search`, `get_meeting`, `update_meeting`, `list_participants`;
`BackgroundJobs.submit` is reused with a
`request_id` and a `kind` per query type; the fake controller serves synthetic pages
so the preview runs without a database.
The real `update_meeting` adapter/title editor depends on accepted M2 B task 4;
root reserves any common UI/contract file against B's concurrent GUI integration.
Ruling 48 pins B4's `get_meeting`/`list_participants` read contracts, the canonical
detail's participant/link/assignment tuples, and read-only
`ApplicationController.memory_service` access to the same injected backend. Bind
the real adapter to that accepted handoff; no UI SQL, competing detail DTO or
inferred speaker identity. Unit widget fakes exercise rendering/request transport
only and cannot close B4/B5 integration or suppression acceptance.

- [ ] Write failing Qt tests: Biblioteca tab exists next to capture; typing five
  characters within 200 ms issues one query; an older delayed response is discarded;
  closing the detail drops its callback; both empty states and their actions.
  Queue an older result before a logically accepted B deletion, deliver it after
  the guard changes, and prove request invalidation prevents row/detail resurrection.
  Deliver structured `corpus_changed` on a continuation and assert the existing
  request/loading path clears that cursor, requests page one and replaces old pages;
  a superseded callback cannot append stale hits or trigger provider work.
- [ ] Implement the tab, search field with debounce, filters, list model and detail
  pane fed only by DTOs from the controller.
- [ ] Write failing tests for title editing: stale `row_version` shows conflict and
  keeps the stored title; current version updates and refreshes the list.
- [ ] Implement title editing through accepted B task-4 `update_meeting`, preserving
  its error categories, version precondition and FTS maintenance contract.
- [ ] Write failing accessibility tests: tab order across search, filters, list,
  detail and editor; accessible names present; contrast check in both themes using
  the existing palette tokens.
- [ ] Extend `fake_controller.py` and `preview.py` so `here-gui-preview` shows a
  populated library with synthetic meetings.
- [ ] Run focused Qt tests and the full offscreen suite; self-review; independent
  agent review; commit as "ui: library tab with search, filters and editing".

## Closing the group

- Update `README.md` (library tab), `ARCHITECTURE.md` (search service and request
  flow), `docs/MILESTONE_3_PLAN.md` (criteria 1–10, 17–18 status), `ACCEPTANCE.md`
  (M3 responsiveness row with the benchmark run) and `WHATS_NEW.md`.
- Open one PR with the two task commits, self-review and independent agent review in
  the description. Root integrates any required accepted B changes, verifies the
  exact head and merges only after received/responded Copilot and green Windows CI
  plus independent review. No unmet title-edit/reindex dependency is marked complete.
  M3 B may start from frozen/reviewed A with A's PR open once the required
  interfaces and contract-documentation PRs are accepted; rebase and final
  integration follow A's merge. Milestone criteria remain unchanged. This does not
  start M3 B or duplicate its existing preparation.
- Root aligns the M3 milestone agenda with ruling 35 in the contract docs PR
  without changing acceptance criteria. After M2 B merges, M5 B may advance with
  its claims hook waiting for M4 B; other group dependencies remain in STATUS.
  The accepted documentation merges and root GO launched the current two fronts;
  subsequent integrations and group closure still require the gates above.
