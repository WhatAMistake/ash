# Command and workflow audit

This describes the current working tree, including the earlier stages. Status
is **offline**: registration is checked for every public catalogue entry, but
neither live Telegram delivery nor live-model quality was tested. “Route” means
registration/catalogue coverage only; “behavior” means at least some relevant
mocked or offline behavior is covered, not that every branch is tested.
Model-backed commands depend on the configured provider and may fail or vary.

## Verification

- Full working-tree regression: `python -m pytest -q` — 171 passed, one
  `PytestReturnNotNoneWarning` from `scripts/test_changelog.py::test_changelog`.
- An isolated audit-only staged snapshot was also tested: 105 passed with the
  same warning. It does not contain the earlier-stage uncommitted tests.
- `python -m compileall -q src tests` and `git diff --check` passed.
- The audit was performed with earlier-stage changes present but uncommitted.
  The audit-only commit deliberately does not include those changes; its
  behavior and this table depend on the complete working tree. The group
  continuation and idea-callback scope fixes are intertwined with that
  earlier uncommitted implementation and remain in the working tree. Live
  Telegram, Telegram Business, broadcast delivery and model output were not
  exercised.

| Command | Purpose | Status | Tests | Limitations |
|---|---|---|---|---|
| `/start` | Welcome and keyboard | Registered | Route | Enables quotes for new users; also clears current workflow |
| `/help` | Grouped command guide | Registered | Route | The Telegram menu lists the full public catalogue |
| `/lang` | Show/set language | Registered | Route | ru/en; preference is per user |
| `/switchlang` | Toggle language | Registered | Route | Preference is per user |
| `/reset` | Reset session, workflow and stats | Registered | Serialization | Stats are shared across chats for the user |
| `/block` | Creative block | Registered | Behavior | Model; no tool-specific discussion state |
| `/develop` | Develop an idea | Registered | Route | Model; no tool-specific discussion state |
| `/character` | Character development | Registered | Route | Model; no tool-specific discussion state |
| `/dialogue` | Dialogue help | Registered | Route | Model; no tool-specific discussion state |
| `/prompt` | Random writing prompt | Registered | Route | Depends on prompt data |
| `/idea` | Random idea, optional reference | Registered | Behavior | Reference uses model; choices expire/restart |
| `/feedback` | Constructive critique | Registered | Behavior | Minimum 20 characters |
| `/roast` | Sharp ironic roast | Registered | Route | Minimum 50 characters; model |
| `/praise` | Reasoned praise | Registered | Route | Minimum 50 characters; model |
| `/style` | Style analysis | Registered | Route | Minimum 20 characters; model |
| `/corrector` | Proofread | Registered | Behavior | Minimum 3 characters; formatting limitations |
| `/editor` | Clichés and alternatives | Registered | Behavior | Minimum 5 characters; model |
| `/methodique` | Craft-book analysis | Registered | Behavior | Minimum 5 characters; RAG sources may be unavailable |
| `/methodichque` | Legacy alias for `/methodique` | Registered | Filter/argument parsing | Kept for compatibility |
| `/summary` | Summary and format selection | Registered | Behavior | Minimum 10 characters; model |
| `/upload` | Request a document | Registered | Route; extraction tests | TXT/DOCX/PDF only; extraction and size limits in README |
| `/count_me` | Count and record words/characters | Registered | Route | Counts belong to the user, not the chat |
| `/stats` | Show writing statistics | Registered | Route | User-wide, persisted locally |
| `/done` | Process draft or exit discussion | Registered | Behavior | Summary requires a subsequent format selection |
| `/cancel` | Cancel current workflow | Registered | Behavior | Does not reset chat history or stats |
| `/lobster` | Lobster monologue | Registered | Behavior | Model-backed; group auto-posts also exist |
| `/xyx` | Хух | Registered | Behavior | Also posted automatically in active groups |
| `/pun` | Pun on replied-to or last message | Registered | Behavior | Needs a suitable prior message |
| `/cry_baby` | Crying interaction | Registered | Route | Next message completes interaction |
| `/cite` | Writer quote | Registered | Behavior | Uses the local quote corpus |
| `/cite_on` | Enable quotes | Registered | Route | Quotes are checked during activity, not a timed scheduler |
| `/cite_off` | Disable quotes | Registered | Route | Per-user preference |
| `/cite_when` | Quote timing | Registered | Route | Approximate; depends on subsequent activity |
| `/dev_feedback` | Message developer | Registered | Route | Sends to `ADMIN_ID`; requires configuration |
| `/admin` | Confirmed broadcast | Registered | Changelog broadcast tests only | Admin only; manual broadcast lacks direct behavior tests |
| `/debug` | Environment status | Registered | No dedicated test | Admin only |
| `/confo_enable37` | Toggle group access | Registered | No dedicated test | Admin only; intentionally not in public menu |

## Interactions and residual risks

- Group tool workflows and writer sessions are scoped by group and user;
  ordinary group conversations use the separate reply-context store. Language,
  activity, error cooldown, quote preferences and statistics remain per user.
  Group continuation requires a reply or mention. The main reply keyboard and
  `/idea` inline choices are different Telegram markup types; callback tests
  cover ownership, expiration, duplicate delivery and parallel users.
  `/reset` does not delete the persisted group reply-context snapshots; their
  retention and deletion are described in `group-context.md`.
- Business secretary has separate update handlers, sessions and execution keys;
  existing offline tests cover selected isolation and disabled-connection cases,
  not a live Telegram Business session. Broadcast confirmation remains separate
  from writing tools; administrative group confirmation still requires a reply
  or mention.
- Unsupported file extensions are silently ignored. The menu update is
  best-effort; a Telegram API failure leaves the previous menu in place.
- Main reply-keyboard labels are matched before general text and can start a
  command while a draft is pending; entering a new tool may replace that draft.
  Document selection and model-error recovery have partial, not exhaustive,
  coverage. All in-memory workflow state is lost on restart.

## Candidates for approval, not removed

- `_handle_button` has no call sites; explicit button handlers are registered
  instead. Consider removing it only after confirming no external integrations.
- `_handle_message` contains a second changelog-update branch and a keyboard
  trigger branch after earlier routing; parts appear unreachable with normal
  dispatcher order. Review unusual direct calls and admin recovery before
  removal.
- `/methodichque` is a historical spelling alias, not a conflict. Keep it for
  existing links and users unless a compatibility policy explicitly changes.
- The six editorial entry handlers repeat document/inline validation; a future
  shared dispatcher could reduce coupling, but should preserve thresholds,
  document delivery and error recovery. No public commands are proposed for
  removal.
