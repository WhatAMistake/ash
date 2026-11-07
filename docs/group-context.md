# Group reply context

Group `/xyx` and `/lobster` posts (including automatic posts), and ordinary
group-chat answers, are indexed only after Telegram accepts them. The local
`data/group_context.sqlite3` database stores chat and message IDs, kind, text,
timestamp and a random episode ID. All delivered parts of a lobster share an
episode ID; each part retains its own Telegram message ID. Partial sends remain
usable. Telegram's reply author must be the bot; a lookup is scoped to the
current group. If the lookup misses, the reply's Telegram text or caption is
used when available. If Telegram supplies no text, Ash answers without that
context.

Each successful group-chat answer points to a separate history snapshot scoped
to the group and user. Replying to that answer restores that snapshot; replying
to an older answer forks it. A different user replying to the same answer gets
the visible answer as context but not the other user's history. An unrelated
mention or a reply to an automatic post starts a new discussion. Group chat
never uses a user's private WriterBot history. The model receives the selected
part, nearby episode parts and the user's reply as JSON **user content**, not
system instructions; the normal assistant turns are actual previous answers.

Records expire after seven days and are pruned on startup and access. Storage is
limited to 5,000 outgoing messages and 1,000 conversation snapshots, with at
most ten clipped turns per snapshot. Episode context sent to the model is also
bounded. The database contains public message text and user IDs: back it up or
delete it according to your privacy policy. It is excluded from git. If storage
is unavailable, replies still work using Telegram's available reply text, but
history and restart recovery are unavailable.
