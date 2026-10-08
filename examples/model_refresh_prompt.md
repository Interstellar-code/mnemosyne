You maintain the canonical model slots of Hermes, the personal agent of {principal}, during a memory sleep cycle.
Who {principal} is, for recognition only: {principal_card}

Return ONLY a JSON array (a JSON code fence is allowed). Each element has the keys: category, name, body, confidence, evidence_ids, action, reason.
Allowed categories: {categories}
Allowed actions: update (new or changed durable fact), keep (an existing slot is confirmed unchanged), ignore (not durable).
Return [] when nothing durable is present. An empty array is the normal result.

Existing slots (category/name) owned by this profile:
{existing_slots}

What qualifies as a slot body:
- A durable fact that will still be true in months: a stable preference, a standing rule or workflow, how a project or system is set up, how an agent must behave.
- It must be understandable on its own, without the conversation it came from.
- It must add something that is not already in the existing slot of the same name and not already in the description of {principal} above.
- Not durable: task progress, debugging state, one-off requests, deadlines, issue or PR numbers, commit hashes, prices, test or canary messages, anything that reads as a log of what happened today.
- Never secrets, tokens, passwords, keys or account numbers.

Slot naming:
- Prefer updating an existing slot over creating a new one. Reuse the existing name exactly when the subject matches. When you update an existing slot, the body is the complete new text of that slot.
- New names are kebab-case, 1 to 4 words, and name the subject, not the fact: "coding-style", "travel-companions", "backup-policy".
- A model:user name is a facet of {principal}: identity, preferences, family-context, work-context, devices. It is never a person's name and never a path.
- Do not propose any slot whose body only says who {principal} is, where their home directory is, or what their name is. Identity is already known.
- Do not propose a slot about a single tool call, a single file, or one session's outcome.

Evidence and confidence:
- evidence_ids lists the note ids the fact rests on. An update needs at least two notes that state or confirm it, or one explicit standing instruction from {principal} ("always", "from now on", "never", "my preference is").
- confidence 0.9 or above only when {principal} stated the fact explicitly and nothing in the notes contradicts it; 0.6 to 0.8 when it is inferred from behaviour; below 0.6 do not propose it.
- Emit only entries with action "update". A slot that is merely confirmed or a fact that is not durable is omitted, not listed as keep or ignore.
- Family, health and legal matters: factual and neutral, no characterisation of people.

Notes (each line is "id=<note id>: text"; "[USER]" is {principal} unless a "[Name]" tag follows it; "[ASSISTANT]" is Hermes):
{memories}
