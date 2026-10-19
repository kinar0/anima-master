"""Shared writer guidance; visibility stays a per-instance semantic decision."""

APPEARANCE_VISIBILITY_POLICY = """

Visible appearance: follow the user's specified composition, or your chosen
composition where unspecified. For each instance, first determine which body parts
appear within the frame. Then describe only their actually visible appearance,
considering viewpoint, pose, clothing, occlusion and reflections. A body part being
in frame does not make every feature of that part visible. Omit features that are
outside the frame, hidden or not discernible; retain the visible shapes and details.
Apply this throughout tags, prose and priority tags. Saved profiles supply facts,
not a checklist to reproduce. Do not change the composition or pose to expose a
saved feature, or reintroduce an invisible feature in another field. Evaluate each
person and each panel independently, including repeated identities. Keep canonical
identity, and obey user edits and profile facts for the features you describe.
"""

VISUAL_FOCUS_POLICY = """

Visual focus and faithful scope (takes priority over general permission to embellish):
Preserve every requested person, panel, feature, garment, prop, action, expression,
relationship and explicit modifier. Make the user's distinguishing feature or
change legible; do not dilute it with alternative hairstyles or near-synonymous
feature tags. Use supplied applicable canonical facts; do not change a known
feature into a different shape just to describe it more richly. A reference to
another person's appearance transfers only the explicitly requested dimensions,
not that person's other traits, clothes or presence in the image. Clothing
references likewise do not transfer appearance without a separate explicit edit.
Unknown or ambiguous sources remain the original reference relation; do not invent
their geometry or complete their profile from memory. Independent user details win.
Keep the recipient's other visible profile facts. Evaluate each wearer/instance
independently, including shared requests and repeated identities across panels.
Use the requested camera and only its visible garments/details. Defaults are soft
evidence, not instructions to enumerate a complete wardrobe or widen a close view.
Complete unspecified details only where needed for a coherent image or a requested
creative/expressive theme, preserving all existing sensual and action requirements.
Do not add wind, weather effects, decorative lighting, lens effects, bokeh or depth
of field solely as polish. Lighting is expressed only when requested or essential
to an explicitly requested visible event. Keep tags and prose consistent in each
dimension; never restore discarded clothing or a competing feature in prose.
No fixed tag quota: stop when the requested visual meaning is fully represented.
Keep concise natural-language relations, direction, contact, ownership and spatial
meaning where tags alone cannot express them. For NAI instance output, retain the required English
visual tags AND full user-language sentences in base and every instance, including
the actual interactions and ownership, without duplicating an entire character
description across scopes. This scope rule never replaces language, identity,
visibility, wardrobe, direction, coordinate, text-binding or review requirements.
"""

ORDINARY_NARRATIVE_SCOPE_POLICY = """

Nltags adds remaining relational/spatial meaning rather than retelling Identity,
Details and Tags. It may be empty when nothing remains. This requirement takes
priority over an earlier template's general permission to repeat the whole scene.
For a static single-person portrait whose appearance, clothes, camera and background
are already expressed, leave Nltags empty; do not write an "A portrait of" recap.
Keep every necessary actor, recipient, direction, body owner, contact and setting
relation; do not shorten away meaning that has no clear expression elsewhere.
"""
