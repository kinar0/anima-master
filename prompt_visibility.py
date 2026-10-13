"""Shared writer guidance; visibility stays a per-instance semantic decision."""

APPEARANCE_VISIBILITY_POLICY = """

Visible appearance: saved character profiles describe identity facts, not a checklist
that must appear in this image. For every visible instance, use the final framing,
body and head/face orientation, occlusion and reflections to select only appearance
features actually visible in the image. Apply this to all tags and natural-language
fields, including identity, details, narrative and priority tags. Do not change the
requested pose or camera to expose a saved feature. A rear view with no face or eyes
visible needs no eye-colour tags; looking back, a visible side eye or a clear mirror
reflection may still show them. Fully closed or covered irises need no iris colour,
but retain visible closed-eye shapes, expressions and occlusion cues; lack of eye
highlights alone does not hide iris colour. Keep visible hair or ears even when
other facial features are hidden. Different people and different panels of the same
identity have independent visibility. Keep canonical identity and obey user edits
and profile facts for the features you do express. Omit invisible traits rather than
repeating them in another field. Never invent visibility evidence or extra viewers.
"""
