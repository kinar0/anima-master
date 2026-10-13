# llm-nai-toolbox attribution

Upstream project: https://github.com/sadlywish/llm-nai-toolbox

Upstream package author: **lyf** (package metadata). Release inspected: **0.9.8**.
The upstream README and package both declare **MIT**. Their original contents,
including the upstream project's own source acknowledgements, are retained in
`README.upstream.md` and `package.upstream.json`.

This plugin imports the Chinese aliases of general tags from the user's copy of
`resources/tagdb/tags_index_v2.json`. It does not import Toolbox's runtime code,
character profiles, artist database, credentials, or model API client. The import
manifest records the input SHA-256, extraction rules, additions, conflicts and
ambiguities. The translations remain editable vocabulary references.

2026-10-13 maintenance reuse: `third_party/llm_nai_toolbox/vocabulary_review_catalog.json`
contains a selected subset of general aliases from `tags_index_v2.json`,
definitions/traps/contrasts from `tag_gloss.json`, and categories from
`tag_browse.json`. This is separate review evidence, not an applied generation
mapping library. Source version, file SHA-256 values, local validation and scope
are recorded in that catalog and `vocabulary_quality/round_02/`. No Toolbox
runtime code, character features, artist records or model client were reused in
this maintenance feature. The original author, MIT declaration and upstream
acknowledgements above remain preserved.

Neither the provided source tree nor the release includes a separate project
copyright/license-text file. No copyright year or holder notice has been
invented. The upstream author/license declaration is preserved above and in the
original files. The standard MIT permission text is supplied in `LICENSE-MIT.txt`.
If upstream publishes a more specific notice, preserve that notice with future
updates. Electron's license is not used as a substitute.

MIT text reference: https://opensource.org/license/mit
